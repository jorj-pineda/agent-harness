"""Code exploration and editing tools bound to a workspace.

Each tool resolves paths under the workspace root, logs invocations, and returns
structured results (repo-relative paths, line numbers) for downstream grounding.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from workspace import Workspace, WorkspaceError

from .base import Tool, ToolError
from .process import ProcessTimeoutError, command_environment, run_process
from .registry import ToolRegistry

log = logging.getLogger(__name__)

MAX_READ_BYTES = 256_000
MAX_READ_LINES = 500
MAX_GREP_HITS = 100
MAX_LIST_ENTRIES = 200
MAX_TREE_ENTRIES = 300
GREP_TIMEOUT_S = 10.0
GIT_TIMEOUT_S = 15.0
RUN_COMMAND_TIMEOUT_S = 120.0
MAX_WRITE_BYTES = 512_000
MAX_COMMAND_OUTPUT_CHARS = 32_000
MAX_GREP_OUTPUT_BYTES = 2_000_000

ALLOWED_ROOT_COMMANDS = frozenset({"pytest", "ruff", "mypy", "git", "python"})
ALLOWED_GIT_SUBCOMMANDS = frozenset({"diff", "status", "show"})


class ReadFileInput(BaseModel):
    path: str = Field(..., min_length=1, description="Repo-relative file path.")
    start_line: int | None = Field(
        default=None,
        ge=1,
        description="1-indexed start line (inclusive). Defaults to 1.",
    )
    end_line: int | None = Field(
        default=None,
        ge=1,
        description="1-indexed end line (inclusive). Defaults to file end.",
    )


class GrepRepoInput(BaseModel):
    pattern: str = Field(..., min_length=1, description="Regex pattern to search for.")
    path: str = Field(default=".", description="Repo-relative file or directory to search.")
    glob: str | None = Field(
        default=None,
        description="Optional glob filter (e.g. '*.py') applied to file names.",
    )


class ListDirInput(BaseModel):
    path: str = Field(default=".", description="Repo-relative directory path.")


class TreeInput(BaseModel):
    path: str = Field(default=".", description="Repo-relative directory path.")
    depth: int = Field(default=2, ge=1, le=6, description="Maximum directory depth.")


class GitDiffInput(BaseModel):
    path: str | None = Field(
        default=None,
        description="Optional repo-relative path to limit the diff.",
    )


class GitStatusInput(BaseModel):
    pass


class WriteFileInput(BaseModel):
    path: str = Field(..., min_length=1, description="Repo-relative file path to write.")
    content: str = Field(..., description="Full file contents (UTF-8).")


class ReplaceTextInput(BaseModel):
    path: str = Field(..., min_length=1, description="Repo-relative existing file path.")
    old_text: str = Field(..., min_length=1, description="Exact text to replace once.")
    new_text: str = Field(..., description="Replacement text; empty string deletes the match.")
    expected_sha256: str = Field(
        ...,
        pattern=r"^[0-9a-f]{64}$",
        description="Whole-file SHA-256 from read_file. Refuses edits after the file changes.",
    )


class RunCommandInput(BaseModel):
    argv: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Command argv list, e.g. ['pytest', 'test_calc.py']. First token must be "
            "allowlisted (pytest, ruff, mypy, git, python -m pytest)."
        ),
    )


class EmitPlanInput(BaseModel):
    steps: list[str] = Field(
        ...,
        min_length=1,
        description="Ordered plan steps before making filesystem edits.",
    )
    summary: str | None = Field(
        default=None,
        description="Optional one-line goal for the turn.",
    )


def _workspace_error(exc: WorkspaceError) -> ToolError:
    return ToolError(str(exc))


def build_code_tools(
    workspace: Workspace, *, command_env: Mapping[str, str] | None = None
) -> list[Tool]:
    """Build code tools bound to a specific workspace."""

    def read_file(args: ReadFileInput) -> dict[str, Any]:
        log.info("code_tool=read_file path=%s", args.path)
        try:
            target = workspace.resolve(args.path, must_exist=True)
        except WorkspaceError as exc:
            raise _workspace_error(exc) from exc
        if not target.is_file():
            raise ToolError(f"Not a file: {args.path}")

        start = args.start_line or 1
        if args.end_line is not None and args.end_line < start:
            raise ToolError("end_line must be >= start_line")
        if args.end_line is not None and args.end_line - start + 1 > MAX_READ_LINES:
            raise ToolError(f"Line range exceeds {MAX_READ_LINES} lines; narrow the request.")

        # Scan the full file for its line count and version hash, retaining only
        # the requested range. A large file can therefore be read in slices.
        digest = hashlib.sha256()
        selected: list[str] = []
        selected_bytes = 0
        total_lines = 0
        with target.open("rb") as stream:
            for raw_line in stream:
                digest.update(raw_line)
                total_lines += 1
                if total_lines < start or (
                    args.end_line is not None and total_lines > args.end_line
                ):
                    continue
                if len(selected) >= MAX_READ_LINES:
                    raise ToolError(
                        f"Line range exceeds {MAX_READ_LINES} lines; narrow the request."
                    )
                selected_bytes += len(raw_line)
                if selected_bytes > MAX_READ_BYTES:
                    raise ToolError(
                        f"Selected range exceeds {MAX_READ_BYTES} bytes; narrow the request."
                    )
                selected.append(
                    raw_line.decode("utf-8", errors="replace").removesuffix("\n").removesuffix("\r")
                )

        if total_lines and start > total_lines:
            raise ToolError(f"start_line {start} beyond file length {total_lines}")
        if not total_lines and start != 1:
            raise ToolError(f"start_line {start} beyond file length 0")
        end = min(args.end_line or total_lines, total_lines)
        return {
            "path": workspace.relative_str(target),
            "start_line": start,
            "end_line": end,
            "content": "\n".join(selected),
            "total_lines": total_lines,
            "sha256": digest.hexdigest(),
        }

    async def grep_repo(args: GrepRepoInput) -> list[dict[str, Any]]:
        log.info("code_tool=grep_repo pattern=%r path=%s", args.pattern, args.path)
        try:
            target = workspace.resolve(args.path, must_exist=True)
        except WorkspaceError as exc:
            raise _workspace_error(exc) from exc

        if shutil.which("rg"):
            return await _grep_ripgrep(workspace, target, args.pattern, args.glob, command_env)

        try:
            regex = re.compile(args.pattern)
        except re.error as exc:
            raise ToolError(f"Invalid regex pattern: {exc}") from exc
        return _grep_python(workspace, target, regex, args.glob)

    def list_dir(args: ListDirInput) -> list[dict[str, str]]:
        log.info("code_tool=list_dir path=%s", args.path)
        try:
            target = workspace.resolve(args.path, must_exist=True)
        except WorkspaceError as exc:
            raise _workspace_error(exc) from exc
        if not target.is_dir():
            raise ToolError(f"Not a directory: {args.path}")

        entries: list[dict[str, str]] = []
        for child in sorted(target.iterdir(), key=lambda p: p.name):
            rel = workspace.relative_str(child)
            if workspace.is_ignored(Path(rel)):
                continue
            kind = "dir" if child.is_dir() else "file"
            entries.append({"path": rel, "kind": kind})
            if len(entries) >= MAX_LIST_ENTRIES:
                break
        return entries

    def tree(args: TreeInput) -> list[dict[str, str]]:
        log.info("code_tool=tree path=%s depth=%d", args.path, args.depth)
        try:
            target = workspace.resolve(args.path, must_exist=True)
        except WorkspaceError as exc:
            raise _workspace_error(exc) from exc
        if not target.is_dir():
            raise ToolError(f"Not a directory: {args.path}")

        rows: list[dict[str, str]] = []
        for path, kind, depth in _walk_tree(workspace, target, args.depth):
            rows.append({"path": path, "kind": kind, "depth": str(depth)})
            if len(rows) >= MAX_TREE_ENTRIES:
                break
        return rows

    async def git_status(_args: GitStatusInput) -> dict[str, str | bool]:
        log.info("code_tool=git_status")
        return await _run_git(workspace, ["status", "--porcelain"], command_env)

    async def git_diff(args: GitDiffInput) -> dict[str, str | bool]:
        log.info("code_tool=git_diff path=%s", args.path)
        cmd = ["diff", "--no-color"]
        if args.path is not None:
            try:
                resolved = workspace.resolve(args.path, must_exist=False)
                cmd.append("--")
                cmd.append(workspace.relative_str(resolved))
            except WorkspaceError as exc:
                raise _workspace_error(exc) from exc
        return await _run_git(workspace, cmd, command_env)

    def write_file(args: WriteFileInput) -> dict[str, Any]:
        log.info("code_tool=write_file path=%s bytes=%d", args.path, len(args.content.encode()))
        try:
            target = workspace.resolve(args.path, must_exist=False)
        except WorkspaceError as exc:
            raise _workspace_error(exc) from exc
        if target.is_dir():
            raise ToolError(f"Refusing to write a directory path: {args.path}")

        encoded = args.content.encode("utf-8")
        if len(encoded) > MAX_WRITE_BYTES:
            raise ToolError(f"Content exceeds {MAX_WRITE_BYTES} bytes.")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded)
        rel = workspace.relative_str(target)
        return {"path": rel, "bytes_written": len(encoded)}

    def replace_text(args: ReplaceTextInput) -> dict[str, Any]:
        log.info("code_tool=replace_text path=%s", args.path)
        try:
            target = workspace.resolve(args.path, must_exist=True)
        except WorkspaceError as exc:
            raise _workspace_error(exc) from exc
        if not target.is_file():
            raise ToolError(f"Not a file: {args.path}")

        raw = target.read_bytes()
        current_sha256 = hashlib.sha256(raw).hexdigest()
        if current_sha256 != args.expected_sha256:
            raise ToolError(
                "File changed since read_file; read the file again before replacing text."
            )
        try:
            before = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"File is not valid UTF-8: {args.path}") from exc

        count = before.count(args.old_text)
        if count != 1:
            raise ToolError(_replace_mismatch_error(before, args.old_text, count))
        after = before.replace(args.old_text, args.new_text, 1)
        encoded = after.encode("utf-8")
        if len(encoded) > MAX_WRITE_BYTES:
            raise ToolError(f"Result exceeds {MAX_WRITE_BYTES} bytes.")
        if encoded == raw:
            raise ToolError("Replacement would not change the file.")

        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{workspace.relative_str(target)}",
                tofile=f"b/{workspace.relative_str(target)}",
            )
        )
        # Replace atomically so a failed write leaves the existing file intact.
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
                temporary_path = temporary.name
                temporary.write(encoded)
            os.chmod(temporary_path, target.stat().st_mode)
            # Catch changes made while preparing the replacement as well.
            if hashlib.sha256(target.read_bytes()).hexdigest() != args.expected_sha256:
                raise ToolError("File changed during replacement; read the file again.")
            os.replace(temporary_path, target)
            temporary_path = None
        finally:
            if temporary_path is not None:
                os.unlink(temporary_path)

        return {
            "path": workspace.relative_str(target),
            "bytes_written": len(encoded),
            "replacement_count": 1,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "diff": _truncate_output(diff),
        }

    async def run_command(args: RunCommandInput) -> dict[str, Any]:
        log.info("code_tool=run_command argv=%s", args.argv)
        argv = [str(token) for token in args.argv]
        _validate_command_argv(argv)
        executable = argv[0]
        if shutil.which(executable, path=command_environment(command_env).get("PATH")) is None:
            raise ToolError(f"Command not found on PATH: {executable}")

        try:
            proc = await run_process(
                argv,
                cwd=workspace.root,
                timeout_seconds=RUN_COMMAND_TIMEOUT_S,
                output_limit=MAX_COMMAND_OUTPUT_CHARS,
                env=command_env,
            )
        except ProcessTimeoutError as exc:
            raise ToolError(str(exc)) from exc

        return {
            "argv": argv,
            "exit_code": proc.exit_code,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "stdout_truncated": proc.stdout_truncated,
            "stderr_truncated": proc.stderr_truncated,
            "success": proc.exit_code == 0,
        }

    def emit_plan(args: EmitPlanInput) -> dict[str, Any]:
        log.info("code_tool=emit_plan steps=%d", len(args.steps))
        cleaned = [step.strip() for step in args.steps if step.strip()]
        if not cleaned:
            raise ToolError("emit_plan requires at least one non-empty step.")
        return {
            "steps": cleaned,
            "summary": args.summary,
            "step_count": len(cleaned),
        }

    return [
        Tool(
            name="read_file",
            description=(
                "Read a UTF-8 text file under the workspace. Returns repo-relative path, "
                "line range, and content. Use before editing or citing code."
            ),
            input_model=ReadFileInput,
            fn=read_file,
        ),
        Tool(
            name="grep_repo",
            description=(
                "Search the workspace for a regex pattern. Returns matching lines with "
                "repo-relative paths and line numbers."
            ),
            input_model=GrepRepoInput,
            fn=grep_repo,
        ),
        Tool(
            name="list_dir",
            description="List immediate children of a workspace directory (non-recursive).",
            input_model=ListDirInput,
            fn=list_dir,
        ),
        Tool(
            name="tree",
            description="List files and directories up to a bounded depth under a path.",
            input_model=TreeInput,
            fn=tree,
        ),
        Tool(
            name="git_status",
            description="Run `git status --porcelain` in the workspace (read-only).",
            input_model=GitStatusInput,
            fn=git_status,
        ),
        Tool(
            name="git_diff",
            description=(
                "Run `git diff` in the workspace (read-only). Optionally limit to one path."
            ),
            input_model=GitDiffInput,
            fn=git_diff,
        ),
        Tool(
            name="emit_plan",
            description=(
                "Record a structured plan (ordered steps) in the tool trace before editing "
                "files. No filesystem changes."
            ),
            input_model=EmitPlanInput,
            fn=emit_plan,
        ),
        Tool(
            name="write_file",
            description=(
                "Replace a UTF-8 text file under the workspace with new content. "
                "Creates parent directories as needed. Returns bytes written."
            ),
            input_model=WriteFileInput,
            fn=write_file,
        ),
        Tool(
            name="replace_text",
            description=(
                "Replace one exact text span in an existing UTF-8 file. Read the file first; "
                "pass its sha256 as expected_sha256. Refuses stale or ambiguous matches "
                "and returns the applied diff. Prefer this for small edits."
            ),
            input_model=ReplaceTextInput,
            fn=replace_text,
        ),
        Tool(
            name="run_command",
            description=(
                "Run an allowlisted verification command in the workspace root "
                "(pytest, ruff, mypy, git diff/status/show, python -m pytest). "
                "Returns exit_code, stdout, stderr, and success flag."
            ),
            input_model=RunCommandInput,
            fn=run_command,
            timeout_seconds=RUN_COMMAND_TIMEOUT_S + 5.0,
        ),
    ]


def register_code_tools(
    registry: ToolRegistry,
    *,
    workspace: Workspace,
    command_env: Mapping[str, str] | None = None,
) -> None:
    """Register code tools (read, write, verify) on the given registry."""
    for tool in build_code_tools(workspace, command_env=command_env):
        registry.register(tool)


def _replace_mismatch_error(text: str, old_text: str, count: int) -> str:
    if count > 1:
        return (
            f"Expected exactly one old_text match; found {count}. "
            "Include more surrounding lines in old_text so it matches once."
        )
    unescaped = old_text.replace("\\n", "\n")
    if "\n" not in old_text and unescaped != old_text and text.count(unescaped) == 1:
        return (
            "Expected exactly one old_text match; found 0. old_text contains the two "
            "characters backslash and n where the file has line breaks. Resend old_text "
            "and new_text with real line breaks."
        )
    return (
        "Expected exactly one old_text match; found 0. "
        "Copy old_text exactly from the latest read_file content."
    )


def _validate_command_argv(argv: list[str]) -> None:
    if not argv:
        raise ToolError("argv must not be empty")
    for token in argv:
        if not token or "\n" in token or "\x00" in token:
            raise ToolError("argv tokens must be non-empty single-line strings")

    root = argv[0]
    if root not in ALLOWED_ROOT_COMMANDS:
        hint = " Use 'python', not 'python3'." if root == "python3" else ""
        raise ToolError(
            f"Command not allowlisted: {root!r}.{hint} Allowed: pytest, ruff, mypy, "
            "git diff/status/show, python -m pytest."
        )

    if root == "git":
        if len(argv) < 2 or argv[1] not in ALLOWED_GIT_SUBCOMMANDS:
            raise ToolError("git subcommand not allowlisted (diff, status, show)")
    elif root == "python" and (len(argv) < 3 or argv[1] != "-m" or argv[2] != "pytest"):
        raise ToolError("python is only allowed as: python -m pytest ...")


def _truncate_output(text: str) -> str:
    if len(text) <= MAX_COMMAND_OUTPUT_CHARS:
        return text
    return text[:MAX_COMMAND_OUTPUT_CHARS] + "\n...(truncated)"


async def _grep_ripgrep(
    workspace: Workspace,
    target: Path,
    pattern: str,
    glob: str | None,
    command_env: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    cmd = [
        "rg",
        "--line-number",
        "--no-heading",
        "--with-filename",
        "--null",
        f"--max-count={MAX_GREP_HITS}",
    ]
    if glob:
        cmd.append(f"--glob={glob}")
    cmd.extend(["--", pattern, str(target)])

    try:
        proc = await run_process(
            cmd,
            cwd=workspace.root,
            timeout_seconds=GREP_TIMEOUT_S,
            output_limit=MAX_GREP_OUTPUT_BYTES,
            env=command_env,
        )
    except ProcessTimeoutError as exc:
        raise ToolError(str(exc)) from exc

    if proc.stdout_truncated:
        raise ToolError("grep_repo output exceeded limit; narrow the path or glob.")
    if proc.exit_code not in (0, 1):
        raise ToolError(f"rg failed: {proc.stderr.strip() or proc.stdout.strip()}")

    hits: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        if len(hits) >= MAX_GREP_HITS:
            break
        parsed = _parse_rg_line(workspace, line)
        if parsed is not None:
            hits.append(parsed)
    return hits


def _parse_rg_line(workspace: Workspace, line: str) -> dict[str, Any] | None:
    # rg --null --with-filename: path\0line:content
    file_path, separator, rest = line.partition("\x00")
    if not separator:
        return None
    line_no, separator, text = rest.partition(":")
    if not separator or not line_no.isdigit():
        return None
    abs_path = Path(file_path).resolve()
    try:
        rel = workspace.relative_str(abs_path)
    except ValueError:
        return None
    return {"path": rel, "line": int(line_no), "text": text}


def _grep_python(
    workspace: Workspace,
    target: Path,
    regex: re.Pattern[str],
    glob: str | None,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for file_path in _iter_files(workspace, target):
        if glob and not fnmatch(file_path.name, glob):
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                hits.append(
                    {
                        "path": workspace.relative_str(file_path),
                        "line": line_no,
                        "text": line,
                    }
                )
                if len(hits) >= MAX_GREP_HITS:
                    return hits
    return hits


def _iter_files(workspace: Workspace, target: Path) -> Iterator[Path]:
    if target.is_file():
        yield target
        return
    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace.root)
        if workspace.is_ignored(rel):
            continue
        yield path


def _walk_tree(
    workspace: Workspace,
    target: Path,
    max_depth: int,
) -> Iterator[tuple[str, str, int]]:
    root_depth = len(target.relative_to(workspace.root).parts)

    def _walk(current: Path, depth: int) -> Iterator[tuple[str, str, int]]:
        rel_depth = len(current.relative_to(workspace.root).parts) - root_depth
        if rel_depth > max_depth:
            return
        rel = workspace.relative_str(current)
        kind = "dir" if current.is_dir() else "file"
        yield rel, kind, rel_depth
        if current.is_dir() and rel_depth < max_depth:
            for child in sorted(current.iterdir(), key=lambda p: p.name):
                child_rel = child.relative_to(workspace.root)
                if workspace.is_ignored(child_rel):
                    continue
                yield from _walk(child, depth + 1)

    yield from _walk(target, 0)


async def _run_git(
    workspace: Workspace,
    git_args: list[str],
    command_env: Mapping[str, str] | None,
) -> dict[str, str | bool]:
    git_dir = workspace.root / ".git"
    if not git_dir.exists():
        raise ToolError("Not a git repository (no .git directory in workspace root).")

    cmd = ["git", "-C", str(workspace.root), *git_args]
    try:
        proc = await run_process(
            cmd,
            cwd=workspace.root,
            timeout_seconds=GIT_TIMEOUT_S,
            output_limit=MAX_COMMAND_OUTPUT_CHARS,
            env=command_env,
        )
    except ProcessTimeoutError as exc:
        raise ToolError(str(exc)) from exc

    if proc.exit_code not in (0, 1):
        raise ToolError(f"git failed: {proc.stderr.strip() or proc.stdout.strip()}")

    return {
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "stdout_truncated": proc.stdout_truncated,
        "stderr_truncated": proc.stderr_truncated,
    }
