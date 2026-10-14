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
import stat
import tempfile
from collections.abc import Iterator, Mapping
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from workspace import Workspace, WorkspaceError
from workspace.npm import npm_test_prefix

from .base import Tool, ToolError
from .process import ProcessTimeoutError, command_environment, run_process
from .registry import ToolRegistry

log = logging.getLogger(__name__)

MAX_READ_BYTES = 256_000
MAX_READ_LINES = 500
MAX_READ_SCAN_BYTES = 16_000_000
MAX_GREP_HITS = 100
MAX_LIST_ENTRIES = 200
MAX_TREE_ENTRIES = 300
GREP_TIMEOUT_S = 10.0
GIT_TIMEOUT_S = 15.0
RUN_COMMAND_TIMEOUT_S = 120.0
MAX_WRITE_BYTES = 512_000
MAX_COMMAND_OUTPUT_CHARS = 32_000
MAX_GREP_OUTPUT_BYTES = 2_000_000
MAX_GREP_FILE_BYTES = 2_000_000
MAX_GREP_SCAN_BYTES = 16_000_000

ALLOWED_ROOT_COMMANDS = frozenset({"pytest", "ruff", "mypy", "git", "python", "python3", "npm"})
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
            "allowlisted (pytest, ruff, mypy, git, python -m pytest, python3 -m pytest, "
            "npm [--prefix <repo-relative-dir>] run test:run [-- <test args>])."
        ),
    )

    @field_validator("argv", mode="before")
    @classmethod
    def reject_string_argv(cls, value: object) -> object:
        if isinstance(value, str):
            raise ValueError(
                "Resend argv as a JSON array of separate strings, not a quoted array "
                'or shell command. Example: {"argv": ["python", "-m", "pytest"]}. '
                "Use the configured project check's exact tokens when provided."
            )
        return value


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


def _open_regular_file(target: Path) -> BinaryIO:
    """Reject final-component swaps to links/devices; parent path races remain."""
    descriptor = os.open(
        target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    try:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ToolError("File is no longer a regular file; inspect the path again.")
    except BaseException:
        stream.close()
        raise
    return stream


def _atomic_write_file(target: Path, content: bytes) -> None:
    """Publish complete content with a same-directory rename, leaving failures intact.

    New files retain the normal 0666/umask creation policy. Existing permission
    bits are copied; inode identity, hard links and other metadata are not preserved.
    This is neither a stale-content guard nor a power-loss durability guarantee.
    """
    mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else None
    temporary = target.parent / f".agent-harness-write-{uuid4().hex}"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666 if mode is None else mode & 0o777
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            if mode is not None:
                os.fchmod(stream.fileno(), mode)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


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
        scan_bytes = 0
        with _open_regular_file(target) as stream:
            while raw_line := stream.readline(MAX_READ_SCAN_BYTES - scan_bytes + 1):
                scan_bytes += len(raw_line)
                if scan_bytes > MAX_READ_SCAN_BYTES:
                    raise ToolError(
                        f"read_file whole-file scan exceeds {MAX_READ_SCAN_BYTES} bytes; "
                        "no complete hash or line count is available. Choose a smaller file; "
                        "narrowing the line range cannot reduce the whole-file scan."
                    )
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
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ToolError(
                        f"Selected line {total_lines} is not valid UTF-8; "
                        "read_file cannot return exact text for this range."
                    ) from exc
                selected.append(line)

        if total_lines and start > total_lines:
            raise ToolError(f"start_line {start} beyond file length {total_lines}")
        if not total_lines and start != 1:
            raise ToolError(f"start_line {start} beyond file length 0")
        end = min(args.end_line or total_lines, total_lines)
        content = "".join(selected)
        # Omit only the selected span's final terminator. Internal terminators
        # are exact source evidence for multiline replace_text preconditions.
        if content.endswith("\n"):
            content = content[:-1]
            if content.endswith("\r"):
                content = content[:-1]
        return {
            "path": workspace.relative_str(target),
            "start_line": start,
            "end_line": end,
            "content": content,
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
            kind = _entry_kind(child)
            entries.append({"path": rel, "kind": kind})
            if len(entries) > MAX_LIST_ENTRIES:
                raise ToolError(
                    f"list_dir exceeded {MAX_LIST_ENTRIES} entries; results are incomplete. "
                    "Choose a narrower directory or use tree with a shallow depth."
                )
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
            if len(rows) > MAX_TREE_ENTRIES:
                raise ToolError(
                    f"tree exceeded {MAX_TREE_ENTRIES} entries; results are incomplete. "
                    "Narrow the path or reduce the depth."
                )
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

        if target.exists() and not target.is_file():
            raise ToolError(f"Refusing to replace a non-regular file: {args.path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_file(target, encoded)
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
        if argv[0] == "npm":
            prefix = npm_test_prefix(argv)
            if prefix is None:
                raise ToolError("npm is only allowed as npm [--prefix DIR] run test:run [-- ARGS]")
            directory = workspace.resolve(prefix)
            if not directory.is_dir():
                raise ToolError("npm prefix must be a directory inside the workspace")
        # Use the existing Python execution path, even if PATH has a different python3.
        if argv[0] == "python3":
            argv[0] = "python"
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
                "line range, and content. Use before editing or citing code. "
                f"Whole-file hash/line scanning is limited to {MAX_READ_SCAN_BYTES} bytes."
            ),
            input_model=ReadFileInput,
            fn=read_file,
        ),
        Tool(
            name="grep_repo",
            description=(
                "Search the workspace for a regex pattern. Returns matching lines with "
                "repo-relative paths and line numbers, excluding workspace-ignored paths. "
                f"More than {MAX_GREP_HITS} matching lines returns an error; narrow the "
                "path, glob, or pattern and retry."
            ),
            input_model=GrepRepoInput,
            fn=grep_repo,
        ),
        Tool(
            name="list_dir",
            description=(
                "List immediate children of a workspace directory (non-recursive). "
                f"More than {MAX_LIST_ENTRIES} entries returns an explicit limit error. "
                "Symlinks are labeled as leaves."
            ),
            input_model=ListDirInput,
            fn=list_dir,
        ),
        Tool(
            name="tree",
            description=(
                "List files and directories up to the requested depth under a path. "
                f"More than {MAX_TREE_ENTRIES} entries returns an explicit limit error; "
                "narrow the path or depth. Symlinks are leaves, never traversed."
            ),
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
                "(pytest, ruff, mypy, git diff/status/show, python -m pytest, python3 -m pytest, "
                "npm [--prefix <repo-relative-dir>] run test:run [-- <test args>]). "
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
        raise ToolError(
            f"Command not allowlisted: {root!r}. Allowed: pytest, ruff, mypy, "
            "git diff/status/show, python -m pytest, python3 -m pytest, "
            "npm [--prefix DIR] run test:run [-- ARGS]."
        )

    if root == "npm" and npm_test_prefix(argv) is None:
        raise ToolError("npm is only allowed as npm [--prefix DIR] run test:run [-- ARGS]")
    if root == "git":
        if len(argv) < 2 or argv[1] not in ALLOWED_GIT_SUBCOMMANDS:
            raise ToolError("git subcommand not allowlisted (diff, status, show)")
    elif root in {"python", "python3"} and (
        len(argv) < 3 or argv[1] != "-m" or argv[2] != "pytest"
    ):
        raise ToolError(f"{root} is only allowed as: {root} -m pytest ...")


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
        # One extra matching line per file detects overflow without an unbounded
        # result count. The process output cap remains a separate preventive bound.
        f"--max-count={MAX_GREP_HITS + 1}",
    ]
    if glob:
        cmd.append(f"--glob={glob}")
    # Negative globs come last so a caller's positive glob cannot expose ignored
    # workspace components. Workspace ignores are literal names, not rg patterns.
    for name in workspace.ignore_globs:
        escaped = re.sub(r"([\\*?\[\]{}!])", r"\\\1", name)
        cmd.append(f"--glob=!{escaped}")
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
        parsed = _parse_rg_line(workspace, line)
        if parsed is not None:
            hits.append(parsed)
            if len(hits) > MAX_GREP_HITS:
                raise _grep_match_limit_error()
    return hits


def _parse_rg_line(workspace: Workspace, line: str) -> dict[str, Any] | None:
    # rg --null --with-filename: path\0line:content
    file_path, separator, rest = line.partition("\x00")
    if not separator:
        return None
    line_no, separator, text = rest.partition(":")
    if not separator or not line_no.isdigit():
        return None
    try:
        abs_path = workspace.resolve(file_path)
        rel = workspace.relative_str(abs_path)
    except (ValueError, WorkspaceError):
        return None
    return {"path": rel, "line": int(line_no), "text": text}


def _grep_python(
    workspace: Workspace,
    target: Path,
    regex: re.Pattern[str],
    glob: str | None,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    bytes_read = 0
    for file_path in _iter_files(workspace, target):
        if glob and not fnmatch(file_path.name, glob):
            continue
        try:
            allowance = min(MAX_GREP_FILE_BYTES, MAX_GREP_SCAN_BYTES - bytes_read)
            with file_path.open("rb") as stream:
                raw = stream.read(allowance + 1)
        except OSError as exc:
            raise ToolError(
                "grep_repo fallback could not read a file; results are incomplete. "
                "Narrow the path or glob and retry."
            ) from exc
        if len(raw) > allowance:
            raise ToolError(
                "grep_repo fallback exceeded its file or total scan byte limit; "
                "results are incomplete. Narrow the path or glob, or install ripgrep."
            )
        bytes_read += len(raw)
        text = raw.decode("utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                hits.append(
                    {
                        "path": workspace.relative_str(file_path),
                        "line": line_no,
                        "text": line,
                    }
                )
                if len(hits) > MAX_GREP_HITS:
                    raise _grep_match_limit_error()
    return hits


def _grep_match_limit_error() -> ToolError:
    return ToolError(
        f"grep_repo exceeded {MAX_GREP_HITS} matching lines; results are incomplete. "
        "Narrow the path, glob, or pattern and search again."
    )


def _iter_files(workspace: Workspace, target: Path) -> Iterator[Path]:
    if target.is_file():
        yield target
        return
    for parent, directories, files in os.walk(target, followlinks=False):
        current = Path(parent)
        directories[:] = sorted(
            name
            for name in directories
            if not workspace.is_ignored((current / name).relative_to(workspace.root))
            and not (current / name).is_symlink()
        )
        for name in sorted(files):
            try:
                path = workspace.resolve(str(current / name))
            except WorkspaceError:
                continue
            if path.is_file():
                yield path


def _entry_kind(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    return "dir" if path.is_dir() else "file"


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
        kind = _entry_kind(current)
        yield rel, kind, rel_depth
        if kind == "dir" and rel_depth < max_depth:
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
