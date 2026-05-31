"""Read-only code exploration tools bound to a jailed `Workspace`.

Each tool resolves paths under the workspace root, logs invocations, and returns
structured results (repo-relative paths, line numbers) for downstream grounding.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from collections.abc import Iterator
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from workspace import Workspace, WorkspaceError

from .base import Tool, ToolError
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


class RunCommandInput(BaseModel):
    argv: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Command argv list, e.g. ['pytest', 'test_calc.py']. First token must be "
            "allowlisted (pytest, ruff, mypy, git, python -m pytest)."
        ),
    )


def _workspace_error(exc: WorkspaceError) -> ToolError:
    return ToolError(str(exc))


def build_code_tools(workspace: Workspace) -> list[Tool]:
    """Build read-only code tools bound to a specific workspace."""

    def read_file(args: ReadFileInput) -> dict[str, Any]:
        log.info("code_tool=read_file path=%s", args.path)
        try:
            target = workspace.resolve(args.path, must_exist=True)
        except WorkspaceError as exc:
            raise _workspace_error(exc) from exc
        if not target.is_file():
            raise ToolError(f"Not a file: {args.path}")

        raw = target.read_bytes()
        if len(raw) > MAX_READ_BYTES:
            raise ToolError(
                f"File exceeds {MAX_READ_BYTES} bytes; narrow with start_line/end_line."
            )
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        start = args.start_line or 1
        end = args.end_line or len(lines)
        if start > len(lines):
            raise ToolError(f"start_line {start} beyond file length {len(lines)}")
        end = min(end, len(lines))
        if end < start:
            raise ToolError("end_line must be >= start_line")
        if end - start + 1 > MAX_READ_LINES:
            raise ToolError(f"Line range exceeds {MAX_READ_LINES} lines; narrow the request.")

        slice_lines = lines[start - 1 : end]
        return {
            "path": workspace.relative_str(target),
            "start_line": start,
            "end_line": end,
            "content": "\n".join(slice_lines),
            "total_lines": len(lines),
        }

    def grep_repo(args: GrepRepoInput) -> list[dict[str, Any]]:
        log.info("code_tool=grep_repo pattern=%r path=%s", args.pattern, args.path)
        try:
            target = workspace.resolve(args.path, must_exist=True)
        except WorkspaceError as exc:
            raise _workspace_error(exc) from exc

        if shutil.which("rg"):
            return _grep_ripgrep(workspace, target, args.pattern, args.glob)

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

    def git_status(_args: GitStatusInput) -> dict[str, str]:
        log.info("code_tool=git_status")
        return _run_git(workspace, ["status", "--porcelain"])

    def git_diff(args: GitDiffInput) -> dict[str, str]:
        log.info("code_tool=git_diff path=%s", args.path)
        cmd = ["diff", "--no-color"]
        if args.path is not None:
            try:
                resolved = workspace.resolve(args.path, must_exist=False)
                cmd.append("--")
                cmd.append(workspace.relative_str(resolved))
            except WorkspaceError as exc:
                raise _workspace_error(exc) from exc
        return _run_git(workspace, cmd)

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

    def run_command(args: RunCommandInput) -> dict[str, Any]:
        log.info("code_tool=run_command argv=%s", args.argv)
        argv = [str(token) for token in args.argv]
        _validate_command_argv(argv)
        executable = argv[0]
        if shutil.which(executable) is None:
            raise ToolError(f"Command not found on PATH: {executable}")

        try:
            proc = subprocess.run(  # noqa: S603 — argv validated against allowlist; cwd jailed
                argv,
                cwd=str(workspace.root),
                capture_output=True,
                text=True,
                timeout=RUN_COMMAND_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"run_command timed out after {RUN_COMMAND_TIMEOUT_S}s") from exc

        return {
            "argv": argv,
            "exit_code": proc.returncode,
            "stdout": _truncate_output(proc.stdout),
            "stderr": _truncate_output(proc.stderr),
            "success": proc.returncode == 0,
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
            name="write_file",
            description=(
                "Replace a UTF-8 text file under the workspace with new content. "
                "Creates parent directories as needed. Returns bytes written."
            ),
            input_model=WriteFileInput,
            fn=write_file,
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
        ),
    ]


def register_code_tools(registry: ToolRegistry, *, workspace: Workspace) -> None:
    """Register code tools (read, write, verify) on the given registry."""
    for tool in build_code_tools(workspace):
        registry.register(tool)


def _validate_command_argv(argv: list[str]) -> None:
    if not argv:
        raise ToolError("argv must not be empty")
    for token in argv:
        if not token or "\n" in token or "\x00" in token:
            raise ToolError("argv tokens must be non-empty single-line strings")

    root = argv[0]
    if root not in ALLOWED_ROOT_COMMANDS:
        raise ToolError(f"Command not allowlisted: {root!r}")

    if root == "git":
        if len(argv) < 2 or argv[1] not in ALLOWED_GIT_SUBCOMMANDS:
            raise ToolError("git subcommand not allowlisted (diff, status, show)")
    elif root == "python" and (len(argv) < 3 or argv[1] != "-m" or argv[2] != "pytest"):
        raise ToolError("python is only allowed as: python -m pytest ...")


def _truncate_output(text: str) -> str:
    if len(text) <= MAX_COMMAND_OUTPUT_CHARS:
        return text
    return text[:MAX_COMMAND_OUTPUT_CHARS] + "\n...(truncated)"


def _grep_ripgrep(
    workspace: Workspace,
    target: Path,
    pattern: str,
    glob: str | None,
) -> list[dict[str, Any]]:
    cmd = [
        "rg",
        "--line-number",
        "--no-heading",
        f"--max-count={MAX_GREP_HITS}",
        pattern,
        str(target),
    ]
    if glob:
        cmd.insert(-1, f"--glob={glob}")

    try:
        proc = subprocess.run(  # noqa: S603 — argv allowlist; pattern/path jailed under workspace
            cmd,
            capture_output=True,
            text=True,
            timeout=GREP_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"grep_repo timed out after {GREP_TIMEOUT_S}s") from exc

    if proc.returncode not in (0, 1):
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
    # rg --no-heading: path:line:content
    parts = line.split(":", 2)
    if len(parts) != 3:
        return None
    file_path, line_no, text = parts
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


def _run_git(workspace: Workspace, git_args: list[str]) -> dict[str, str]:
    git_dir = workspace.root / ".git"
    if not git_dir.exists():
        raise ToolError("Not a git repository (no .git directory in workspace root).")

    cmd = ["git", "-C", str(workspace.root), *git_args]
    try:
        proc = subprocess.run(  # noqa: S603 — argv allowlist; pattern/path jailed under workspace
            cmd,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"git timed out after {GIT_TIMEOUT_S}s") from exc

    if proc.returncode not in (0, 1):
        raise ToolError(f"git failed: {proc.stderr.strip() or proc.stdout.strip()}")

    return {"stdout": proc.stdout, "stderr": proc.stderr}
