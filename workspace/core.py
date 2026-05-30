"""Path jail for code tools — every file operation resolves under `Workspace.root`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class WorkspaceError(Exception):
    """Raised when a path escapes the sandbox or violates workspace rules."""


DEFAULT_IGNORE_GLOBS: tuple[str, ...] = (
    ".git",
    "node_modules",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
)


@dataclass(frozen=True)
class Workspace:
    """A jailed repository root for read/write code tools."""

    root: Path
    ignore_globs: tuple[str, ...] = field(default=DEFAULT_IGNORE_GLOBS)

    def __post_init__(self) -> None:
        resolved = self.root.expanduser().resolve()
        if not resolved.is_dir():
            raise WorkspaceError(f"Workspace root is not a directory: {resolved}")
        object.__setattr__(self, "root", resolved)

    def resolve(self, path: str = ".", *, must_exist: bool = True) -> Path:
        """Resolve `path` under the workspace root; reject escapes and ignored dirs."""
        raw = Path(path)
        candidate = raw.resolve() if raw.is_absolute() else (self.root / raw).resolve()

        try:
            relative = candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError(f"Path escapes workspace: {path}") from exc

        if self._is_ignored(relative):
            raise WorkspaceError(f"Path is ignored: {path}")

        if must_exist and not candidate.exists():
            raise WorkspaceError(f"Path not found: {path}")

        return candidate

    def relative_str(self, absolute: Path) -> str:
        """Return a repo-relative POSIX path string for tool results."""
        return absolute.relative_to(self.root).as_posix()

    def is_ignored(self, relative: Path) -> bool:
        """True when any path component matches an ignore glob."""
        return self._is_ignored(relative)

    def _is_ignored(self, relative: Path) -> bool:
        parts = relative.parts
        return any(part in self.ignore_globs for part in parts)
