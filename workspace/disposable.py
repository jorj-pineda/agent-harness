"""Temporary repository copies and filesystem-level change tracking for evals.

The copy prevents accidental edits to the source repository. It does not isolate
executed code from the host OS; only trusted fixtures belong in this mode.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from .changes import WorkspaceChanges, compare, snapshot
from .core import DEFAULT_IGNORE_GLOBS, Workspace, WorkspaceError

COPY_IGNORE_PATTERNS = (*DEFAULT_IGNORE_GLOBS, ".env", ".env.*")


def _ignored(name: str) -> bool:
    return any(fnmatch(name, pattern) for pattern in COPY_IGNORE_PATTERNS)


def _reject_symlinks(root: Path) -> None:
    for directory, dirs, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        dirs[:] = sorted(name for name in dirs if not _ignored(name))
        for name in (*dirs, *(name for name in files if not _ignored(name))):
            if (parent / name).is_symlink():
                raise WorkspaceError(
                    f"Symlinks are not supported in disposable workspaces: {parent / name}"
                )


def _snapshot(root: Path) -> dict[str, str]:
    return snapshot(root, ignore=COPY_IGNORE_PATTERNS)


@dataclass
class DisposableWorkspace:
    workspace: Workspace
    baseline: dict[str, str]

    def changes(self) -> WorkspaceChanges:
        return compare(self.baseline, _snapshot(self.workspace.root))


@contextmanager
def disposable_workspace(source: Path) -> Iterator[DisposableWorkspace]:
    """Copy a trusted source tree, then delete the copy on context exit."""
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise WorkspaceError(f"Source is not a directory: {source}")
    # Check before copying so symlinks cannot lead copytree outside the source.
    _reject_symlinks(source)
    with tempfile.TemporaryDirectory(prefix="agent_harness_eval_") as temporary:
        root = Path(temporary) / "repo"
        shutil.copytree(
            source,
            root,
            symlinks=True,
            ignore=shutil.ignore_patterns(*COPY_IGNORE_PATTERNS),
        )
        workspace = Workspace(root=root)
        yield DisposableWorkspace(workspace=workspace, baseline=_snapshot(root))
