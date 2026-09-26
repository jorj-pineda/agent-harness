"""Temporary repository copies and filesystem-level change tracking for evals.

The copy prevents accidental edits to the source repository. It does not isolate
executed code from the host OS; only trusted fixtures belong in this mode.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from .core import DEFAULT_IGNORE_GLOBS, Workspace, WorkspaceError

COPY_IGNORE_PATTERNS = (*DEFAULT_IGNORE_GLOBS, ".env", ".env.*")


def _ignored(name: str) -> bool:
    return any(fnmatch(name, pattern) for pattern in COPY_IGNORE_PATTERNS)


@dataclass(frozen=True)
class WorkspaceChanges:
    added: tuple[str, ...]
    modified: tuple[str, ...]
    deleted: tuple[str, ...]


def _files(root: Path) -> Iterator[Path]:
    for directory, dirs, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        dirs[:] = sorted(name for name in dirs if not _ignored(name))
        for name in dirs:
            if (parent / name).is_symlink():
                raise WorkspaceError(f"Symlinks are not supported in disposable workspaces: {name}")
        for name in sorted(files):
            if _ignored(name):
                continue
            path = parent / name
            if path.is_symlink():
                raise WorkspaceError(f"Symlinks are not supported in disposable workspaces: {path}")
            if path.is_file():
                yield path


def _snapshot(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in _files(root):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(64 * 1024), b""):
                digest.update(chunk)
        hashes[path.relative_to(root).as_posix()] = digest.hexdigest()
    return hashes


@dataclass
class DisposableWorkspace:
    workspace: Workspace
    baseline: dict[str, str]

    def changes(self) -> WorkspaceChanges:
        current = _snapshot(self.workspace.root)
        previous = self.baseline
        return WorkspaceChanges(
            added=tuple(sorted(current.keys() - previous.keys())),
            modified=tuple(
                sorted(
                    path
                    for path in current.keys() & previous.keys()
                    if current[path] != previous[path]
                )
            ),
            deleted=tuple(sorted(previous.keys() - current.keys())),
        )


@contextmanager
def disposable_workspace(source: Path) -> Iterator[DisposableWorkspace]:
    """Copy a trusted source tree, then delete the copy on context exit."""
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise WorkspaceError(f"Source is not a directory: {source}")
    # Check before copying so symlinks cannot lead copytree outside the source.
    for _ in _files(source):
        pass
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
