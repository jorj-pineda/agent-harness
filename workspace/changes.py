"""Content snapshots of a workspace and the file changes between two snapshots.

A snapshot hashes every non-ignored file, so comparing two snapshots reports
content changes from any source: file tools, commands, or concurrent edits by
another process. It cannot attribute a change to its author.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from .core import WorkspaceError


class SnapshotLimitError(WorkspaceError):
    """Raised when a workspace is too large to snapshot within the given limits."""


@dataclass(frozen=True)
class WorkspaceChanges:
    added: tuple[str, ...]
    modified: tuple[str, ...]
    deleted: tuple[str, ...]


def _ignored(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch(name, pattern) for pattern in patterns)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(
    root: Path,
    *,
    ignore: Iterable[str],
    max_files: int = 0,
    max_bytes: int = 0,
) -> dict[str, str]:
    """Map repo-relative POSIX paths to content fingerprints.

    Symlinks are not followed; a link is fingerprinted by its target string.
    Unreadable files are fingerprinted by size and mtime. Nonzero limits raise
    `SnapshotLimitError` before hashing more than they allow.
    """
    patterns = tuple(ignore)
    fingerprints: dict[str, str] = {}
    total_bytes = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        dirs[:] = sorted(name for name in dirs if not _ignored(name, patterns))
        entries = [name for name in dirs if (parent / name).is_symlink()]
        entries += [name for name in files if not _ignored(name, patterns)]
        for name in sorted(entries):
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if max_files and len(fingerprints) >= max_files:
                raise SnapshotLimitError(f"Workspace has more than {max_files} tracked files")
            if path.is_symlink():
                fingerprints[relative] = f"symlink:{os.readlink(path)}"
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            if not path.is_file():
                continue
            total_bytes += stat.st_size
            if max_bytes and total_bytes > max_bytes:
                raise SnapshotLimitError(f"Workspace has more than {max_bytes} tracked bytes")
            try:
                fingerprints[relative] = _hash_file(path)
            except FileNotFoundError:
                continue
            except OSError:
                fingerprints[relative] = f"unreadable:{stat.st_size}:{stat.st_mtime_ns}"
    return fingerprints


def compare(before: dict[str, str], after: dict[str, str]) -> WorkspaceChanges:
    return WorkspaceChanges(
        added=tuple(sorted(after.keys() - before.keys())),
        modified=tuple(
            sorted(path for path in after.keys() & before.keys() if after[path] != before[path])
        ),
        deleted=tuple(sorted(before.keys() - after.keys())),
    )
