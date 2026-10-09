"""Content snapshots of a workspace and the file changes between two snapshots.

A snapshot hashes every non-ignored file, so comparing two snapshots reports
content changes from any source: file tools, commands, or concurrent edits by
another process. It cannot attribute a change to its author.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import stat as stat_module
from collections.abc import Iterable
from dataclasses import dataclass, field
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


@dataclass
class TextCapture:
    """Bounded, ephemeral text retained from the same read as each fingerprint."""

    max_file_bytes: int
    max_bytes: int
    texts: dict[str, str] = field(default_factory=dict)
    omitted: dict[str, str] = field(default_factory=dict)
    bytes_used: int = 0


@dataclass(frozen=True)
class FileDiff:
    path: str
    diff: str = ""
    reason: str | None = None


def _ignored(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch(name, pattern) for pattern in patterns)


def _hash_file(path: Path, capture_bytes: int | None) -> tuple[str, bytes | None]:
    digest = hashlib.sha256()
    content: bytearray | None = bytearray() if capture_bytes is not None else None
    # Do not follow a file replaced by a symlink between enumeration and opening.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as stream:
        if not stat_module.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise OSError("Snapshot entry is no longer a regular file")
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(chunk)
            if content is not None and capture_bytes is not None:
                if len(content) + len(chunk) <= capture_bytes:
                    content.extend(chunk)
                else:
                    content = None
    return digest.hexdigest(), bytes(content) if content is not None else None


def snapshot(
    root: Path,
    *,
    ignore: Iterable[str],
    max_files: int = 0,
    max_bytes: int = 0,
    text_capture: TextCapture | None = None,
) -> dict[str, str]:
    """Map repo-relative POSIX paths to content fingerprints.

    Symlinks are not followed; a link is fingerprinted by its target string.
    Unreadable files are fingerprinted by size and mtime. Nonzero limits raise
    `SnapshotLimitError` before hashing more than they allow.
    An optional fresh `TextCapture` retains bounded UTF-8 content alongside hashes;
    omitted text does not prevent path tracking. This is not an atomic snapshot.
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
                if text_capture is not None:
                    text_capture.omitted[relative] = "Symlink content is not captured"
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
                limit: int | None = None
                if text_capture is not None:
                    if any(
                        _ignored(part, (".env", ".env.*")) for part in path.relative_to(root).parts
                    ):
                        text_capture.omitted[relative] = "Environment file content is not captured"
                    elif not hasattr(os, "O_NOFOLLOW"):
                        text_capture.omitted[relative] = "Safe text capture unavailable on this OS"
                    else:
                        limit = min(
                            text_capture.max_file_bytes,
                            text_capture.max_bytes - text_capture.bytes_used,
                        )
                fingerprint, content = _hash_file(path, limit)
                fingerprints[relative] = fingerprint
                if text_capture is not None and limit is not None:
                    if content is None:
                        text_capture.omitted[relative] = "Text capture byte limit exceeded"
                    else:
                        try:
                            text = content.decode("utf-8")
                            if "\0" in text:
                                raise UnicodeError("Binary content")
                        except UnicodeError:
                            text_capture.omitted[relative] = "Binary or non-UTF-8 content"
                        else:
                            text_capture.texts[relative] = text
                            text_capture.bytes_used += len(content)
            except FileNotFoundError:
                continue
            except OSError:
                fingerprints[relative] = f"unreadable:{stat.st_size}:{stat.st_mtime_ns}"
                if text_capture is not None:
                    text_capture.omitted[relative] = "File content could not be read safely"
    return fingerprints


def compare(before: dict[str, str], after: dict[str, str]) -> WorkspaceChanges:
    return WorkspaceChanges(
        added=tuple(sorted(after.keys() - before.keys())),
        modified=tuple(
            sorted(path for path in after.keys() & before.keys() if after[path] != before[path])
        ),
        deleted=tuple(sorted(before.keys() - after.keys())),
    )


def review_diffs(
    changes: WorkspaceChanges, before: TextCapture, after: TextCapture, *, max_bytes: int
) -> list[FileDiff]:
    """Return complete per-file unified diffs, or an explicit omission reason.

    This is a content review, not an apply/revert patch or an author attribution.
    An omitted large diff does not prevent later small files from being shown.
    """
    results: list[FileDiff] = []
    remaining = max_bytes
    added, deleted = set(changes.added), set(changes.deleted)
    for path in sorted((*changes.added, *changes.modified, *changes.deleted)):
        reason = before.omitted.get(path) or after.omitted.get(path)
        if reason:
            results.append(FileDiff(path=path, reason=reason))
            continue
        old = "" if path in added else before.texts[path]
        new = "" if path in deleted else after.texts[path]
        if not old and not new:
            results.append(FileDiff(path=path, reason="Empty file added or deleted"))
            continue
        old_label = "/dev/null" if path in added else f"a/{path}"
        new_label = "/dev/null" if path in deleted else f"b/{path}"
        labels = [
            json.dumps(label) if any(c in label for c in '\t\r\n"\\') else label
            for label in (old_label, new_label)
        ]
        parts: list[str] = []
        size = 0
        for line in difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=labels[0],
            tofile=labels[1],
        ):
            if not line.endswith("\n"):
                line += "\n\\ No newline at end of file\n"
            size += len(line.encode("utf-8"))
            if size > remaining:
                reason = "Diff output byte limit exceeded"
                break
            parts.append(line)
        if reason:
            results.append(FileDiff(path=path, reason=reason))
        else:
            remaining -= size
            results.append(FileDiff(path=path, diff="".join(parts)))
    return results
