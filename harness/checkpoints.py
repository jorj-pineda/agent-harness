"""Explicit bounded file-content checkpoints, separate from session/run evidence."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .workspace_review import REVIEW_IGNORE

MAX_CHECKPOINT_ENTRIES = 2000
MAX_CHECKPOINT_BYTES = 16_000_000


class CheckpointFile(BaseModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: int = Field(ge=0, le=0o7777)
    size: int = Field(ge=0, le=MAX_CHECKPOINT_BYTES)


class CheckpointSummary(BaseModel):
    checkpoint_id: str
    session_id: str
    kind: Literal["manual", "turn_before", "turn_after"] = "manual"
    run_id: str | None = None
    paired_before_id: str | None = None
    captured_at: datetime
    root: str
    root_identity: tuple[int, int]
    file_count: int = Field(ge=0, le=MAX_CHECKPOINT_ENTRIES)
    byte_count: int = Field(ge=0, le=MAX_CHECKPOINT_BYTES)
    excluded_patterns: tuple[str, ...] = REVIEW_IGNORE
    restore_supported: Literal[False] = False
    scope_note: str = (
        "Content capture; kind identifies manual capture or an admitted turn boundary. "
        "Saves eligible regular-file bytes and permission bits, excluding environment files "
        "and default ignored components. Empty directories, ownership, ACLs and extended "
        "attributes are not restorable; non-excluded symlinks/special files reject capture. "
        "Non-atomic against external writers; no resume, restore, revert "
        "or attribution permission. "
        "Saved source contents are plaintext in local SQLite, never returned by metadata routes."
    )


class CheckpointRecord(CheckpointSummary):
    files: dict[str, CheckpointFile] = Field(
        default_factory=dict, max_length=MAX_CHECKPOINT_ENTRIES
    )


@dataclass
class CapturedCheckpoint:
    record: CheckpointRecord
    contents: dict[str, bytes]


def capture_checkpoint(
    root: Path, session_id: str, *, protected_paths: tuple[Path, ...] = ()
) -> CapturedCheckpoint:
    """Fail closed before storing anything; cap bytes even if a file grows during read."""
    requested_root = root.absolute()
    root = root.resolve(strict=True)
    if root != requested_root:
        raise ValueError("Checkpoint root must remain a canonical directory, not a symlink.")
    if not root.is_dir():
        raise ValueError("Checkpoint workspace must be a directory.")
    if any(path.resolve().is_relative_to(root) for path in protected_paths):
        raise ValueError("Move harness storage outside the workspace before checkpoint capture.")
    initial_root = root.stat()
    files: dict[str, CheckpointFile] = {}
    contents: dict[str, bytes] = {}
    entries = 0
    byte_count = 0

    def walk_error(error: OSError) -> None:
        raise error

    for parent, directories, names in os.walk(root, followlinks=False, onerror=walk_error):
        current = Path(parent)
        directories[:] = sorted(
            name
            for name in directories
            if not any(fnmatch(name, pattern) for pattern in REVIEW_IGNORE)
        )
        names = sorted(
            name for name in names if not any(fnmatch(name, pattern) for pattern in REVIEW_IGNORE)
        )
        entries += len(directories) + len(names)
        if entries > MAX_CHECKPOINT_ENTRIES:
            raise ValueError("Checkpoint entry limit exceeded; no checkpoint saved.")
        for name in directories:
            if (current / name).is_symlink():
                raise ValueError("Checkpoint does not support non-excluded symlinks.")
        for name in names:
            path = current / name
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("Checkpoint does not support symlinks or escaped paths.")
            descriptor = os.open(
                path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            )
            with os.fdopen(descriptor, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError("Checkpoint does not support special files.")
                raw = stream.read(MAX_CHECKPOINT_BYTES - byte_count + 1)
                after = os.fstat(stream.fileno())
            if len(raw) + byte_count > MAX_CHECKPOINT_BYTES:
                raise ValueError("Checkpoint byte limit exceeded; no checkpoint saved.")
            named = path.lstat()
            if (
                (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_mode)
                != (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_mode)
                or len(raw) != after.st_size
                or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
            ):
                raise ValueError("File changed during checkpoint capture; retry explicitly.")
            relative = path.relative_to(root).as_posix()
            contents[relative] = raw
            files[relative] = CheckpointFile(
                sha256=hashlib.sha256(raw).hexdigest(),
                mode=stat.S_IMODE(before.st_mode),
                size=len(raw),
            )
            byte_count += len(raw)
    final_root = root.stat()
    identity = (initial_root.st_dev, initial_root.st_ino)
    if identity != (final_root.st_dev, final_root.st_ino):
        raise ValueError("Workspace root changed during checkpoint capture.")
    return CapturedCheckpoint(
        record=CheckpointRecord(
            checkpoint_id=uuid4().hex,
            session_id=session_id,
            captured_at=datetime.now(UTC),
            root=str(root),
            root_identity=identity,
            file_count=len(files),
            byte_count=byte_count,
            files=files,
        ),
        contents=contents,
    )
