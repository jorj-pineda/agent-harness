"""Explicit single-file reversal of scoped checkpoint differences, never a Git reset."""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .checkpoints import CheckpointRecord
from .runs import RequestId


class FileRevertInput(BaseModel):
    user_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    path: str = Field(min_length=1, max_length=4096)
    request_id: RequestId

    @field_validator("path")
    @classmethod
    def canonical_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != value
            or value == "."
            or "\x00" in value
        ):
            raise ValueError("Revert path must be a canonical relative file path.")
        return value


class FileRevertRecord(BaseModel):
    revert_id: str = Field(default_factory=lambda: uuid4().hex)
    request: FileRevertInput
    before_checkpoint_id: str
    after_checkpoint_id: str
    status: Literal["pending", "applied", "rejected", "uncertain", "interrupted"] = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    reason: str | None = None
    scope_note: str = (
        "Explicit reversal of one observed checkpoint difference, not agent attribution or "
        "independent acceptance. Requires current eligible file bytes/mode to match the after "
        "point. Other paths and Git index are untouched. No whole-checkpoint restore or resume. "
        "External writers can race final checks; process-local admission is not isolation. "
        "Replacement changes inode identity and does not restore "
        "ownership/ACLs/extended attributes. "
        "Pending/interrupted/uncertain requests are never automatically executed or retried."
    )


class RevertConflict(ValueError):
    pass


@dataclass
class FileRevertOutcome:
    status: Literal["applied", "rejected", "uncertain"]
    reason: str


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mode,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _named_state(parent: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _check_named(parent: int, name: str, expected: os.stat_result | None) -> None:
    current = _named_state(parent, name)
    if (current is None) != (expected is None) or (
        current is not None and expected is not None and _identity(current) != _identity(expected)
    ):
        raise RevertConflict("Current file changed during revert preparation; no change made.")


def revert_file(
    before: CheckpointRecord, after: CheckpointRecord, path: str, before_content: bytes | None
) -> FileRevertOutcome:
    """Use bound directory descriptors, checking content/mode before a single mutation."""
    if os.name != "posix":
        return FileRevertOutcome(
            "rejected", "File revert currently requires POSIX directory descriptors."
        )
    descriptors: list[int] = []
    temporary: str | None = None
    parent = -1
    try:
        if (before.root, before.root_identity) != (after.root, after.root_identity):
            raise RevertConflict("Checkpoint roots differ; no change made.")
        original = before.files.get(path)
        expected = after.files.get(path)
        if original == expected:
            raise RevertConflict("No eligible checkpoint difference for this path.")
        if original is not None and (
            before_content is None
            or len(before_content) != original.size
            or hashlib.sha256(before_content).hexdigest() != original.sha256
        ):
            raise RevertConflict("Saved before content is unavailable or inconsistent.")
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        root = Path(before.root)
        if root.resolve(strict=True) != root:
            raise RevertConflict("Workspace root is no longer canonical.")
        root_fd = os.open(root, flags)
        descriptors.append(root_fd)
        root_stat = os.fstat(root_fd)
        if (root_stat.st_dev, root_stat.st_ino) != before.root_identity:
            raise RevertConflict("Workspace root identity changed; no change made.")
        parent = root_fd
        parts = PurePosixPath(path).parts
        if not parts or ".." in parts or PurePosixPath(path).is_absolute():
            raise RevertConflict("Invalid revert path.")
        for part in parts[:-1]:
            parent = os.open(part, flags, dir_fd=parent)
            descriptors.append(parent)
        name = parts[-1]
        current = _named_state(parent, name)
        if expected is None:
            if current is not None:
                raise RevertConflict("Deleted path now exists; later work preserved.")
        else:
            if current is None or not stat.S_ISREG(current.st_mode):
                raise RevertConflict("Current path is missing or not a regular file.")
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent,
            )
            with os.fdopen(descriptor, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(current):
                    raise RevertConflict("Current file changed before reading.")
                raw = stream.read(expected.size + 1)
                settled = os.fstat(stream.fileno())
            if (
                _identity(opened) != _identity(settled)
                or len(raw) != expected.size
                or hashlib.sha256(raw).hexdigest() != expected.sha256
                or stat.S_IMODE(settled.st_mode) != expected.mode
            ):
                raise RevertConflict(
                    "Current content or mode differs from after capture; later work preserved."
                )
            _check_named(parent, name, current)
        if original is not None:
            temporary = f".harness-revert-{uuid4().hex}"
            descriptor = os.open(
                temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=parent
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(before_content or b"")
                stream.flush()
                os.fchmod(stream.fileno(), original.mode)
            _check_named(parent, name, current)
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            temporary = None
        else:
            _check_named(parent, name, current)
            os.unlink(name, dir_fd=parent)
        return FileRevertOutcome(
            "applied", "Recorded before state applied to the selected path; run checks again."
        )
    except RevertConflict as exc:
        return FileRevertOutcome("rejected", str(exc))
    except (OSError, ValueError):
        return FileRevertOutcome(
            "uncertain",
            "File operation unavailable or failed; inspect file and journal before new work.",
        )
    finally:
        if temporary is not None and parent >= 0:
            with suppress(OSError):
                os.unlink(temporary, dir_fd=parent)
        for descriptor in reversed(descriptors):
            os.close(descriptor)
