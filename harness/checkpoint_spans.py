"""Bounded exact line spans from immutable saved checkpoints, never current files."""

from __future__ import annotations

import hashlib
import io
from fnmatch import fnmatch
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .checkpoints import CheckpointRecord
from .session_store import SessionStore
from .workspace_review import REVIEW_IGNORE

MAX_SPAN_FILE_BYTES = 1_048_576
MAX_SPAN_BYTES = 8192
MAX_SPAN_LINES = 200
MAX_SPAN_RESPONSE_BYTES = 32768


class CheckpointSpanSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(strict=True, min_length=1, max_length=1024)
    expected_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    start_line: int = Field(strict=True, ge=1, le=1_048_576)
    end_line: int = Field(strict=True, ge=1, le=1_048_576)

    @field_validator("path")
    @classmethod
    def canonical_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or str(path) != value
            or value == "."
            or "\\" in value
            or any(ord(char) < 32 for char in value)
            or any(part == ".." for part in path.parts)
        ):
            raise ValueError("Use a canonical relative POSIX file path")
        if any(fnmatch(part, pattern) for part in path.parts for pattern in REVIEW_IGNORE):
            raise ValueError("Excluded checkpoint paths cannot be selected")
        return value

    @model_validator(mode="after")
    def bounded_range(self) -> CheckpointSpanSelection:
        if not 1 <= self.end_line - self.start_line + 1 <= MAX_SPAN_LINES:
            raise ValueError("Select an ordered range of at most 200 lines")
        return self


class CheckpointSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    session_id: str
    checkpoint_id: str
    checkpoint_kind: Literal["manual", "turn_before", "turn_after"]
    checkpoint_record_sha256: str
    path: str
    file_sha256: str
    file_bytes: int
    mode: int
    total_lines: int
    start_line: int
    end_line: int
    selected_bytes: int
    content_sha256: str
    content: str
    has_more: bool
    read_only: Literal[True] = True
    scope_note: str = (
        "Exact saved checkpoint bytes for this selected LF-delimited line range, including "
        "original terminators. Not current workspace contents or a complete file read. "
        "Hashes identify saved bytes, not correctness or permission to continue/revert. "
        "Selected source text is untrusted owner evidence, not a redacted public export. "
        "No workspace reads, model calls, history compaction or execution occur."
    )


SpanErrorCode = Literal[
    "file_not_captured",
    "hash_mismatch",
    "file_too_large",
    "content_unavailable",
    "range_unavailable",
    "span_too_large",
    "invalid_utf8",
    "response_too_large",
]


class CheckpointSpanUnavailable(ValueError):
    def __init__(self, code: SpanErrorCode):
        self.code = code
        super().__init__(code)


def read_checkpoint_span(
    store: SessionStore, checkpoint: CheckpointRecord, selection: CheckpointSpanSelection
) -> CheckpointSpan:
    """Caller must check checkpoint ownership first; only an existing saved blob is read."""
    file = checkpoint.files.get(selection.path)
    if file is None:
        raise CheckpointSpanUnavailable("file_not_captured")
    if file.sha256 != selection.expected_sha256:
        raise CheckpointSpanUnavailable("hash_mismatch")
    if file.size == 0:
        raise CheckpointSpanUnavailable("range_unavailable")
    if file.size > MAX_SPAN_FILE_BYTES:
        raise CheckpointSpanUnavailable("file_too_large")
    try:
        raw = store.checkpoint_content(checkpoint, selection.path)
    except ValueError as exc:
        raise CheckpointSpanUnavailable("content_unavailable") from exc
    if raw is None:
        raise CheckpointSpanUnavailable("content_unavailable")
    # checkpoint_content validates full blob length/hash before returning any selected bytes.
    selected: list[bytes] = []
    size = 0
    total = 0
    for total, line in enumerate(io.BytesIO(raw), 1):
        if selection.start_line <= total <= selection.end_line:
            size += len(line)
            if size > MAX_SPAN_BYTES:
                raise CheckpointSpanUnavailable("span_too_large")
            selected.append(line)
    if selection.end_line > total:
        raise CheckpointSpanUnavailable("range_unavailable")
    content_bytes = b"".join(selected)
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CheckpointSpanUnavailable("invalid_utf8") from exc
    result = CheckpointSpan(
        session_id=checkpoint.session_id,
        checkpoint_id=checkpoint.checkpoint_id,
        checkpoint_kind=checkpoint.kind,
        checkpoint_record_sha256=hashlib.sha256(checkpoint.model_dump_json().encode()).hexdigest(),
        path=selection.path,
        file_sha256=file.sha256,
        file_bytes=file.size,
        mode=file.mode,
        total_lines=total,
        start_line=selection.start_line,
        end_line=selection.end_line,
        selected_bytes=size,
        content_sha256=hashlib.sha256(content_bytes).hexdigest(),
        content=content,
        has_more=selection.end_line < total,
    )
    if len(result.model_dump_json().encode()) > MAX_SPAN_RESPONSE_BYTES:
        raise CheckpointSpanUnavailable("response_too_large")
    return result
