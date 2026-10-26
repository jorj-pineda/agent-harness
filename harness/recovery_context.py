"""Explicit bounded assembly of historical recovery evidence, never executable history."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import Field, model_validator

from .checkpoint_spans import CheckpointSpan, CheckpointSpanSelection, read_checkpoint_span
from .checkpoints import CheckpointRecord
from .recovery_review import RecoveryReview, ReviewModel, build_recovery_review
from .runs import RunRecord
from .session_store import SessionStore

MAX_CONTEXT_BYTES = 65536
MAX_CONTEXT_SOURCE_BYTES = 16384
MAX_CONTEXT_SPANS = 4


class RecoveryContextSelection(ReviewModel):
    expected_record_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    spans: list[CheckpointSpanSelection] = Field(max_length=MAX_CONTEXT_SPANS)

    @model_validator(mode="after")
    def disjoint_ranges(self) -> RecoveryContextSelection:
        for i, span in enumerate(self.spans):
            for previous in self.spans[:i]:
                if span.path == previous.path and (
                    span.expected_sha256 != previous.expected_sha256
                    or max(span.start_line, previous.start_line)
                    <= min(span.end_line, previous.end_line)
                ):
                    raise ValueError("Same-file selections must have one hash and disjoint ranges")
        return self


class RecoveryContext(ReviewModel):
    schema_version: Literal[1] = 1
    review: RecoveryReview
    spans: list[CheckpointSpan]
    selected_source_bytes: int
    evidence_sha256: str
    read_only: Literal[True] = True
    execution_supported: Literal[False] = False
    max_serialized_bytes: Literal[65536] = 65536
    scope_note: str = (
        "Historical recovery-context preview, not a system prompt, executable history or "
        "permission to continue. Review excerpts and source text are untrusted owner data. "
        "Only explicitly selected spans from this run's linked final checkpoint are included; "
        "all other source is omitted. The review retains its truncation/omission/unknown flags. "
        "No current workspace, instructions, configuration, context fit or budget is validated. "
        "Original records remain authoritative and unchanged."
    )


ContextErrorCode = Literal[
    "run_active",
    "record_hash_mismatch",
    "checkpoint_mismatch",
    "source_budget_exceeded",
    "context_too_large",
]


class RecoveryContextUnavailable(ValueError):
    def __init__(self, code: ContextErrorCode):
        self.code = code
        super().__init__(code)


def build_recovery_context(
    store: SessionStore, record: RunRecord, selection: RecoveryContextSelection
) -> RecoveryContext:
    """Caller checks run ownership. No disk workspace access or durable writes occur."""
    if record.status == "running":
        raise RecoveryContextUnavailable("run_active")
    digest = hashlib.sha256(record.model_dump_json().encode()).hexdigest()
    if digest != selection.expected_record_sha256:
        raise RecoveryContextUnavailable("record_hash_mismatch")
    checkpoint: CheckpointRecord | None = None
    if selection.spans:
        checkpoint = (
            store.get_checkpoint(record.after_checkpoint_id) if record.after_checkpoint_id else None
        )
        if (
            checkpoint is None
            or checkpoint.checkpoint_id != record.after_checkpoint_id
            or checkpoint.session_id != record.request.session_id
            or checkpoint.run_id != record.run_id
            or checkpoint.kind != "turn_after"
            or checkpoint.paired_before_id != record.before_checkpoint_id
            or record.before_checkpoint_id is None
        ):
            raise RecoveryContextUnavailable("checkpoint_mismatch")
    try:
        review = build_recovery_review(record)
    except ValueError as exc:
        raise RecoveryContextUnavailable("context_too_large") from exc
    spans: list[CheckpointSpan] = []
    source_bytes = 0
    for item in selection.spans:
        assert checkpoint is not None
        span = read_checkpoint_span(store, checkpoint, item)
        source_bytes += span.selected_bytes
        if source_bytes > MAX_CONTEXT_SOURCE_BYTES:
            raise RecoveryContextUnavailable("source_budget_exceeded")
        spans.append(span)
    # Hash the exact ordered compact evidence representation, excluding wrapper metadata.
    evidence = review.model_dump_json() + "\n" + "\n".join(s.model_dump_json() for s in spans)
    result = RecoveryContext(
        review=review,
        spans=spans,
        selected_source_bytes=source_bytes,
        evidence_sha256=hashlib.sha256(evidence.encode()).hexdigest(),
    )
    if len(result.model_dump_json().encode()) > MAX_CONTEXT_BYTES:
        raise RecoveryContextUnavailable("context_too_large")
    return result
