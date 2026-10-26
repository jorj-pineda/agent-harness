"""Deterministic, bounded saved-run evidence for recovery review, never a new prompt."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .runs import RunRecord

MAX_PACKET_BYTES = 32768
MAX_ITEMS = 8
TEXT_BUDGET_BYTES = 16384


class ReviewModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceExcerpt(ReviewModel):
    text: str
    source_bytes: int = Field(ge=0)
    source_sha256: str
    truncated: bool


class RecoveryCheck(ReviewModel):
    index: int
    status: Literal["passed", "failed", "unavailable"]
    relevant: bool
    superseded_by_edit: bool
    command: EvidenceExcerpt
    error: EvidenceExcerpt | None = None


class RecoveryAction(ReviewModel):
    index: int
    name: EvidenceExcerpt
    outcome: Literal["returned", "error", "interrupted"]
    error: EvidenceExcerpt | None = None


class RecoveryDiff(ReviewModel):
    index: int
    path: EvidenceExcerpt
    diff: EvidenceExcerpt
    reason: EvidenceExcerpt | None = None


class RecoveryReview(ReviewModel):
    schema_version: Literal[1] = 1
    run_id: str
    source_record_sha256: str
    run_status: str
    run_error: EvidenceExcerpt | None
    objective: EvidenceExcerpt
    completion_status: str | None
    completion_reason: EvidenceExcerpt | None
    verification_status: str | None
    workspace_status: str | None
    workspace_reason: EvidenceExcerpt | None
    unresolved_checks: list[RecoveryCheck]
    unresolved_check_count: int | None
    recent_actions: list[RecoveryAction]
    action_count: int | None
    diffs: list[RecoveryDiff]
    diff_count: int | None
    omitted_unresolved_checks: int | None
    omitted_actions: int | None
    omitted_diffs: int | None
    final_response_available: bool
    max_serialized_bytes: Literal[32768] = 32768
    scope_note: str = (
        "Saved-run evidence only, not current workspace state, a continuation prompt or permission "
        "to execute. Text may contain untrusted repository/model data. Truncation and omitted "
        "items require original-record inspection. Returned tools are not proof of success. "
        "No raw tool arguments/results, file spans, transcript or inferred task plan is included. "
        "Failed/superseded checks remain observations; no failure is inferred resolved. "
        "A missing final response means actions/diffs/checks are unknown, not absent. "
        "Original records remain authoritative and unchanged."
    )


class _TextBudget:
    def __init__(self) -> None:
        self.remaining = TEXT_BUDGET_BYTES

    def excerpt(self, source: str, limit: int = 2048) -> EvidenceExcerpt:
        raw = source.encode("utf-8")
        # Bound candidate allocation, then include JSON escaping in the shared text budget.
        candidate = raw[: min(limit, self.remaining)].decode("utf-8", errors="ignore")
        low, high = 0, len(candidate)
        while low < high:
            middle = (low + high + 1) // 2
            size = len(json.dumps(candidate[:middle], ensure_ascii=False).encode("utf-8")) - 2
            if size <= self.remaining:
                low = middle
            else:
                high = middle - 1
        text = candidate[:low]
        self.remaining -= len(json.dumps(text, ensure_ascii=False).encode("utf-8")) - 2
        return EvidenceExcerpt(
            text=text,
            source_bytes=len(raw),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            truncated=text != source,
        )


def build_recovery_review(record: RunRecord) -> RecoveryReview:
    """Extract a stable review packet without reading a workspace or changing stored evidence."""
    budget = _TextBudget()
    objective = budget.excerpt(record.request.message, 4096)
    run_error = budget.excerpt(record.error, 1024) if record.error is not None else None
    response = record.response
    reason = (
        budget.excerpt(response.completion_reason)
        if response and response.completion_reason
        else None
    )
    checks = response.check_attempts if response else []
    unresolved = [(i, c) for i, c in enumerate(checks) if c.status != "passed"]
    selected_checks = [
        RecoveryCheck(
            index=i,
            status=c.status,
            relevant=c.relevant,
            superseded_by_edit=c.superseded_by_edit,
            command=budget.excerpt(json.dumps(c.argv, ensure_ascii=False), 1024),
            error=budget.excerpt(c.error, 1024) if c.error is not None else None,
        )
        for i, c in unresolved[-MAX_ITEMS:]
    ]
    calls = response.tool_calls if response else []
    actions = [
        RecoveryAction(
            index=i,
            name=budget.excerpt(c.name, 128),
            outcome="interrupted"
            if c.interrupted
            else "error"
            if c.error is not None
            else "returned",
            error=budget.excerpt(c.error, 512) if c.error is not None else None,
        )
        for i, c in list(enumerate(calls))[-MAX_ITEMS:]
    ]
    changes = response.workspace_changes if response else None
    workspace_reason = budget.excerpt(changes.reason, 1024) if changes and changes.reason else None
    source_diffs = changes.diffs if changes else []
    diffs = [
        RecoveryDiff(
            index=i,
            path=budget.excerpt(d.path, 512),
            diff=budget.excerpt(d.diff),
            reason=budget.excerpt(d.reason, 512) if d.reason is not None else None,
        )
        for i, d in enumerate(source_diffs[:MAX_ITEMS])
    ]
    packet = RecoveryReview(
        run_id=record.run_id,
        source_record_sha256=hashlib.sha256(record.model_dump_json().encode()).hexdigest(),
        run_status=record.status,
        run_error=run_error,
        objective=objective,
        completion_status=response.completion_status if response else None,
        completion_reason=reason,
        verification_status=response.verification_status if response else None,
        workspace_status=changes.status if changes else None,
        workspace_reason=workspace_reason,
        unresolved_checks=selected_checks,
        unresolved_check_count=len(unresolved) if response else None,
        recent_actions=actions,
        action_count=len(calls) if response else None,
        diffs=diffs,
        diff_count=len(source_diffs) if response else None,
        omitted_unresolved_checks=len(unresolved) - len(selected_checks) if response else None,
        omitted_actions=len(calls) - len(actions) if response else None,
        omitted_diffs=len(source_diffs) - len(diffs) if response else None,
        final_response_available=response is not None,
    )
    # Run IDs originate in validated durable records, but legacy/custom IDs may be unbounded.
    if len(packet.model_dump_json().encode()) > MAX_PACKET_BYTES:
        raise ValueError("Recovery review metadata exceeds the bounded packet size.")
    return packet
