"""Structural integrity of normalized history, without repair or resume permission."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from providers.base import ChatMessage, ToolCall


class TranscriptIssue(BaseModel):
    message_index: int
    code: str


class TranscriptReview(BaseModel):
    status: Literal["valid", "invalid"] = "valid"
    message_count: int
    turn_id: str | None = None
    issue_count: int = 0
    issues: list[TranscriptIssue] = Field(default_factory=list)
    issues_truncated: bool = False
    read_only: Literal[True] = True
    scope_note: str = (
        "Checks normalized tool-call/result structure only. Does not validate answer claims, "
        "provider/model compatibility, context fit, or workspace state; not permission to resume. "
        "History is never repaired or replayed by this review."
    )


def tool_batch_issue(calls: Sequence[ToolCall]) -> str | None:
    """IDs must be nonempty and unique within a batch, not across the conversation."""
    seen: set[str] = set()
    for call in calls:
        if not call.id.strip():
            return "empty_tool_call_id"
        if not call.name.strip():
            return "empty_tool_name"
        if call.id in seen:
            return "duplicate_tool_call_id"
        seen.add(call.id)
    return None


def inspect_transcript(
    messages: Sequence[ChatMessage], *, max_issues: int = 100
) -> TranscriptReview:
    """Return bounded diagnostics containing indices/codes, never transcript payloads."""
    if max_issues < 1:
        raise ValueError("max_issues must be positive")
    review = TranscriptReview(message_count=len(messages))
    pending: dict[str, tuple[str, int]] = {}

    def issue(index: int, code: str) -> None:
        review.status = "invalid"
        review.issue_count += 1
        if len(review.issues) < max_issues:
            review.issues.append(TranscriptIssue(message_index=index, code=code))

    def missing_results() -> None:
        for _, index in pending.values():
            issue(index, "missing_tool_result")
        pending.clear()

    for index, message in enumerate(messages):
        if message.role != "tool" and pending:
            missing_results()
        if message.role == "system" and index != 0:
            issue(index, "misplaced_system_message")
        if message.role != "assistant" and message.tool_calls:
            issue(index, "tool_calls_on_non_assistant")
        if message.role != "tool" and (
            message.tool_call_id is not None or message.tool_name is not None
        ):
            issue(index, "tool_result_fields_on_non_tool")
        if message.role == "assistant":
            batch_issue = tool_batch_issue(message.tool_calls)
            if batch_issue:
                issue(index, batch_issue)
            for call in message.tool_calls:
                pending[call.id] = (call.name, index)
        elif message.role == "tool":
            matched = pending.pop(message.tool_call_id, None) if message.tool_call_id else None
            if matched is None:
                issue(index, "orphan_tool_result")
            elif message.tool_name is not None and message.tool_name != matched[0]:
                issue(index, "tool_name_mismatch")
    missing_results()
    review.issues_truncated = review.issue_count > len(review.issues)
    return review


def transcript_block_reason(messages: Sequence[ChatMessage]) -> str | None:
    review = inspect_transcript(messages, max_issues=1)
    if review.status == "valid":
        return None
    return (
        f"Transcript integrity check failed ({review.issues[0].code}); "
        "no inference or tools started. Inspect saved history and create a new session."
    )
