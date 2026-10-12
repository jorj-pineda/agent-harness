"""Bounded tool-event metadata for review, never a replayable transcript."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

MAX_SAVED_TOOL_EVENTS = 200


class SavedToolEvent(BaseModel):
    sequence: int = Field(ge=1)
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    type: Literal["tool_start", "tool_end"]
    tool: str = Field(max_length=128)
    error_observed: bool | None = None
    latency_ms: float | None = Field(default=None, ge=0)


class ToolProgress(BaseModel):
    run_id: str
    coverage: Literal["recorded", "not_recorded"] = "not_recorded"
    events: list[SavedToolEvent] = Field(default_factory=list, max_length=MAX_SAVED_TOOL_EVENTS)
    observed_events: int = Field(default=0, ge=0)
    events_truncated: bool = False
    updated_at: datetime | None = None
    read_only: Literal[True] = True
    scope_note: str = (
        "Bounded prefix of observed tool-start/end metadata; arguments, results, error text, "
        "model messages and tool-call IDs are not stored here. Start precedes tool gates, so "
        "it does not prove dispatch or mutation. End/error presence is not patch acceptance "
        "or verification. A missing end may mean interrupted work or unsaved evidence; "
        "file changes may exist. This is not a transcript, checkpoint, replay, repair, "
        "or permission to resume. Unknown tool names are recorded as unknown_tool."
    )
