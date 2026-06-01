"""Streaming events for live tool traces (slice 9c).

`run_turn` accepts an optional `EventCallback` and fires a `ToolStartEvent`
before each tool invocation and a `ToolEndEvent` after it. The API layer drains
those events onto an SSE channel so the demo panel can draw tool cards as the
turn unfolds — the ReAct loop is **not** duplicated; the callback only
observes it.

`TurnDoneEvent` and `ErrorEvent` are emitted by the API layer once the loop
returns (or raises), carrying the full rule-#5 envelope or the failure detail.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from .state import TurnResponse


class ToolStartEvent(BaseModel):
    type: Literal["tool_start"] = "tool_start"
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolEndEvent(BaseModel):
    type: Literal["tool_end"] = "tool_end"
    tool: str
    latency_ms: float
    error: str | None = None
    result_snippet: str | None = None


class TurnDoneEvent(BaseModel):
    type: Literal["turn_done"] = "turn_done"
    response: TurnResponse


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    detail: str


StreamEvent = ToolStartEvent | ToolEndEvent | TurnDoneEvent | ErrorEvent
EventCallback = Callable[[StreamEvent], Awaitable[None]]
