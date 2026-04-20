"""Harness-layer state: conversation transcript, per-turn records, response shape.

Two axes of history live on `Session`:

- `messages` — the raw chat transcript in provider format. This is what gets
  fed back into the Provider on each iteration of the ReAct loop.
- `turns` — grouped per-user-input records: user input, tool calls (with
  results), memory writes, final answer, latency. This is what the eval
  harness reads.

`TurnResponse` is the rule-#5 payload every API response ships. `confidence`
and `citations` fill in meaningfully in step 7 (grounding); `memory_writes`
in step 8. Today they carry placeholder defaults so the shape is stable from
the start and step 6 doesn't have to reach forward for their eventual
semantics.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from providers.base import ChatMessage


def _uuid() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ToolCallRecord(BaseModel):
    """A completed tool invocation — call args + result (or error) + latency.

    Distinct from `providers.base.ToolCall`, which is the *request* the model
    emits; this is the *record* after the registry has executed it.
    """

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0


class Turn(BaseModel):
    """One user input → final assistant answer, with every intermediate call."""

    turn_id: str = Field(default_factory=_uuid)
    user_input: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    memory_writes: list[str] = Field(default_factory=list)
    final_answer: str = ""
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None


class Session(BaseModel):
    """A live conversation — raw transcript plus structured turn history."""

    session_id: str = Field(default_factory=_uuid)
    user_id: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    turns: list[Turn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class TurnResponse(BaseModel):
    """The rule-#5 payload shipped on every response.

    `confidence` / `citations` / `escalated` fill in at step 7 (grounding).
    `memory_writes` fills in at step 8 (memory layer). Their defaults keep
    the wire shape stable from step 6 onward.
    """

    answer: str
    confidence: float | None = None
    citations: list[str] = Field(default_factory=list)
    escalated: bool = False
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    memory_writes: list[str] = Field(default_factory=list)
    provider: str
    latency_ms: float
