"""Opt-in normalized turn evidence, separate from model context and API payloads.

No provider wire payloads, headers, or exception text are retained. Model-visible
content is still sensitive: collectors must use trusted workspaces and control
access to stored traces. This is not a credential-redaction mechanism.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from providers.base import ChatMessage, FinishReason, TokenUsage, ToolCall


class RequestTrace(BaseModel):
    kind: Literal["request"] = "request"
    iteration: int
    message_count: int
    estimated_prompt_tokens: int
    max_tokens: int | None
    # Only the first request includes history; later requests use message events.
    initial_messages: list[ChatMessage] = Field(default_factory=list)


class ResponseTrace(BaseModel):
    kind: Literal["response"] = "response"
    iteration: int
    content: str
    tool_calls: list[ToolCall]
    finish_reason: FinishReason
    usage: TokenUsage
    model: str
    latency_ms: float


class MessageTrace(BaseModel):
    kind: Literal["recovery", "tool_result"]
    iteration: int
    message: ChatMessage


class RequestFailureTrace(BaseModel):
    kind: Literal["request_failure"] = "request_failure"
    iteration: int
    exception_type: str


TurnTraceRecord = RequestTrace | ResponseTrace | MessageTrace | RequestFailureTrace
