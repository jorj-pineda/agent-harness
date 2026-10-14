"""Opt-in normalized turn evidence, separate from model context and API payloads.

No provider wire payloads, headers, or exception text are retained. Model-visible
content is still sensitive: collectors must use trusted workspaces and control
access to stored traces. This is not a credential-redaction mechanism.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from providers.base import ChatMessage, FinishReason, Role, TokenUsage, ToolCall, ToolSpec

from .context import (
    TokenBudget,
    estimate_message_tokens,
    estimate_tool_tokens,
    message_size_bytes,
)
from .outcome_types import CompletionStatus


class MessageFootprint(BaseModel):
    message_index: int
    role: Role
    tool_name: str | None = None
    payload_bytes: int


class PromptFootprint(BaseModel):
    """Estimator metadata only; no content, arguments, paths or call IDs."""

    message_estimate_tokens: int
    tool_schema_estimate_tokens: int
    heuristic_prompt_tokens: int
    anchor_adjustment_tokens: int
    estimated_prompt_tokens: int
    # Ten largest heuristic payloads, descending bytes then ascending index.
    largest_messages: list[MessageFootprint]
    largest_messages_truncated: bool


def prompt_footprint(
    messages: list[ChatMessage], tools: list[ToolSpec], budget: TokenBudget
) -> PromptFootprint:
    message_tokens = estimate_message_tokens(messages)
    tool_tokens = estimate_tool_tokens(tools)
    heuristic = message_tokens + tool_tokens
    estimated = budget.estimate_prompt(messages, tools)
    rows = [
        MessageFootprint(
            message_index=index,
            role=message.role,
            tool_name=message.tool_name,
            payload_bytes=message_size_bytes(message),
        )
        for index, message in enumerate(messages)
    ]
    rows.sort(key=lambda row: (-row.payload_bytes, row.message_index))
    return PromptFootprint(
        message_estimate_tokens=message_tokens,
        tool_schema_estimate_tokens=tool_tokens,
        heuristic_prompt_tokens=heuristic,
        anchor_adjustment_tokens=estimated - heuristic,
        estimated_prompt_tokens=estimated,
        largest_messages=rows[:10],
        largest_messages_truncated=len(rows) > 10,
    )


class BudgetStopTrace(BaseModel):
    """Token-gate stop evidence; stage distinguishes request prevention from response handling."""

    kind: Literal["budget_stop"] = "budget_stop"
    stage: Literal["pre_request", "post_response"] = "pre_request"
    requested_tool_count: int = 0
    iteration: int
    message_count: int
    status: CompletionStatus
    reason: str
    footprint: PromptFootprint
    prompt_tokens_used: int | None
    completion_tokens_used: int | None
    max_context_tokens: int
    max_total_tokens: int
    max_completion_tokens: int
    min_request_output_tokens: int


class RequestTrace(BaseModel):
    kind: Literal["request"] = "request"
    iteration: int
    message_count: int
    estimated_prompt_tokens: int
    max_tokens: int | None
    footprint: PromptFootprint | None = None
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


TurnTraceRecord = (
    RequestTrace | ResponseTrace | MessageTrace | RequestFailureTrace | BudgetStopTrace
)
