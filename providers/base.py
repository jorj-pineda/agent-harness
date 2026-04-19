"""Provider-agnostic types and protocols for chat/embedding backends.

Nothing above `providers/` may import a specific backend. Model-specific quirks
(Gemma 4 tool-call format vs Anthropic tool_use blocks vs OpenAI tool_calls)
are normalized here at the boundary.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]
FinishReason = Literal["stop", "tool_use", "length", "content_filter", "error"]


class ToolCall(BaseModel):
    """A tool invocation requested by the model.

    `arguments` is the raw model-emitted object; JSON-schema validation
    against the tool spec happens in the tool registry, not here.
    """

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters_schema: dict[str, Any]


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ProviderResponse(BaseModel):
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str
    latency_ms: float
    raw: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ChatProvider(Protocol):
    name: str

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ProviderResponse: ...


@runtime_checkable
class Embedder(Protocol):
    name: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class ProviderError(Exception):
    """Raised by provider implementations for any backend-side failure.

    Keeps provider-specific exception types from leaking upward.
    """
