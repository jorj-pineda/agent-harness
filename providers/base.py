"""Provider-agnostic types and protocols for chat/embedding backends.

Nothing above `providers/` may import a specific backend. Model-specific quirks
(Gemma 4 tool-call format vs Anthropic tool_use blocks vs OpenAI tool_calls)
are normalized here at the boundary.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

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
    tool_name: str | None = Field(
        default=None,
        description="Tool name for role=tool messages (Ollama wire format).",
    )


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters_schema: dict[str, Any]


class TokenUsage(BaseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)


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
class ConversationProvider(Protocol):
    """Optional pure binding; the original provider still owns resources."""

    def for_conversation(self, conversation_id: str) -> ChatProvider: ...


@runtime_checkable
class Embedder(Protocol):
    name: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class ProviderFailure(BaseModel):
    """Allowlisted observations only; never raw exception/request/response text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[
        "read_timeout",
        "connect_timeout",
        "write_timeout",
        "pool_timeout",
        "timeout",
        "connection",
        "http_status",
        "unknown",
    ] = "unknown"
    http_status: int | None = Field(default=None, strict=True, ge=100, le=599)


class ProviderError(Exception):
    """Raised by provider implementations for any backend-side failure.

    Keeps provider-specific exception types from leaking upward.
    """

    def __init__(self, message: str = "", *, failure: ProviderFailure | None = None) -> None:
        super().__init__(message)
        self.failure = failure if failure is not None else ProviderFailure()
