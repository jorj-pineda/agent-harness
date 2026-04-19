from __future__ import annotations

from providers.base import (
    ChatMessage,
    ChatProvider,
    Embedder,
    ProviderResponse,
    TokenUsage,
    ToolCall,
    ToolSpec,
)


def test_chat_message_defaults() -> None:
    msg = ChatMessage(role="user", content="hi")
    assert msg.tool_calls == []
    assert msg.tool_call_id is None


def test_tool_call_roundtrip() -> None:
    tc = ToolCall(id="call_1", name="sql_query", arguments={"q": "SELECT 1"})
    dumped = tc.model_dump()
    restored = ToolCall.model_validate(dumped)
    assert restored == tc


def test_provider_response_defaults() -> None:
    resp = ProviderResponse(
        content="answer",
        finish_reason="stop",
        model="gemma4",
        latency_ms=12.5,
    )
    assert resp.tool_calls == []
    assert resp.usage == TokenUsage()
    assert resp.raw == {}


def test_tool_spec_holds_json_schema() -> None:
    spec = ToolSpec(
        name="sql_query",
        description="Read-only SQL against the support DB.",
        parameters_schema={
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
    )
    assert spec.parameters_schema["required"] == ["sql"]


class _StubChat:
    name = "stub"

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        return ProviderResponse(
            content="",
            finish_reason="stop",
            model=self.name,
            latency_ms=0.0,
        )


class _StubEmbed:
    name = "stub-embed"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 3 for _ in texts]


def test_stubs_satisfy_protocols() -> None:
    assert isinstance(_StubChat(), ChatProvider)
    assert isinstance(_StubEmbed(), Embedder)
