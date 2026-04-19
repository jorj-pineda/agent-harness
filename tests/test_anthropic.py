from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from providers.anthropic import AnthropicProvider
from providers.base import ChatMessage, ChatProvider, Embedder, ProviderError, ToolSpec


def _messages_response(payload: dict[str, Any]) -> tuple[httpx.AsyncClient, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=payload, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client, captured


async def test_chat_plain_text_response() -> None:
    http_client, captured = _messages_response(
        {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello there"}],
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 9, "output_tokens": 3},
        }
    )
    provider = AnthropicProvider(
        api_key="test",
        model="claude-sonnet-4-6",
        http_client=http_client,
    )

    resp = await provider.chat([ChatMessage(role="user", content="hi")])

    body = captured["body"]
    assert body["model"] == "claude-sonnet-4-6"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["max_tokens"] == 4096
    assert body["temperature"] == 0.0
    assert "system" not in body
    assert "tools" not in body

    assert resp.content == "hello there"
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"
    assert resp.usage.prompt_tokens == 9
    assert resp.usage.completion_tokens == 3
    assert resp.model == "claude-sonnet-4-6"
    assert resp.latency_ms >= 0.0


async def test_chat_system_message_becomes_top_level_parameter() -> None:
    http_client, captured = _messages_response(
        {
            "id": "msg_02",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
    )
    provider = AnthropicProvider(api_key="test", model="claude-sonnet-4-6", http_client=http_client)

    await provider.chat(
        [
            ChatMessage(role="system", content="be brief"),
            ChatMessage(role="user", content="hi"),
        ]
    )

    body = captured["body"]
    assert body["system"] == "be brief"
    assert body["messages"] == [{"role": "user", "content": "hi"}]


async def test_chat_tool_use_block_maps_to_tool_call() -> None:
    http_client, captured = _messages_response(
        {
            "id": "msg_03",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_abc123",
                    "name": "sql_query",
                    "input": {"sql": "SELECT 1"},
                }
            ],
            "model": "claude-sonnet-4-6",
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 42, "output_tokens": 7},
        }
    )
    provider = AnthropicProvider(api_key="test", model="claude-sonnet-4-6", http_client=http_client)
    tool = ToolSpec(
        name="sql_query",
        description="read-only SQL",
        parameters_schema={"type": "object", "properties": {"sql": {"type": "string"}}},
    )

    resp = await provider.chat([ChatMessage(role="user", content="how many?")], tools=[tool])

    body = captured["body"]
    assert body["tools"] == [
        {
            "name": "sql_query",
            "description": "read-only SQL",
            "input_schema": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
            },
        }
    ]

    assert resp.content == ""
    assert resp.finish_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.id == "toolu_abc123"
    assert call.name == "sql_query"
    assert call.arguments == {"sql": "SELECT 1"}


async def test_chat_max_tokens_override_wins() -> None:
    http_client, captured = _messages_response(
        {
            "id": "msg_04",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-sonnet-4-6",
            "stop_reason": "max_tokens",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 128},
        }
    )
    provider = AnthropicProvider(
        api_key="test",
        model="claude-sonnet-4-6",
        default_max_tokens=1024,
        http_client=http_client,
    )

    resp = await provider.chat(
        [ChatMessage(role="user", content="go")], temperature=0.7, max_tokens=128
    )

    body = captured["body"]
    assert body["max_tokens"] == 128
    assert body["temperature"] == 0.7
    assert resp.finish_reason == "length"


async def test_chat_tool_result_message_is_user_role() -> None:
    http_client, captured = _messages_response(
        {
            "id": "msg_05",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
    )
    provider = AnthropicProvider(api_key="test", model="claude-sonnet-4-6", http_client=http_client)

    await provider.chat(
        [
            ChatMessage(role="user", content="query please"),
            ChatMessage(role="tool", content="42", tool_call_id="toolu_abc123"),
        ]
    )

    assert captured["body"]["messages"] == [
        {"role": "user", "content": "query please"},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_abc123", "content": "42"}],
        },
    ]


async def test_chat_http_error_wraps_in_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"type": "error", "error": {"type": "invalid_request_error", "message": "boom"}},
            request=request,
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(
        api_key="test",
        model="claude-sonnet-4-6",
        max_retries=0,
        http_client=http_client,
    )

    with pytest.raises(ProviderError):
        await provider.chat([ChatMessage(role="user", content="hi")])


def test_anthropic_satisfies_chat_protocol_only() -> None:
    provider = AnthropicProvider(api_key="test", model="claude-sonnet-4-6")
    assert isinstance(provider, ChatProvider)
    assert not isinstance(provider, Embedder)
