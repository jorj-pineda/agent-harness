from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from providers.base import ChatMessage, ChatProvider, Embedder, ProviderError, ToolSpec
from providers.openai import OpenAIProvider
from tests._cassette import CassetteTransport

# ---------------------------------------------------------------------------
# Original inline-mock helpers (kept for tests that verify request-body shape)
# ---------------------------------------------------------------------------


def _completion_response(payload: dict[str, Any]) -> tuple[httpx.AsyncClient, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=payload, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client, captured


# ---------------------------------------------------------------------------
# Cassette-backed happy-path tests
# ---------------------------------------------------------------------------


async def test_cassette_chat_plain_text(
    cassette: Callable[..., CassetteTransport],
) -> None:
    transport = cassette("openai_chat_plain")
    http_client = httpx.AsyncClient(transport=transport)
    provider = OpenAIProvider(api_key="test", model="gpt-4o", http_client=http_client)

    resp = await provider.chat([ChatMessage(role="user", content="hi")])

    assert resp.content == "hello there"
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"
    assert resp.usage.prompt_tokens == 9
    assert resp.usage.completion_tokens == 3


async def test_cassette_chat_tool_call(
    cassette: Callable[..., CassetteTransport],
) -> None:
    transport = cassette("openai_chat_tool_call")
    http_client = httpx.AsyncClient(transport=transport)
    provider = OpenAIProvider(api_key="test", model="gpt-4o", http_client=http_client)
    tool = ToolSpec(
        name="sql_query",
        description="read-only SQL",
        parameters_schema={"type": "object", "properties": {"sql": {"type": "string"}}},
    )

    resp = await provider.chat([ChatMessage(role="user", content="how many?")], tools=[tool])

    assert resp.content == ""
    assert resp.finish_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.id == "call_abc"
    assert call.name == "sql_query"
    assert call.arguments == {"sql": "SELECT 1"}


async def test_cassette_chat_http_error(
    cassette: Callable[..., CassetteTransport],
) -> None:
    transport = cassette("openai_chat_error")
    http_client = httpx.AsyncClient(transport=transport)
    provider = OpenAIProvider(
        api_key="test", model="gpt-4o", max_retries=0, http_client=http_client
    )

    with pytest.raises(ProviderError):
        await provider.chat([ChatMessage(role="user", content="hi")])


# ---------------------------------------------------------------------------
# Inline-mock tests (verify request body shape and edge cases)
# ---------------------------------------------------------------------------


async def test_chat_plain_text_response() -> None:
    http_client, captured = _completion_response(
        {
            "id": "chatcmpl_01",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello there"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
        }
    )
    provider = OpenAIProvider(api_key="test", model="gpt-4o", http_client=http_client)

    resp = await provider.chat([ChatMessage(role="user", content="hi")])

    body = captured["body"]
    assert body["model"] == "gpt-4o"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["temperature"] == 0.0
    assert "tools" not in body

    assert resp.content == "hello there"
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"
    assert resp.usage.prompt_tokens == 9
    assert resp.usage.completion_tokens == 3


async def test_chat_tool_call_arguments_parsed_from_json_string() -> None:
    http_client, _ = _completion_response(
        {
            "id": "chatcmpl_02",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "sql_query",
                                    "arguments": '{"sql": "SELECT 1"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
        }
    )
    provider = OpenAIProvider(api_key="test", model="gpt-4o", http_client=http_client)
    tool = ToolSpec(
        name="sql_query",
        description="read-only SQL",
        parameters_schema={"type": "object", "properties": {"sql": {"type": "string"}}},
    )

    resp = await provider.chat([ChatMessage(role="user", content="how many?")], tools=[tool])

    assert resp.content == ""
    assert resp.finish_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.id == "call_abc"
    assert call.name == "sql_query"
    assert call.arguments == {"sql": "SELECT 1"}


async def test_chat_http_error_wraps_in_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"type": "invalid_request_error", "message": "boom"}},
            request=request,
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(
        api_key="test", model="gpt-4o", max_retries=0, http_client=http_client
    )

    with pytest.raises(ProviderError):
        await provider.chat([ChatMessage(role="user", content="hi")])


def test_openai_satisfies_chat_protocol_only() -> None:
    provider = OpenAIProvider(api_key="test", model="gpt-4o")
    assert isinstance(provider, ChatProvider)
    assert not isinstance(provider, Embedder)


# ---------------------------------------------------------------------------
# Live tests — require a real OpenAI API key (pytest -m live)
# ---------------------------------------------------------------------------

_has_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", ""),
    reason="OPENAI_API_KEY not set — set it to run live OpenAI tests",
)


@pytest.mark.live
@_has_openai_key
async def test_live_openai_chat() -> None:
    """Smoke test: send a one-turn chat to the real OpenAI API.

    Run with: OPENAI_API_KEY=sk-... pytest -m live -k openai
    This is the call that the openai_chat_plain cassette was derived from.
    """
    provider = OpenAIProvider(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o",
    )
    try:
        resp = await provider.chat(
            [ChatMessage(role="user", content="Reply with exactly one word: hello")],
            max_tokens=16,
        )
        assert resp.content.strip() != ""
        assert resp.finish_reason in ("stop", "length")
        assert "gpt" in resp.model
    finally:
        await provider.aclose()
