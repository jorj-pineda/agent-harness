from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from providers.base import ChatMessage, ChatProvider, Embedder, ProviderError, ToolSpec
from providers.ollama import OllamaProvider
from tests._cassette import CassetteTransport

# ---------------------------------------------------------------------------
# Original inline-mock helpers (kept for tests that verify request-body shape)
# ---------------------------------------------------------------------------


def _chat_response(payload: dict[str, Any]) -> httpx.MockTransport:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=payload, request=request)

    transport = httpx.MockTransport(handler)
    transport.captured = captured  # type: ignore[attr-defined]
    return transport


# ---------------------------------------------------------------------------
# Cassette-backed happy-path tests
# ---------------------------------------------------------------------------


async def test_cassette_chat_plain_text(
    cassette: Callable[..., CassetteTransport],
) -> None:
    transport = cassette("ollama_chat_plain")
    provider = OllamaProvider(
        host="http://fake",
        model="gemma4",
        embed_model="nomic-embed-text",
        transport=transport,
    )

    resp = await provider.chat([ChatMessage(role="user", content="hi")])

    assert resp.content == "hello there"
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"
    assert resp.usage.prompt_tokens == 9
    assert resp.usage.completion_tokens == 3
    assert resp.model == "gemma4"
    assert resp.latency_ms >= 0.0


async def test_cassette_chat_tool_call(
    cassette: Callable[..., CassetteTransport],
) -> None:
    transport = cassette("ollama_chat_tool_call")
    provider = OllamaProvider(
        host="http://fake",
        model="gemma4",
        embed_model="nomic-embed-text",
        transport=transport,
    )
    tool = ToolSpec(
        name="sql_query",
        description="read-only SQL",
        parameters_schema={"type": "object", "properties": {"sql": {"type": "string"}}},
    )

    resp = await provider.chat([ChatMessage(role="user", content="how many rows?")], tools=[tool])

    assert resp.finish_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.name == "sql_query"
    assert call.arguments == {"sql": "SELECT 1"}
    assert call.id.startswith("call_")


async def test_cassette_chat_http_error(
    cassette: Callable[..., CassetteTransport],
) -> None:
    transport = cassette("ollama_chat_error")
    provider = OllamaProvider(
        host="http://fake",
        model="gemma4",
        embed_model="nomic-embed-text",
        transport=transport,
    )

    with pytest.raises(ProviderError):
        await provider.chat([ChatMessage(role="user", content="hi")])


# ---------------------------------------------------------------------------
# Inline-mock tests (verify request body shape and edge cases)
# ---------------------------------------------------------------------------


async def test_chat_plain_text_response() -> None:
    transport = _chat_response(
        {
            "model": "gemma4",
            "message": {"role": "assistant", "content": "hello there"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 9,
            "eval_count": 3,
        }
    )
    provider = OllamaProvider(
        host="http://fake",
        model="gemma4",
        embed_model="nomic-embed-text",
        transport=transport,
    )

    resp = await provider.chat([ChatMessage(role="user", content="hi")])

    body = transport.captured["body"]  # type: ignore[attr-defined]
    assert transport.captured["url"] == "http://fake/api/chat"  # type: ignore[attr-defined]
    assert body["model"] == "gemma4"
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["options"]["temperature"] == 0.0
    assert "tools" not in body

    assert resp.content == "hello there"
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"
    assert resp.usage.prompt_tokens == 9
    assert resp.usage.completion_tokens == 3
    assert resp.model == "gemma4"
    assert resp.latency_ms >= 0.0


async def test_chat_with_tool_calls_synthesizes_ids() -> None:
    transport = _chat_response(
        {
            "model": "gemma4",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "sql_query", "arguments": {"sql": "SELECT 1"}}}
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 42,
            "eval_count": 7,
        }
    )
    provider = OllamaProvider(
        host="http://fake",
        model="gemma4",
        embed_model="nomic-embed-text",
        transport=transport,
    )
    tool = ToolSpec(
        name="sql_query",
        description="read-only SQL",
        parameters_schema={"type": "object", "properties": {"sql": {"type": "string"}}},
    )

    resp = await provider.chat([ChatMessage(role="user", content="how many rows?")], tools=[tool])

    body = transport.captured["body"]  # type: ignore[attr-defined]
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "sql_query",
                "description": "read-only SQL",
                "parameters": {
                    "type": "object",
                    "properties": {"sql": {"type": "string"}},
                },
            },
        }
    ]

    assert resp.finish_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.name == "sql_query"
    assert call.arguments == {"sql": "SELECT 1"}
    assert call.id.startswith("call_")
    assert len(call.id) > len("call_")


async def test_chat_max_tokens_maps_to_num_predict() -> None:
    transport = _chat_response(
        {
            "model": "gemma4",
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
            "done_reason": "length",
        }
    )
    provider = OllamaProvider(
        host="http://fake",
        model="gemma4",
        embed_model="nomic-embed-text",
        transport=transport,
    )

    resp = await provider.chat(
        [ChatMessage(role="user", content="go")], temperature=0.7, max_tokens=128
    )

    body = transport.captured["body"]  # type: ignore[attr-defined]
    assert body["options"] == {"temperature": 0.7, "num_predict": 128}
    assert resp.finish_reason == "length"


async def test_chat_http_error_wraps_in_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"}, request=request)

    provider = OllamaProvider(
        host="http://fake",
        model="gemma4",
        embed_model="nomic-embed-text",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ProviderError):
        await provider.chat([ChatMessage(role="user", content="hi")])


async def test_embed_returns_vectors() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"model": "nomic-embed-text", "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
            request=request,
        )

    provider = OllamaProvider(
        host="http://fake",
        model="gemma4",
        embed_model="nomic-embed-text",
        transport=httpx.MockTransport(handler),
    )

    vectors = await provider.embed(["hello", "world"])

    assert captured["url"] == "http://fake/api/embed"
    assert captured["body"] == {"model": "nomic-embed-text", "input": ["hello", "world"]}
    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


async def test_embed_http_error_wraps_in_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"}, request=request)

    provider = OllamaProvider(
        host="http://fake",
        model="gemma4",
        embed_model="nomic-embed-text",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ProviderError):
        await provider.embed(["hi"])


def test_ollama_satisfies_both_protocols() -> None:
    provider = OllamaProvider(host="http://fake", model="gemma4", embed_model="nomic-embed-text")
    assert isinstance(provider, ChatProvider)
    assert isinstance(provider, Embedder)


# ---------------------------------------------------------------------------
# Live tests — require a running Ollama instance (pytest -m live)
# ---------------------------------------------------------------------------

_ollama_reachable = pytest.mark.skipif(
    not os.environ.get("OLLAMA_HOST", ""),
    reason="OLLAMA_HOST not set — set it to run live Ollama tests",
)


@pytest.mark.live
@_ollama_reachable
async def test_live_ollama_chat() -> None:
    """Smoke test: send a one-turn chat to a real Ollama instance.

    Run with: OLLAMA_HOST=http://localhost:11434 pytest -m live -k ollama
    This is the call that the ollama_chat_plain cassette was derived from.
    """
    provider = OllamaProvider(
        host=os.environ["OLLAMA_HOST"],
        model="gemma4",
        embed_model="nomic-embed-text",
    )
    try:
        resp = await provider.chat(
            [ChatMessage(role="user", content="Reply with exactly one word: hello")],
            max_tokens=16,
        )
        assert resp.content.strip() != ""
        assert resp.finish_reason in ("stop", "length")
        assert resp.model == "gemma4"
    finally:
        await provider.aclose()


@pytest.mark.live
@_ollama_reachable
async def test_live_ollama_embed() -> None:
    """Smoke test: embed two strings via a real Ollama instance."""
    provider = OllamaProvider(
        host=os.environ["OLLAMA_HOST"],
        model="gemma4",
        embed_model="nomic-embed-text",
    )
    try:
        vectors = await provider.embed(["hello", "world"])
        assert len(vectors) == 2
        assert all(len(v) > 0 for v in vectors)
    finally:
        await provider.aclose()
