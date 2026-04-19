"""End-to-end bridge: @tool-decorated function → registry → provider wire body.

The earlier provider tests (test_anthropic.py et al.) verify each provider's
tool serialization on a hand-built ToolSpec. This file locks in the other
half: a decorator-built tool produces a Pydantic JSON-schema that survives
every provider's wire format with descriptions, defaults, and numeric
constraints intact.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel, Field

from providers.anthropic import AnthropicProvider
from providers.base import ChatMessage
from providers.ollama import OllamaProvider
from providers.openai import OpenAIProvider
from tools import ToolRegistry, tool


class SearchArgs(BaseModel):
    query: str = Field(description="Natural-language search query")
    limit: int = Field(default=5, ge=1, le=20, description="Max results to return")


def _registry_with_search() -> ToolRegistry:
    reg = ToolRegistry()

    @tool(registry=reg)
    def search_docs(args: SearchArgs) -> str:
        """Search the support doc corpus and return top-N passages."""
        return f"searched {args.query}"

    return reg


def _mock_client(payload: dict[str, Any]) -> tuple[httpx.AsyncClient, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=payload, request=request)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), captured


def _assert_search_schema(schema: dict[str, Any]) -> None:
    assert schema["type"] == "object"
    props = schema["properties"]
    assert props["query"]["type"] == "string"
    assert props["query"]["description"] == "Natural-language search query"
    assert props["limit"]["type"] == "integer"
    assert props["limit"]["default"] == 5
    assert props["limit"]["minimum"] == 1
    assert props["limit"]["maximum"] == 20
    assert schema["required"] == ["query"]


async def test_decorator_tool_spec_round_trips_into_anthropic_request() -> None:
    reg = _registry_with_search()
    http_client, captured = _mock_client(
        {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    provider = AnthropicProvider(api_key="t", model="claude-sonnet-4-6", http_client=http_client)

    await provider.chat(
        [ChatMessage(role="user", content="hi")],
        tools=reg.as_tool_specs(),
    )

    (wire_tool,) = captured["body"]["tools"]
    assert wire_tool["name"] == "search_docs"
    assert wire_tool["description"] == "Search the support doc corpus and return top-N passages."
    _assert_search_schema(wire_tool["input_schema"])


async def test_decorator_tool_spec_round_trips_into_openai_request() -> None:
    reg = _registry_with_search()
    http_client, captured = _mock_client(
        {
            "id": "chatcmpl_01",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    provider = OpenAIProvider(api_key="t", model="gpt-4o", http_client=http_client)

    await provider.chat(
        [ChatMessage(role="user", content="hi")],
        tools=reg.as_tool_specs(),
    )

    (wire_tool,) = captured["body"]["tools"]
    assert wire_tool["type"] == "function"
    fn = wire_tool["function"]
    assert fn["name"] == "search_docs"
    assert fn["description"] == "Search the support doc corpus and return top-N passages."
    _assert_search_schema(fn["parameters"])


async def test_decorator_tool_spec_round_trips_into_ollama_request() -> None:
    reg = _registry_with_search()
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "model": "gemma4",
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
                "done_reason": "stop",
            },
            request=request,
        )

    provider = OllamaProvider(
        host="http://fake",
        model="gemma4",
        embed_model="nomic-embed-text",
        transport=httpx.MockTransport(handler),
    )

    await provider.chat(
        [ChatMessage(role="user", content="hi")],
        tools=reg.as_tool_specs(),
    )

    (wire_tool,) = captured["body"]["tools"]
    assert wire_tool["type"] == "function"
    fn = wire_tool["function"]
    assert fn["name"] == "search_docs"
    assert fn["description"] == "Search the support doc corpus and return top-N passages."
    _assert_search_schema(fn["parameters"])
