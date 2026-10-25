"""Replay-state boundary contracts, not a model compatibility or quality result."""

import json

import httpx
import pytest
from pydantic import ValidationError

from harness.context import BudgetStop, TokenBudget, message_size_bytes
from harness.trace import request_identity
from providers.anthropic import AnthropicProvider
from providers.base import ChatMessage, ContinuationState, ProviderError
from providers.ollama import OllamaProvider
from providers.openai import OpenAIProvider


def state(text="synthetic replay"):
    return ContinuationState(format="openai-reasoning-content-v1", text=text)


@pytest.mark.parametrize("text", ["", "café 🙂\r\n\\literal", "x" * 65536, "🙂" * 16384])
def test_state_round_trip_is_exact_at_utf8_boundary(text):
    message = ChatMessage(role="assistant", content="answer", continuation=state(text))
    assert ChatMessage.model_validate_json(message.model_dump_json()) == message
    assert message.content == "answer" and message.continuation.text == text


@pytest.mark.parametrize("text", ["x" * 65537, "🙂" * 16385, 1, {}, ["x"]])
def test_state_rejects_oversize_or_nontext_without_coercion(text):
    with pytest.raises(ValidationError):
        state(text)


@pytest.mark.parametrize("role", ["system", "user", "tool"])
def test_only_assistant_can_carry_replay_state(role):
    with pytest.raises(ValidationError, match="only to assistant"):
        ChatMessage(role=role, continuation=state())


def test_unknown_protocol_and_extra_fields_are_not_replayable():
    with pytest.raises(ValidationError):
        ContinuationState(format="unknown", text="opaque")
    with pytest.raises(ValidationError):
        ContinuationState(format="openai-reasoning-content-v1", text="opaque", api_key="unused")


def test_legacy_message_serialization_and_identity_remain_exact():
    legacy = {
        "role": "assistant",
        "content": "answer",
        "tool_calls": [],
        "tool_call_id": None,
        "tool_name": None,
    }
    message = ChatMessage.model_validate(legacy)
    assert message.model_dump() == legacy
    assert json.loads(message.model_dump_json()) == legacy
    original = request_identity([message], [], 256)
    assert (
        request_identity([ChatMessage.model_validate_json(json.dumps(legacy))], [], 256) == original
    )
    changed = message.model_copy(update={"continuation": state()})
    assert request_identity([changed], [], 256) != original


def test_replay_state_counts_toward_context_before_dispatch():
    plain = ChatMessage(role="assistant", content="answer")
    replay = plain.model_copy(update={"continuation": state("x" * 4000)})
    assert message_size_bytes(replay) > message_size_bytes(plain) + 4000
    budget = TokenBudget(max_context=1000, min_request_output=256)
    assert not isinstance(budget.request_allowance([plain], []), BudgetStop)
    assert isinstance(budget.request_allowance([replay], []), BudgetStop)


@pytest.mark.parametrize("provider_kind", ["openai", "anthropic", "ollama", "bound_openai"])
async def test_unsupported_adapters_reject_state_before_http(provider_kind):
    seen = []
    transport = httpx.MockTransport(lambda request: seen.append(request) or httpx.Response(500))
    if provider_kind == "ollama":
        provider = OllamaProvider(
            host="http://fixture", model="fixture", embed_model="unused", transport=transport
        )
    elif provider_kind == "anthropic":
        provider = AnthropicProvider(
            api_key="fixture", model="fixture", http_client=httpx.AsyncClient(transport=transport)
        )
    else:
        provider = OpenAIProvider(
            api_key="fixture",
            model="fixture",
            http_client=httpx.AsyncClient(transport=transport),
            conversation_header="x-session" if provider_kind == "bound_openai" else None,
        )
    view = (
        provider.for_conversation("fixture-session")
        if provider_kind == "bound_openai"
        else provider
    )
    try:
        with pytest.raises(ProviderError, match="cannot replay"):
            await view.chat([ChatMessage(role="assistant", continuation=state())])
        assert seen == []
    finally:
        await provider.aclose()


async def test_default_adapter_retains_synthetic_reasoning_only_in_raw_response():
    body = {
        "id": "synthetic",
        "object": "chat.completion",
        "created": 0,
        "model": "SCRIPTED-not-Beam",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "synthetic continuation marker",
                    "tool_calls": [
                        {
                            "id": "one",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"probe.txt"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }
    provider = OpenAIProvider(
        api_key="fixture",
        model="SCRIPTED-not-Beam",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
        ),
    )
    try:
        response = await provider.chat([ChatMessage(role="user", content="synthetic probe")])
        assert (
            response.raw["choices"][0]["message"]["reasoning_content"]
            == "synthetic continuation marker"
        )
        normalized = ChatMessage(
            role="assistant", content=response.content, tool_calls=response.tool_calls
        )
        assert normalized.continuation is None
        assert normalized.content == "" and normalized.tool_calls[0].id == "one"
    finally:
        await provider.aclose()
