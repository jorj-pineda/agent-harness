"""Synthetic SDK/loop continuation contracts; no live model or quality result."""

import json
from unittest.mock import patch

import httpx
import pytest

from harness.config import Settings
from harness.configuration_review import compare_profiles
from harness.execution_profile import capture_execution_profile
from harness.loop import run_turn
from harness.providers import build_configured_provider
from harness.state import Session
from harness.trace import ResponseTrace, request_identity
from providers.base import ChatMessage, ContinuationState, ProviderError
from providers.openai import OpenAIProvider, compatible_transport_options
from tests.test_harness_loop import _echo_tool, _registry

ABSENT = object()


def reply(reasoning=ABSENT, *, call=False, finish=None, usage=True):
    message = {"role": "assistant", "content": "" if call else "done"}
    if reasoning is not ABSENT:
        message["reasoning_content"] = reasoning
    if call:
        message["tool_calls"] = [
            {
                "id": "echo-once",
                "type": "function",
                "function": {"name": "echo", "arguments": '{"text":"tool result"}'},
            }
        ]
    body = {
        "id": "fixture",
        "object": "chat.completion",
        "created": 0,
        "model": "SCRIPTED",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish or ("tool_calls" if call else "stop"),
                "message": message,
            }
        ],
    }
    if usage:
        body["usage"] = {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    return httpx.Response(200, json=body)


def provider_for(handler, enabled=True):
    return OpenAIProvider(
        api_key="synthetic-private-key",
        model="SCRIPTED",
        provider_name="openai_compatible",
        base_url="https://fixture.invalid/inference/go/openai/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **compatible_transport_options("opencode_go", replay_reasoning_content=enabled),
    )


@pytest.mark.parametrize("reasoning", [ABSENT, None, "", "café 🙂\r\n\\literal"])
async def test_opt_in_round_trip_preserves_exact_replay_without_answer_leak(reasoning):
    seen = []

    def respond(request):
        seen.append(request)
        return reply(reasoning, call=True) if len(seen) == 1 else reply("synthetic final replay")

    provider = provider_for(respond)
    session = Session()
    trace = []
    try:
        result = await run_turn(
            session=session,
            user_input="Synthetic echo",
            provider=provider.for_conversation(session.session_id),
            registry=_registry(_echo_tool()),
            trace=trace,
            max_request_output_tokens=256,
            max_context_tokens=8192,
            max_total_tokens_per_turn=64000,
        )
        assert len(seen) == 2 and result.answer == "done"
        assert result.completion_status == "completed" and result.verification_status == "not_run"
        assert all(r.headers["x-opencode-session"] == session.session_id for r in seen)
        second = json.loads(seen[1].content)
        assistant = next(m for m in second["messages"] if m["role"] == "assistant")
        assert assistant["content"] == "" and assistant["tool_calls"][0]["id"] == "echo-once"
        if reasoning is ABSENT or reasoning is None:
            assert "reasoning_content" not in assistant
            assert session.messages[1].continuation is None
        else:
            assert assistant["reasoning_content"] == reasoning
            assert session.messages[1].continuation.text == reasoning
        assert "reasoning_content" not in second["messages"][-1]
        assert Session.model_validate_json(session.model_dump_json()) == session
        responses = [r for r in trace if r.kind == "response"]
        assert responses[-1].continuation.text == "synthetic final replay"
        assert ResponseTrace.model_validate_json(responses[-1].model_dump_json()) == responses[-1]
        assert "synthetic final replay" not in result.model_dump_json()
        assert request_identity(session.messages, [], 256) != request_identity(
            [
                ChatMessage(
                    role=m.role,
                    content=m.content,
                    tool_calls=m.tool_calls,
                    tool_call_id=m.tool_call_id,
                    tool_name=m.tool_name,
                )
                for m in session.messages
            ],
            [],
            256,
        )
    finally:
        await provider.aclose()
    assert provider._client.is_closed() and provider._client.max_retries == 0


@pytest.mark.parametrize(
    "reasoning", [123, {}, ["private malformed marker"], "x" * 65537, "🙂" * 16385]
)
async def test_bad_returned_state_prevents_tool_dispatch_without_private_error_text(reasoning):
    seen = []
    provider = provider_for(lambda request: seen.append(request) or reply(reasoning, call=True))
    session = Session()
    try:
        with pytest.raises(ProviderError, match="Invalid or oversized") as error:
            await run_turn(
                session=session,
                user_input="Synthetic echo",
                provider=provider.for_conversation(session.session_id),
                registry=_registry(_echo_tool()),
            )
        assert "private malformed marker" not in str(error.value)
        assert len(seen) == 1 and session.turns[0].tool_calls == []
        assert all(m.role != "assistant" for m in session.messages)
    finally:
        await provider.aclose()


async def test_large_valid_replay_counts_before_next_model_request():
    seen = []
    provider = provider_for(lambda request: seen.append(request) or reply("x" * 6000, call=True))
    session = Session()
    try:
        result = await run_turn(
            session=session,
            user_input="Synthetic echo",
            provider=provider.for_conversation(session.session_id),
            registry=_registry(_echo_tool()),
            max_context_tokens=1000,
            max_request_output_tokens=256,
        )
        assert len(seen) == 1 and result.completion_status == "budget_exhausted"
        assert "context" in result.completion_reason
        assert session.messages[1].continuation.text == "x" * 6000
    finally:
        await provider.aclose()


@pytest.mark.parametrize(
    "finish,usage,expected", [("length", True, "incomplete"), ("stop", False, "incomplete")]
)
async def test_replay_does_not_bypass_truncation_or_usage_guards(finish, usage, expected):
    seen = []
    provider = provider_for(
        lambda request: seen.append(request) or reply("synthetic state", finish=finish, usage=usage)
    )
    session = Session()
    try:
        result = await run_turn(
            session=session,
            user_input="Synthetic",
            provider=provider.for_conversation(session.session_id),
            registry=_registry(),
            max_request_output_tokens=256,
        )
        assert result.completion_status == expected and len(seen) == 1
        assert session.messages[-1].continuation.text == "synthetic state"
    finally:
        await provider.aclose()


def test_factory_opt_in_and_profile_drift_do_not_change_legacy_recipes():
    base = {
        "_env_file": None,
        "openai_compatible_base_url": "https://fixture.invalid/v1",
        "openai_compatible_model": "SCRIPTED",
        "openai_compatible_api_key": "private-key",
        "openai_compatible_transport": "opencode_go",
    }
    off = Settings(**base)
    on = Settings(**base, openai_compatible_replay_reasoning_content=True)
    with patch("harness.providers.create_chat_provider") as factory:
        build_configured_provider("openai_compatible", on)
    assert factory.call_args.kwargs["replay_reasoning_content"] is True
    with patch("harness.providers.create_chat_provider") as factory:
        build_configured_provider("openai_compatible", off)
    assert "replay_reasoning_content" not in factory.call_args.kwargs
    old_options = {
        "default_headers": {"User-Agent": "agent-harness/0.1"},
        "conversation_header": "x-opencode-session",
        "max_retries": 0,
    }
    assert compatible_transport_options("opencode_go") == old_options
    registry = _registry()
    saved = capture_execution_profile(off, "openai_compatible", registry)
    current = capture_execution_profile(on, "openai_compatible", registry)
    assert compare_profiles(saved, current).changed_fields == ["provider_options_sha256"]
    assert capture_execution_profile(off, "ollama", registry) == capture_execution_profile(
        on, "ollama", registry
    )
    assert "private-key" not in current.model_dump_json()


async def test_default_refuses_supplied_state_and_does_not_enable_replay():
    seen = []
    provider = provider_for(
        lambda request: seen.append(request) or reply("synthetic state"), enabled=False
    )
    try:
        view = provider.for_conversation("session")
        response = await view.chat([ChatMessage(role="user", content="task")])
        assert response.continuation is None and "continuation" not in response.model_dump()
        with pytest.raises(ProviderError, match="cannot replay"):
            await view.chat(
                [
                    ChatMessage(
                        role="assistant",
                        continuation=ContinuationState(
                            format="openai-reasoning-content-v1", text="replay"
                        ),
                    )
                ]
            )
        assert len(seen) == 1
    finally:
        await provider.aclose()
