import asyncio

import httpx
import pytest

from evals.heldout_input_guard import InputPins, PinnedOllamaProvider
from evals.heldout_inputs import normalized_input
from providers.base import ChatMessage, ProviderError
from providers.ollama import OllamaProvider


@pytest.fixture
async def setup_guard():
    messages = [ChatMessage(role="user", content="inspect code")]
    wire = []

    def capture(request):
        wire.append(request.content)
        return httpx.Response(
            200,
            json={
                "model": "test",
                "message": {"content": "ok"},
                "done": True,
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )

    adapter = OllamaProvider(
        host="http://127.0.0.1:11447",
        model="test",
        embed_model="unused",
        num_ctx=16384,
        think=False,
        transport=httpx.MockTransport(capture),
    )
    try:
        await adapter.chat(messages, tools=None, temperature=0.0, max_tokens=2048)
    finally:
        await adapter.aclose()
    return messages, InputPins(normalized_input(messages, None, 0.0, 2048), wire[0])


def make(pins, respond, **overrides):
    config = {
        "pins": pins,
        "host": "http://127.0.0.1:11447",
        "model": "test",
        "num_ctx": 16384,
        "think": False,
        "timeout_seconds": 180,
        "transport": httpx.MockTransport(respond),
    }
    return PinnedOllamaProvider(**{**config, **overrides})


@pytest.mark.parametrize("change", ["messages", "temperature", "max_tokens", "tools"])
async def test_normalized_mismatch_refuses_before_adapter_transport_and_latches(
    setup_guard, change
):
    messages, pins = setup_guard
    calls = []
    guard = make(pins, lambda request: calls.append(request))
    kwargs = {"tools": None, "temperature": 0.0, "max_tokens": 2048}
    if change == "messages":
        messages = [ChatMessage(role="user", content="different")]
    elif change == "temperature":
        kwargs["temperature"] = 0.5
    elif change == "max_tokens":
        kwargs["max_tokens"] = 1024
    else:
        kwargs["tools"] = []
    try:
        with pytest.raises(ValueError, match="normalized"):
            await guard.chat(messages, **kwargs)
        with pytest.raises(ValueError):
            await guard.chat(setup_guard[0], max_tokens=2048)
        assert not calls and guard.transport.attempted_requests == 0
    finally:
        await guard.aclose()


@pytest.mark.parametrize("change", [{"model": "other"}, {"num_ctx": 8192}, {"think": True}])
async def test_wire_mismatch_refuses_before_forwarding_and_latches(setup_guard, change):
    messages, pins = setup_guard
    calls = []
    guard = make(pins, lambda request: calls.append(request), **change)
    try:
        with pytest.raises(ValueError, match="wire"):
            await guard.chat(messages, max_tokens=2048)
        with pytest.raises(ValueError):
            await guard.chat(messages, max_tokens=2048)
        assert not calls and guard.transport.attempted_requests == 0
    finally:
        await guard.aclose()


async def test_exact_first_request_passes_and_successful_followup_remains_possible(setup_guard):
    messages, pins = setup_guard
    seen = []

    def respond(request):
        seen.append(request.content)
        return httpx.Response(
            200,
            json={
                "model": "test",
                "message": {"content": "ok"},
                "done": True,
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )

    guard = make(pins, respond)
    try:
        assert (await guard.chat(messages, max_tokens=2048)).content == "ok"
        assert (
            await guard.chat([ChatMessage(role="user", content="followup")], max_tokens=1024)
        ).content == "ok"
        assert seen[0] == pins.wire and len(seen) == 2
    finally:
        await guard.aclose()
    with pytest.raises(ValueError):
        await guard.chat(messages, max_tokens=2048)


@pytest.mark.parametrize("fault", ["timeout", "cancel", "http", "json"])
async def test_uncertain_or_failed_submission_cannot_be_retried(setup_guard, fault):
    messages, pins = setup_guard
    calls = []

    def respond(request):
        calls.append(request)
        if fault == "timeout":
            raise httpx.ReadTimeout("fixture")
        if fault == "cancel":
            raise asyncio.CancelledError()
        if fault == "http":
            return httpx.Response(500)
        return httpx.Response(200, content=b"bad json")

    guard = make(pins, respond)
    try:
        with pytest.raises((ProviderError, asyncio.CancelledError, ValueError)):
            await guard.chat(messages, max_tokens=2048)
        with pytest.raises(ValueError):
            await guard.chat(messages, max_tokens=2048)
        assert len(calls) == 1 and guard.transport.attempted_requests == 1
    finally:
        await guard.aclose()


async def test_concurrent_request_cannot_pass_while_first_submission_is_unresolved(setup_guard):
    messages, pins = setup_guard
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def respond(request):
        calls.append(request)
        entered.set()
        await release.wait()
        return httpx.Response(
            200, json={"model": "test", "message": {"content": "ok"}, "done": True}
        )

    guard = make(pins, respond)
    first = asyncio.create_task(guard.chat(messages, max_tokens=2048))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        with pytest.raises(ValueError, match="unresolved"):
            await guard.chat(messages, max_tokens=2048)
        release.set()
        await first
        with pytest.raises(ValueError):
            await guard.chat(messages, max_tokens=2048)
        assert len(calls) == 1
    finally:
        release.set()
        await first
        await guard.aclose()


@pytest.mark.parametrize(
    "fault",
    ["missing_model", "empty_model", "null_model", "missing_done", "false_done", "numeric_done"],
)
async def test_native_identity_and_terminal_state_refuse_before_response_and_latch(
    setup_guard, fault
):
    messages, pins = setup_guard
    calls = []

    def respond(request):
        calls.append(request)
        payload = {
            "model": "test",
            "done": True,
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "write_file",
                            "arguments": {"path": "file.py", "content": "scripted"},
                        }
                    }
                ]
            },
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        if fault == "missing_model":
            del payload["model"]
        elif fault == "empty_model":
            payload["model"] = ""
        elif fault == "null_model":
            payload["model"] = None
        elif fault == "missing_done":
            del payload["done"]
        elif fault == "false_done":
            payload["done"] = False
        else:
            payload["done"] = 1
        return httpx.Response(200, json=payload)

    guard = make(pins, respond)
    try:
        with pytest.raises(ValueError, match="explicit model"):
            await guard.chat(messages, max_tokens=2048)
        with pytest.raises(ValueError):
            await guard.chat(messages, max_tokens=2048)
        assert len(calls) == 1 and guard.failed
    finally:
        await guard.aclose()
