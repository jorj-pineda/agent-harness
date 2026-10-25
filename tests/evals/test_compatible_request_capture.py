"""Actual compatible SDK/HTTP paths with canned responses, never hosted inference."""

import asyncio
import json

import httpx
import pytest

from evals.projects.duodoro.provider_probe import FirstRequestGuard
from evals.projects.duodoro.request_capture import (
    CapturedProvider,
    CapturedTransport,
    CaptureLimits,
    RequestCapture,
)
from harness.trace import request_identity
from providers.base import ChatMessage, ProviderError, ToolCall, ToolSpec
from providers.openai import OpenAIProvider, compatible_transport_options


def completed():
    return httpx.Response(
        200,
        json={
            "id": "fixture-response",
            "object": "chat.completion",
            "created": 0,
            "model": "SCRIPTED",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        },
    )


def setup(tmp_path, respond, *, prefix="/v1", limits=None):
    capture = RequestCapture(
        tmp_path.resolve() / "inputs", chat_path=prefix + "/chat/completions", limits=limits
    )
    transport = CapturedTransport(httpx.MockTransport(respond), capture)
    adapter = OpenAIProvider(
        api_key="private-api-key",
        model="SCRIPTED",
        provider_name="openai_compatible",
        base_url="https://private-endpoint.invalid" + prefix,
        timeout_seconds=180,
        http_client=httpx.AsyncClient(transport=transport),
        **compatible_transport_options("opencode_go"),
    )
    return CapturedProvider(adapter, capture), capture, adapter


@pytest.mark.parametrize("prefix", ["/v1", "/inference/go/v1", "/inference/go/openai/v1"])
async def test_compatible_sdk_records_exact_bytes_and_preserves_bound_routing(tmp_path, prefix):
    messages = [
        ChatMessage(role="user", content="café 🙂\n"),
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="call-id", name="read", arguments={"path": "a.py"})],
        ),
        ChatMessage(role="tool", content="result", tool_call_id="call-id", tool_name="read"),
    ]
    tools = [ToolSpec(name="read", description="fixture", parameters_schema={"type": "object"})]
    before = [m.model_dump() for m in messages]
    seen = []

    def respond(request):
        seen.append(request)
        number = len(seen)
        normalized = json.loads(
            (capture.directory / f"request-{number:04d}-normalized.json").read_text()
        )
        assert normalized["messages"] == before
        assert normalized["identity"] == request_identity(messages, tools, 2048).model_dump()
        assert (
            capture.directory / f"request-{number:04d}-wire.json"
        ).read_bytes() == request.content
        assert request.url.path == prefix + "/chat/completions"
        assert request.headers["Authorization"] == "Bearer private-api-key"
        assert request.headers["x-opencode-session"] == "private-session-a"
        assert request.headers["User-Agent"] == "agent-harness/0.1"
        assert request.extensions["timeout"]["read"] == 180
        return completed()

    provider, capture, adapter = setup(tmp_path, respond, prefix=prefix)
    guard = FirstRequestGuard(provider, request_identity(messages, tools, 2048).request_sha256)
    first = guard.for_conversation("private-session-a")
    other = guard.for_conversation("private-session-b")
    assert list(capture.directory.iterdir()) == []
    assert "x-opencode-session" not in adapter._client.default_headers
    try:
        response = await first.chat(messages, tools=tools, max_tokens=2048)
        assert response.content == "ok" and response.usage.prompt_tokens == 10
        # A fresh pure view for the same conversation retains its header/recorder.
        await guard.for_conversation("private-session-a").chat(
            messages, tools=tools, max_tokens=2048
        )
        evidence = {p.name: p.read_bytes() for p in capture.directory.iterdir()}
        with pytest.raises(ValueError, match="cannot mix conversations"):
            await other.chat(messages, tools=tools, max_tokens=2048)
        with pytest.raises(ValueError, match="cannot mix conversations"):
            await provider.chat(messages, tools=tools, max_tokens=2048)
        assert guard.forwarded_requests == 2
        assert (
            len(seen) == 2
            and {p.name: p.read_bytes() for p in capture.directory.iterdir()} == evidence
        )
        assert [m.model_dump() for m in messages] == before
        wire = json.loads(seen[0].content)
        assert (
            wire["model"] == "SCRIPTED" and wire["max_tokens"] == 2048 and wire["temperature"] == 0
        )
        assert json.loads(wire["messages"][1]["tool_calls"][0]["function"]["arguments"]) == {
            "path": "a.py"
        }
        assert wire["messages"][2]["tool_call_id"] == "call-id"
        text = b"".join(evidence.values()).decode()
        for private in [
            "private-api-key",
            "private-session-a",
            "private-session-b",
            "private-endpoint",
        ]:
            assert private not in text
        assert capture.directory.stat().st_mode & 0o777 == 0o700
        assert all(p.stat().st_mode & 0o777 == 0o600 for p in capture.directory.iterdir())
        assert adapter._client.max_retries == 0
        assert "x-opencode-session" not in adapter._client.default_headers
    finally:
        await provider.aclose()
    assert adapter._client.is_closed()


@pytest.mark.parametrize(
    "path",
    ["/models", "/v1/responses", "https://host/v1/chat/completions", "/v1/../chat/completions"],
)
def test_unsupported_chat_path_rejects_before_directory_creation(tmp_path, path):
    with pytest.raises(ValueError, match="Unsupported"):
        RequestCapture(tmp_path / "inputs", chat_path=path)
    assert not (tmp_path / "inputs").exists()


@pytest.mark.parametrize(
    "method,path",
    [("GET", "/v1/chat/completions"), ("POST", "/api/chat"), ("POST", "/v1/completions")],
)
async def test_selected_compatible_path_mismatch_never_forwards(tmp_path, method, path):
    capture = RequestCapture(tmp_path / "inputs", chat_path="/v1/chat/completions")
    capture.begin([], None, 0.0, 2048)
    seen = []
    transport = CapturedTransport(
        httpx.MockTransport(lambda request: seen.append(request) or completed()), capture
    )
    try:
        with pytest.raises(ValueError, match="chat requests only"):
            await transport.handle_async_request(
                httpx.Request(method, "https://fixture.invalid" + path, json={})
            )
        assert not seen and len(list(capture.directory.iterdir())) == 1
        capture.end(returned=False)
        with pytest.raises(ValueError, match="Failed"):
            capture.begin([], None, 0.0, 2048)
    finally:
        await transport.aclose()


@pytest.mark.parametrize("bound", ["file", "total", "count"])
async def test_compatible_bounds_block_before_sdk_forwarding(tmp_path, bound):
    seen = []
    limits = CaptureLimits(
        max_file_bytes=1 if bound == "file" else 100000,
        max_total_bytes=1 if bound == "total" else 100000,
        max_requests=1,
    )
    provider, _capture, _ = setup(
        tmp_path, lambda request: seen.append(request) or completed(), limits=limits
    )
    bound_provider = provider.for_conversation("private-session")
    try:
        if bound == "count":
            await bound_provider.chat([ChatMessage(role="user", content="task")])
        with pytest.raises(ValueError, match="bound"):
            await bound_provider.chat([ChatMessage(role="user", content="task")])
        assert len(seen) == (1 if bound == "count" else 0)
    finally:
        await provider.aclose()


async def test_compatible_failure_records_input_once_without_headers_or_retry(tmp_path):
    seen = []

    def fail(request):
        seen.append(request)
        raise httpx.ReadTimeout("private-failure-detail", request=request)

    provider, capture, adapter = setup(tmp_path, fail)
    try:
        with pytest.raises(ProviderError):
            await provider.for_conversation("private-session").chat(
                [ChatMessage(role="user", content="task")]
            )
        assert (
            len(seen) == 1
            and adapter._client.max_retries == 0
            and len(list(capture.directory.iterdir())) == 2
        )
        text = "".join(p.read_text() for p in capture.directory.iterdir())
        assert (
            "private-failure-detail" not in text
            and "private-api-key" not in text
            and "private-session" not in text
        )
    finally:
        await provider.aclose()


async def test_compatible_wire_storage_failure_poison_blocks_further_forwarding(
    tmp_path, monkeypatch
):
    seen = []
    provider, capture, _ = setup(tmp_path, lambda request: seen.append(request) or completed())
    original = capture._write

    def fail(suffix, data):
        if suffix == "wire":
            raise OSError("fixture-storage-failure")
        original(suffix, data)

    monkeypatch.setattr(capture, "_write", fail)
    bound = provider.for_conversation("private-session")
    try:
        with pytest.raises(ProviderError):
            await bound.chat([ChatMessage(role="user", content="task")])
        assert not seen and len(list(capture.directory.iterdir())) == 1
        with pytest.raises(ValueError, match="Failed"):
            await bound.chat([ChatMessage(role="user", content="task")])
    finally:
        await provider.aclose()


async def test_compatible_cancellation_retains_pair_and_closes_body(tmp_path):
    started = asyncio.Event()
    closed = []

    class Waiting(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b"unreachable"

        async def aclose(self):
            closed.append(True)

    provider, capture, _ = setup(tmp_path, lambda request: httpx.Response(200, stream=Waiting()))
    task = asyncio.create_task(
        provider.for_conversation("private-session").chat(
            [ChatMessage(role="user", content="task")]
        )
    )
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed and len(list(capture.directory.iterdir())) == 2
    finally:
        await provider.aclose()


async def test_invalid_conversation_binding_does_not_start_recording(tmp_path):
    provider, capture, adapter = setup(
        tmp_path, lambda request: pytest.fail("Invalid binding dispatched")
    )
    try:
        with pytest.raises(ValueError, match="Conversation ID"):
            provider.for_conversation("invalid session with spaces")
        assert list(capture.directory.iterdir()) == []
        assert "x-opencode-session" not in adapter._client.default_headers
    finally:
        await provider.aclose()


async def test_bound_first_input_and_review_guards_still_prevent_recording(tmp_path):
    seen = []
    provider, capture, _ = setup(tmp_path, lambda request: seen.append(request) or completed())
    messages = [ChatMessage(role="user", content="original")]
    expected = request_identity(messages, [], 2048).request_sha256
    guard = FirstRequestGuard(provider, expected)
    view = guard.for_conversation("private-session")
    review = FirstRequestGuard(provider, expected, review_only=True)
    try:
        with pytest.raises(ValueError, match="first-input mismatch"):
            await view.chat([ChatMessage(role="user", content="changed")], max_tokens=2048)
        with pytest.raises(ValueError, match="Review-only"):
            await review.for_conversation("private-session").chat(messages, max_tokens=2048)
        assert not seen and list(capture.directory.iterdir()) == []
        assert guard.forwarded_requests == review.forwarded_requests == 0
        await view.chat(messages, max_tokens=2048)
        assert len(seen) == 1 and guard.forwarded_requests == 1
    finally:
        await guard.aclose()


def test_compatible_api_binding_edit_failure_and_restart_preserve_evidence(tmp_path):
    from fastapi.testclient import TestClient

    from tests.api.test_session_persistence import make_app

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "work.py"
    target.write_text("user = 1\n")
    target.chmod(0o640)
    (workspace / "preexisting.txt").write_text("keep\n")
    seen = []

    def respond(request):
        seen.append(request)
        assert request.headers["x-opencode-session"] == sid
        if len(seen) == 2:
            raise httpx.ReadTimeout("private-fixture-failure", request=request)
        body = completed().json()
        body["choices"][0] = {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "write-once",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps(
                                {"path": "work.py", "content": "user = 1\nagent = 2\n"}
                            ),
                        },
                    }
                ],
            },
        }
        return httpx.Response(200, json=body)

    provider, capture, adapter = setup(tmp_path, respond, prefix="/inference/go/v1")
    with TestClient(make_app(tmp_path, provider), raise_server_exceptions=False) as client:
        sid = client.post(
            "/sessions", json={"user_id": "dev", "workspace_root": str(workspace)}
        ).json()["session_id"]
        result = client.post(
            "/chat",
            json={"user_id": "dev", "session_id": sid, "message": "Edit", "request_id": "one"},
        )
        assert result.status_code == 500
        saved = client.get("/runs", params={"user_id": "dev"}).json()[0]
        assert saved["status"] == "failed" and saved["response"] is None
        events = saved["budget_evidence"]["events"]
        requests = [e for e in events if e["kind"] == "request"]
        records = [
            json.loads((capture.directory / f"request-{n:04d}-normalized.json").read_text())
            for n in [1, 2]
        ]
        assert [r["identity"] for r in records] == [e["identity"] for e in requests]
        assert events[-1]["kind"] == "request_failure" and events[-1]["usage"] is None
    assert adapter._client.is_closed() and len(seen) == 2
    evidence = {p.name: p.read_bytes() for p in capture.directory.iterdir()}
    assert len(evidence) == 4
    assert target.read_text() == "user = 1\nagent = 2\n"
    assert target.stat().st_mode & 0o777 == 0o640
    assert (workspace / "preexisting.txt").read_text() == "keep\n"
    text = b"".join(evidence.values()).decode()
    assert (
        sid not in text and "private-api-key" not in text and "private-fixture-failure" not in text
    )
    review = OpenAIProvider(
        api_key="unused-private-key",
        model="SCRIPTED",
        base_url="https://review.invalid/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: pytest.fail("Review dispatched"))
        ),
        **compatible_transport_options("opencode_go"),
    )
    with TestClient(make_app(tmp_path, review)) as client:
        assert client.get(f"/runs/{saved['run_id']}", params={"user_id": "dev"}).json() == saved
    assert {p.name: p.read_bytes() for p in capture.directory.iterdir()} == evidence
    assert target.read_text() == "user = 1\nagent = 2\n"
