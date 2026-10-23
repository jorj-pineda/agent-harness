"""Input retention/forwarding contracts, not inference or request-cause diagnosis."""

import asyncio
import json
import stat

import httpx
import pytest

from evals.projects.duodoro.request_capture import (
    CapturedProvider,
    CapturedTransport,
    CaptureLimits,
    RequestCapture,
)
from harness.trace import request_identity
from providers.base import ChatMessage, ProviderError, ToolCall, ToolSpec
from providers.ollama import OllamaProvider


def setup(tmp_path, respond, *, limits=None):
    capture = RequestCapture(tmp_path.resolve() / "inputs", limits=limits)
    transport = CapturedTransport(httpx.MockTransport(respond), capture)
    adapter = OllamaProvider(
        host="http://private-endpoint",
        model="SCRIPTED",
        embed_model="unused",
        num_ctx=16384,
        think=False,
        transport=transport,
    )
    return CapturedProvider(adapter, capture), capture


def completed():
    return httpx.Response(
        200,
        json={"message": {"content": "ok"}, "prompt_eval_count": 10, "eval_count": 2},
    )


async def test_actual_adapter_exact_normalized_and_wire_before_forwarding(tmp_path):
    messages = [
        ChatMessage(role="user", content="café 🙂\n"),
        ChatMessage(
            role="assistant", tool_calls=[ToolCall(id="preserve-id", name="read", arguments={})]
        ),
        ChatMessage(role="tool", content="result", tool_call_id="preserve-id", tool_name="read"),
    ]
    tools = [ToolSpec(name="read", description="fixture", parameters_schema={"type": "object"})]
    before = [m.model_dump() for m in messages]
    sent = []

    def respond(request):
        sent.append(request)
        # Evidence is closed and readable before the first network action.
        normalized = json.loads((capture.directory / "request-0001-normalized.json").read_bytes())
        assert normalized["messages"] == before
        assert normalized["identity"] == request_identity(messages, tools, 2048).model_dump()
        assert normalized["temperature"] == 0.0 and normalized["max_tokens"] == 2048
        assert normalized["tools"] == [tool.model_dump() for tool in tools]
        assert (capture.directory / "request-0001-wire.json").read_bytes() == request.content
        assert request.extensions["timeout"]["read"] == 60.0
        return completed()

    provider, capture = setup(tmp_path, respond)
    try:
        response = await provider.chat(messages, tools=tools, max_tokens=2048)
        assert response.content == "ok" and response.usage.prompt_tokens == 10
        assert response.raw["message"]["content"] == "ok"
        assert [m.model_dump() for m in messages] == before
        wire = json.loads(sent[0].content)
        assert wire["stream"] is False and wire["think"] is False
        assert wire["options"] == {"temperature": 0.0, "num_predict": 2048, "num_ctx": 16384}
        assert capture.requests == 1 and capture.stored_bytes == sum(
            p.stat().st_size for p in capture.directory.iterdir()
        )
        assert stat.S_IMODE(capture.directory.stat().st_mode) == 0o700
        assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in capture.directory.iterdir())
        assert "private-endpoint" not in "".join(p.read_text() for p in capture.directory.iterdir())
    finally:
        await provider.aclose()


async def test_failed_fifth_request_retained_with_real_normalized_history(tmp_path):
    seen = []

    def respond(request):
        seen.append(request)
        if len(seen) == 5:
            assert (capture.directory / "request-0005-wire.json").read_bytes() == request.content
            raise httpx.ReadTimeout("not retained in input evidence", request=request)
        return completed()

    provider, capture = setup(tmp_path, respond)
    messages = [ChatMessage(role="user", content="task")]
    try:
        for _ in range(4):
            response = await provider.chat(messages, max_tokens=2048)
            messages.extend([ChatMessage(role="assistant", content=response.content)])
        with pytest.raises(ProviderError) as error:
            await provider.chat(messages, max_tokens=2048)
        assert error.value.failure.kind == "read_timeout"
        saved = json.loads((capture.directory / "request-0005-normalized.json").read_bytes())
        assert saved["identity"] == request_identity(messages, [], 2048).model_dump()
        assert saved["messages"] == [m.model_dump() for m in messages]
        assert len(list(capture.directory.iterdir())) == 10 and len(seen) == 5
        assert all("not retained" not in p.read_text() for p in capture.directory.iterdir())
    finally:
        await provider.aclose()


@pytest.mark.parametrize("which", ["file", "total", "count"])
async def test_bounds_stop_before_another_inner_dispatch(tmp_path, which):
    seen = []
    limits = CaptureLimits(
        max_file_bytes=1 if which == "file" else 100000,
        max_total_bytes=1 if which == "total" else 100000,
        max_requests=1,
    )
    provider, _capture = setup(
        tmp_path, lambda request: seen.append(request) or completed(), limits=limits
    )
    try:
        if which == "count":
            await provider.chat([ChatMessage(role="user", content="task")])
        with pytest.raises(ValueError, match="bound"):
            await provider.chat([ChatMessage(role="user", content="task")])
        assert len(seen) == (1 if which == "count" else 0)
    finally:
        await provider.aclose()


async def test_wire_bound_and_recording_io_error_never_forward(tmp_path, monkeypatch):
    seen = []
    provider, capture = setup(tmp_path, lambda request: seen.append(request) or completed())
    original = capture._write

    def fail_wire(suffix, data):
        if suffix == "wire":
            raise OSError("SCRIPTED storage failure")
        original(suffix, data)

    monkeypatch.setattr(capture, "_write", fail_wire)
    try:
        with pytest.raises(OSError):
            await provider.chat([ChatMessage(role="user", content="task")])
        assert not seen
        assert (capture.directory / "request-0001-normalized.json").exists()
        with pytest.raises(ValueError, match="Failed"):
            await provider.chat([ChatMessage(role="user", content="task")])
    finally:
        await provider.aclose()


async def test_cancellation_retains_input_and_delegates_response_cleanup(tmp_path):
    started = asyncio.Event()
    closed = []

    class Waiting(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b"unreachable"

        async def aclose(self):
            closed.append(True)

    provider, capture = setup(tmp_path, lambda request: httpx.Response(200, stream=Waiting()))
    task = asyncio.create_task(provider.chat([ChatMessage(role="user", content="task")]))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(list(capture.directory.iterdir())) == 2 and closed
    finally:
        await provider.aclose()


def test_existing_or_noncanonical_directory_refused_without_overwrite(tmp_path):
    path = tmp_path.resolve() / "inputs"
    RequestCapture(path)
    (path / "sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        RequestCapture(path)
    assert (path / "sentinel").read_text() == "keep"
    with pytest.raises(ValueError):
        RequestCapture(path / ".." / "other")


async def test_concurrent_call_rejects_without_touching_first_capture(tmp_path):
    started = asyncio.Event()
    finish = asyncio.Event()

    async def respond(request):
        started.set()
        await finish.wait()
        return completed()

    provider, capture = setup(tmp_path, respond)
    first = asyncio.create_task(provider.chat([ChatMessage(role="user", content="first")]))
    try:
        await asyncio.wait_for(started.wait(), 2)
        with pytest.raises(ValueError, match="overlapping"):
            await provider.chat([ChatMessage(role="user", content="second")])
        finish.set()
        assert (await first).content == "ok" and capture.requests == 1
    finally:
        finish.set()
        await first
        await provider.aclose()


async def test_unsupported_identity_stops_without_forward_or_lossy_encoding(tmp_path):
    seen = []
    provider, capture = setup(tmp_path, lambda request: seen.append(request) or completed())
    message = ChatMessage(
        role="assistant",
        tool_calls=[ToolCall(id="id", name="fixture", arguments={"bad": object()})],
    )
    try:
        with pytest.raises(ValueError, match="Unsupported"):
            await provider.chat([message])
        assert not seen and list(capture.directory.iterdir()) == []
    finally:
        await provider.aclose()


@pytest.mark.parametrize("bound", ["file", "total"])
async def test_wire_record_bound_after_normalized_record_blocks_network(tmp_path, bound):
    seen = []
    capture = RequestCapture(tmp_path.resolve() / "inputs")
    capture.begin([ChatMessage(role="user", content="task")], None, 0.0, 2048)
    normalized_size = capture.stored_bytes
    data = b'{"message":"' + b"x" * 1000 + b'"}'
    capture.limits = CaptureLimits(
        max_file_bytes=len(data) - 1 if bound == "file" else len(data),
        max_total_bytes=normalized_size + len(data) - 1 if bound == "total" else 100000,
    )
    transport = CapturedTransport(
        httpx.MockTransport(lambda request: seen.append(request) or completed()), capture
    )
    try:
        with pytest.raises(ValueError, match="byte bound"):
            await transport.handle_async_request(
                httpx.Request("POST", "http://fixture/api/chat", content=data)
            )
        assert not seen and len(list(capture.directory.iterdir())) == 1
    finally:
        capture.end(returned=False)
        await transport.aclose()


async def test_same_request_object_forwarded_headers_and_url_not_recorded(tmp_path):
    capture = RequestCapture(tmp_path.resolve() / "inputs")
    capture.begin([], [], 0.0, None)
    request = httpx.Request(
        "POST",
        "http://private-endpoint/api/chat",
        content=b'{ "messages" : [] }',
        headers={"Authorization": "private-header"},
        extensions={"timeout": {"read": 60}},
    )
    seen = []
    transport = CapturedTransport(
        httpx.MockTransport(lambda item: seen.append(item) or completed()), capture
    )
    try:
        await transport.handle_async_request(request)
        capture.end(returned=True)
        assert seen == [request] and seen[0] is request
        files = "".join(p.read_text() for p in capture.directory.iterdir())
        assert "private-header" not in files and "private-endpoint" not in files
        assert (capture.directory / "request-0001-wire.json").read_bytes() == request.content
    finally:
        await transport.aclose()


async def test_duplicate_wire_pair_cannot_forward_twice(tmp_path):
    capture = RequestCapture(tmp_path.resolve() / "inputs")
    capture.begin([], None, 0.0, None)
    seen = []
    transport = CapturedTransport(
        httpx.MockTransport(lambda item: seen.append(item) or completed()), capture
    )
    request = httpx.Request("POST", "http://fixture/api/chat", json={})
    try:
        await transport.handle_async_request(request)
        with pytest.raises(ValueError, match="duplicate"):
            await transport.handle_async_request(request)
        # Even an inner provider swallowing the capture error cannot return tools.
        with pytest.raises(ValueError, match="without paired"):
            capture.end(returned=True)
        assert len(seen) == 1
    finally:
        await transport.aclose()


async def test_missing_wire_record_cannot_return_proposed_tools(tmp_path):
    from providers.base import ProviderResponse

    class Inner:
        name = "fixture"
        closed = False

        async def chat(self, *args, **kwargs):
            return ProviderResponse(
                content="",
                tool_calls=[ToolCall(id="id", name="write_file", arguments={})],
                finish_reason="tool_use",
                model="SCRIPTED",
                latency_ms=0,
            )

        async def aclose(self):
            self.closed = True

    inner = Inner()
    capture = RequestCapture(tmp_path.resolve() / "inputs")
    provider = CapturedProvider(inner, capture)
    with pytest.raises(ValueError, match="without paired"):
        await provider.chat([ChatMessage(role="user", content="task")])
    await provider.aclose()
    assert inner.closed and len(list(capture.directory.iterdir())) == 1


def test_api_edit_then_failure_retains_inputs_and_restart_is_review_only(tmp_path):
    from fastapi.testclient import TestClient

    from tests.api.test_session_persistence import make_app

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "work.py").write_text("user = 1\n")
    (workspace / "preexisting.txt").write_text("keep\n")
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 2:
            raise httpx.ReadTimeout("SCRIPTED", request=request)
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "write_file",
                                "arguments": {
                                    "path": "work.py",
                                    "content": "user = 1\nagent = 2\n",
                                },
                            }
                        }
                    ],
                },
                "prompt_eval_count": 100,
                "eval_count": 20,
            },
        )

    provider, capture = setup(tmp_path, respond)
    with TestClient(make_app(tmp_path, provider), raise_server_exceptions=False) as client:
        sid = client.post(
            "/sessions", json={"user_id": "dev", "workspace_root": str(workspace)}
        ).json()["session_id"]
        result = client.post(
            "/chat",
            json={
                "user_id": "dev",
                "session_id": sid,
                "message": "Bounded edit",
                "request_id": "one",
            },
        )
        assert result.status_code == 500
        saved = client.get("/runs", params={"user_id": "dev"}).json()[0]
        assert saved["status"] == "failed" and saved["provider_failure"]["kind"] == "read_timeout"
        assert saved["response"] is None
        requests = [e for e in saved["budget_evidence"]["events"] if e["kind"] == "request"]
        records = [
            json.loads((capture.directory / f"request-{number:04d}-normalized.json").read_bytes())
            for number in [1, 2]
        ]
        assert [e["identity"] for e in requests] == [value["identity"] for value in records]
    evidence = {p.name: p.read_bytes() for p in capture.directory.iterdir()}
    assert len(evidence) == 4 and len(calls) == 2
    assert (workspace / "work.py").read_text() == "user = 1\nagent = 2\n"
    assert (workspace / "preexisting.txt").read_text() == "keep\n"
    # Saved GET inspection uses no observer allocation and cannot append inputs.
    review = OllamaProvider(
        host="http://fixture",
        model="SCRIPTED",
        embed_model="unused",
        transport=httpx.MockTransport(lambda request: pytest.fail("Review dispatched inference")),
    )
    with TestClient(make_app(tmp_path, review)) as client:
        assert client.get(f"/runs/{saved['run_id']}", params={"user_id": "dev"}).json() == saved
    assert {p.name: p.read_bytes() for p in capture.directory.iterdir()} == evidence
