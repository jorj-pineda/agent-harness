"""Buffered/streamed experimental transport contracts, not model-quality tests."""

import asyncio
import json

import httpx
import pytest

from evals.projects.duodoro import transport_probe as probe
from providers.base import ChatMessage, ProviderError
from providers.ollama import OllamaProvider


class Body(httpx.AsyncByteStream):
    def __init__(self, chunks, *, error=None, hold=None):
        self.chunks = chunks
        self.error = error
        self.hold = hold
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.hold is not None:
            self.hold.set()
            await asyncio.Event().wait()
        if self.error:
            raise self.error("private detail")

    async def aclose(self):
        self.closed = True


def frame(**fields):
    return json.dumps(fields, ensure_ascii=False).encode() + b"\n"


def provider(body, *, streamed=True, status=200):
    sent = []
    observations = []

    def respond(request):
        sent.append(request)
        return httpx.Response(status, stream=body)

    transport = probe.ProbeTransport(
        httpx.MockTransport(respond), streamed=streamed, observe=observations.append
    )
    adapter = OllamaProvider(
        host="http://private",
        model="fixture",
        embed_model="unused",
        transport=transport,
        num_ctx=16384,
        think=False,
    )
    return adapter, sent, observations


@pytest.mark.parametrize("split", [1, 7, 9999])
async def test_stream_assembles_unicode_complete_tools_and_terminal_usage(split):
    tool = {
        "function": {
            "index": 0,
            "name": "write_file",
            "arguments": {"path": "a.py", "content": "🙂\n"},
        }
    }
    data = (
        frame(done=False, message={"content": "café "}, eval_count=999)
        + frame(done=False, message={"content": "🙂", "tool_calls": [tool]})
        + frame(done=True, done_reason="stop", prompt_eval_count=50, eval_count=9)
    )
    body = Body([data[i : i + split] for i in range(0, len(data), split)])
    adapter, sent, observations = provider(body)
    try:
        response = await adapter.chat([ChatMessage(role="user", content="task")], max_tokens=2048)
        assert response.content == "café 🙂"
        assert response.tool_calls[0].name == "write_file"
        assert response.tool_calls[0].arguments == tool["function"]["arguments"]
        assert response.usage.prompt_tokens == 50 and response.usage.completion_tokens == 9
        assert json.loads(sent[0].content)["stream"] is True
        assert observations[0].outcome == "completed" and observations[0].frames == 3
        assert observations[0].terminal_received and body.closed
        assert "private" not in observations[0].model_dump_json()
    finally:
        await adapter.aclose()


async def test_buffered_preserves_json_and_all_non_stream_request_fields():
    payload = {"message": {"content": "ok"}, "done": True, "prompt_eval_count": 3, "eval_count": 2}
    body = Body([json.dumps(payload).encode()])
    adapter, sent, observations = provider(body, streamed=False)
    try:
        response = await adapter.chat([ChatMessage(role="user", content="task")], max_tokens=2048)
        assert response.raw == payload and response.content == "ok"
        request = json.loads(sent[0].content)
        assert request == {
            "model": "fixture",
            "messages": [{"role": "user", "content": "task"}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 2048, "num_ctx": 16384},
            "think": False,
        }
        assert sent[0].extensions["timeout"]["read"] == 60.0
        assert observations[0].terminal_received is None and body.closed
    finally:
        await adapter.aclose()


@pytest.mark.parametrize(
    "data",
    [
        frame(done=False, message={"content": "unfinished"}),
        frame(done=True) + frame(done=True),
        frame(done=True) + b"private invalid JSON\n",
        frame(
            done=False,
            message={"tool_calls": [{"function": {"name": "write_file", "arguments": "fragment"}}]},
        ),
        frame(done=False, error="private backend error"),
        frame(done=1),
        b"\xff\n",
    ],
)
async def test_incomplete_or_invalid_stream_never_returns_partial_response(data):
    body = Body([data])
    adapter, sent, observations = provider(body)
    try:
        with pytest.raises(ProviderError) as failure:
            await adapter.chat([ChatMessage(role="user", content="task")])
        assert failure.value.failure.kind == "unknown"
        assert len(sent) == 1 and body.closed and observations[0].outcome == "protocol_error"
        assert "private" not in observations[0].model_dump_json()
    finally:
        await adapter.aclose()


async def test_only_terminal_usage_is_authoritative_and_no_final_newline_is_required():
    data = (
        frame(done=False, prompt_eval_count=999, eval_count=999, message={"content": "hello"})
        + json.dumps({"done": True, "done_reason": "length", "eval_count": None}).encode()
    )
    body = Body([data])
    adapter, _, _ = provider(body)
    try:
        response = await adapter.chat([ChatMessage(role="user", content="task")])
        assert response.usage.prompt_tokens is None and response.usage.completion_tokens is None
        assert response.finish_reason == "length"
        assert response.call_observation.prompt_usage_field == "absent"
        assert response.call_observation.completion_usage_field == "null"
    finally:
        await adapter.aclose()


@pytest.mark.parametrize("bound", ["bytes", "frames", "tools"])
async def test_assembly_bounds_close_response_without_partial_output(monkeypatch, bound):
    data = frame(done=False, message={"content": "hello"}) + frame(done=True)
    if bound == "bytes":
        monkeypatch.setattr(probe, "MAX_RESPONSE_BYTES", len(data) - 1)
    elif bound == "frames":
        monkeypatch.setattr(probe, "MAX_FRAMES", 1)
    else:
        monkeypatch.setattr(probe, "MAX_TOOL_CALLS", 1)
        tool = {"function": {"name": "read_file", "arguments": {"path": "a"}}}
        data = frame(done=True, message={"tool_calls": [tool, tool]})
    body = Body([data])
    adapter, _, obs = provider(body)
    try:
        with pytest.raises(ProviderError):
            await adapter.chat([ChatMessage(role="user", content="task")])
        assert body.closed and obs[0].outcome == "protocol_error"
    finally:
        await adapter.aclose()


async def test_partial_stream_timeout_retains_unknown_usage_and_cleans_up():
    body = Body([frame(done=False, message={"content": "partial"})], error=httpx.ReadTimeout)
    adapter, sent, obs = provider(body)
    try:
        with pytest.raises(ProviderError) as failure:
            await adapter.chat([ChatMessage(role="user", content="task")])
        assert failure.value.failure.kind == "read_timeout"
        assert failure.value.failure.call_observation.prompt_usage_field is None
        assert len(sent) == 1 and body.closed and obs[0].outcome == "transport_error"
    finally:
        await adapter.aclose()


async def test_cancel_mid_stream_closes_owned_response():
    started = asyncio.Event()
    body = Body([frame(done=False, message={"content": "partial"})], hold=started)
    adapter, _, obs = provider(body)
    try:
        task = asyncio.create_task(adapter.chat([ChatMessage(role="user", content="task")]))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert body.closed and obs[0].outcome == "cancelled"
    finally:
        await adapter.aclose()


async def test_http_failure_closes_without_parsing_private_body():
    body = Body([b"private"])
    adapter, _, obs = provider(body, status=503)
    try:
        with pytest.raises(ProviderError) as failure:
            await adapter.chat([ChatMessage(role="user", content="task")])
        assert (
            failure.value.failure.kind == "http_status" and failure.value.failure.http_status == 503
        )
        assert body.closed and obs[0].body_chunks == 0 and obs[0].outcome == "http_error"
    finally:
        await adapter.aclose()


async def test_partial_tool_call_cannot_mutate_workspace_before_terminal(tmp_path):
    from fastapi.testclient import TestClient

    from tests.api.test_session_persistence import make_app

    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "work.py"
    target.write_text("user = 1\n")
    partial = frame(
        done=False,
        message={
            "tool_calls": [
                {
                    "function": {
                        "name": "write_file",
                        "arguments": {"path": "work.py", "content": "lost user work"},
                    }
                }
            ]
        },
    )
    body = Body([partial], error=httpx.ReadTimeout)
    adapter, sent, observations = provider(body)
    # Real API lifecycle/tools, scripted HTTP body: no inference or live sockets.
    with TestClient(make_app(tmp_path, adapter), raise_server_exceptions=False) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
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
        assert saved["status"] == "failed" and saved["response"] is None
        assert saved["provider_failure"]["kind"] == "read_timeout"
        assert target.read_text() == "user = 1\n"
    assert len(sent) == 1 and body.closed and observations[0].outcome == "transport_error"


async def test_real_loopback_chunks_change_inactivity_behavior_without_changing_timeout():
    """Synthetic server measures HTTP mechanics, never calls Ollama or a model."""
    active = set()
    tasks = set()
    modes = []

    async def respond(reader, writer):
        current = asyncio.current_task()
        tasks.add(current)
        active.add(writer)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            size = next(
                int(line.split(b":", 1)[1])
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            request = json.loads(await reader.readexactly(size))
            modes.append(request["stream"])
            writer.write(
                b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: application/x-ndjson\r\n\r\n"
            )
            await writer.drain()
            if request["stream"]:
                for _ in range(12):
                    writer.write(frame(done=False, message={"content": "x"}))
                    await writer.drain()
                    await asyncio.sleep(0.1)
                writer.write(
                    frame(done=True, done_reason="stop", prompt_eval_count=3, eval_count=12)
                )
                await writer.drain()
            else:
                await asyncio.sleep(1.2)
                writer.write(
                    json.dumps(
                        {
                            "done": True,
                            "message": {"content": "x" * 12},
                            "prompt_eval_count": 3,
                            "eval_count": 12,
                        }
                    ).encode()
                )
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            writer.close()
            active.discard(writer)
            tasks.discard(current)
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    observations = []
    try:
        for streamed in [False, True]:
            transport = probe.ProbeTransport(
                httpx.AsyncHTTPTransport(), streamed=streamed, observe=observations.append
            )
            adapter = OllamaProvider(
                host=f"http://127.0.0.1:{port}",
                model="SCRIPTED",
                embed_model="unused",
                timeout_seconds=0.5,
                transport=transport,
            )
            try:
                if not streamed:
                    with pytest.raises(ProviderError) as failure:
                        await adapter.chat([ChatMessage(role="user", content="SCRIPTED")])
                    assert failure.value.failure.kind == "read_timeout"
                else:
                    response = await adapter.chat([ChatMessage(role="user", content="SCRIPTED")])
                    assert response.content == "x" * 12
                    assert response.usage.completion_tokens == 12
            finally:
                await adapter.aclose()
        assert modes == [False, True]
        assert observations[0].outcome == "transport_error"
        assert observations[1].outcome == "completed"
        assert observations[1].duration_ms > 500
        assert observations[1].terminal_received
    finally:
        server.close()
        await server.wait_closed()
        remaining = list(tasks)
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)
        assert not active
