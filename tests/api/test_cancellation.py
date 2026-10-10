from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from providers.base import ToolCall

from .conftest import Harness, ScriptedProvider, make_response
from .test_session_persistence import make_app
from .test_stream import _parse_sse


@pytest.mark.parametrize("transport", ["http", "stream"])
def test_cancel_persists_partial_review_and_retires_session(
    harness: Harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("user = 1\n")
    sid = harness.client.post(
        "/sessions", json={"user_id": "dev", "workspace_root": str(repo)}
    ).json()["session_id"]
    started = threading.Event()
    original = harness.provider.chat
    harness.provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "a.py", "content": "user = 1\nagent = 2\n"},
                )
            ],
            prompt_tokens=10,
            completion_tokens=5,
        )
    )

    async def hold(*args, **kwargs):
        if harness.provider.calls:
            started.set()
            await asyncio.Event().wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(harness.provider, "chat", hold)
    payload = {"user_id": "dev", "session_id": sid, "message": "Update a.py"}

    def submit():
        if transport == "http":
            result = harness.client.post("/chat", json=payload)
            assert result.status_code == 200, result.text
            return result.json()
        result = harness.client.get("/chat/stream", params=payload)
        assert result.status_code == 200, result.text
        return json.loads(_parse_sse(result.text)[-1]["data"])["response"]

    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(submit)
        try:
            assert started.wait(2)
            assert (
                harness.client.post(
                    f"/sessions/{sid}/cancel", json={"user_id": "other"}
                ).status_code
                == 403
            )
            response = harness.client.post(f"/sessions/{sid}/cancel", json={"user_id": "dev"})
            assert response.status_code == 202, response.text
            assert response.json()["status"] == "requested"
        finally:
            response = active.result(timeout=5)
    assert response["completion_status"] == "cancelled"
    assert response["escalated"] is True
    assert "+agent = 2\n" in response["workspace_changes"]["diffs"][0]["diff"]
    assert response["token_usage"]["completion_tokens"] is None  # Interrupted model usage unknown.
    saved = harness.client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
    assert saved["responses"][0]["response"] == response
    assert saved["session"]["turns"][0]["finished_at"] is not None
    assert harness.client.post("/chat", json=payload).status_code == 409
    assert harness.client.get("/chat/stream", params=payload).status_code == 409
    assert len(harness.provider.calls) == 1
    # Separate application lifecycle reads the same cancellation envelope.
    with TestClient(make_app(tmp_path, ScriptedProvider())) as client:
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved


def test_cancel_unknown_idle_and_archived(harness: Harness) -> None:
    assert (
        harness.client.post("/sessions/missing/cancel", json={"user_id": "dev"}).status_code == 404
    )
    sid = harness.client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
    assert (
        harness.client.post(f"/sessions/{sid}/cancel", json={"user_id": "dev"}).status_code == 409
    )
    assert harness.client.post(f"/sessions/{sid}/cancel", json={"user_id": ""}).status_code == 422


@pytest.mark.parametrize("transport", ["http", "stream"])
async def test_disconnect_keeps_owned_turn_available_for_explicit_cancel(
    tmp_path: Path, transport: str
) -> None:
    import httpx

    from tests.test_turn_admission import HoldingProvider

    provider = HoldingProvider()
    app = make_app(tmp_path, provider)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        sid = (await client.post("/sessions", json={"user_id": "dev"})).json()["session_id"]
        payload = {"user_id": "dev", "session_id": sid, "message": "Explain"}
        if transport == "http":
            request_task = asyncio.create_task(client.post("/chat", json=payload))
            await asyncio.wait_for(provider.started.wait(), 2)
            request_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request_task
        else:
            from urllib.parse import urlencode

            receive_queue = asyncio.Queue()
            await receive_queue.put({"type": "http.request", "body": b"", "more_body": False})
            sent = []

            async def send(event):
                sent.append(event)

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/chat/stream",
                "raw_path": b"/chat/stream",
                "query_string": urlencode(payload).encode(),
                "root_path": "",
                "headers": [],
                "client": ("127.0.0.1", 123),
                "server": ("test", 80),
            }
            request_task = asyncio.create_task(app(scope, receive_queue.get, send))
            await asyncio.wait_for(provider.started.wait(), 2)
            await receive_queue.put({"type": "http.disconnect"})
            await asyncio.wait_for(request_task, 2)
            assert sent[0]["status"] == 200
        assert sid in app.state.components.cancellations
        cancel = await client.post(f"/sessions/{sid}/cancel", json={"user_id": "dev"})
        assert cancel.status_code == 202
        await asyncio.wait_for(asyncio.gather(*app.state.components.tasks), 2)
        saved = (await client.get(f"/sessions/{sid}", params={"user_id": "dev"})).json()
        assert saved["responses"][0]["response"]["completion_status"] == "cancelled"
        assert sid not in app.state.components.sessions


async def test_shutdown_cancels_detached_turn_before_closing_storage(tmp_path: Path) -> None:
    import httpx

    from harness.session_store import SessionStore
    from tests.test_turn_admission import HoldingProvider

    provider = HoldingProvider()
    app = make_app(tmp_path, provider)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        sid = (await client.post("/sessions", json={"user_id": "dev"})).json()["session_id"]
        request_task = asyncio.create_task(
            client.post("/chat", json={"user_id": "dev", "session_id": sid, "message": "Explain"})
        )
        await asyncio.wait_for(provider.started.wait(), 2)
        request_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request_task
    store = SessionStore(tmp_path / "sessions.db")
    try:
        saved = store.get(sid)
        assert saved is not None
        assert saved.responses[0].response.completion_status == "cancelled"
        assert not app.state.components.tasks
    finally:
        store.close()


def test_cancel_during_final_review_is_too_late(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import harness.runtime as runtime

    started, release = threading.Event(), threading.Event()
    original = runtime._change_report

    async def hold(*args, **kwargs):
        started.set()
        assert await asyncio.to_thread(release.wait, 3)
        return await original(*args, **kwargs)

    monkeypatch.setattr(runtime, "_change_report", hold)
    sid = harness.client.post(
        "/sessions", json={"user_id": "dev", "workspace_root": str(tmp_path)}
    ).json()["session_id"]
    harness.provider.script(make_response(content="Complete"))
    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(
            harness.client.post,
            "/chat",
            json={"user_id": "dev", "session_id": sid, "message": "Explain"},
        )
        try:
            assert started.wait(2)
            result = harness.client.post(f"/sessions/{sid}/cancel", json={"user_id": "dev"})
            assert result.status_code == 409
            assert "finalizing" in result.json()["detail"]
        finally:
            release.set()
        assert active.result(timeout=5).json()["completion_status"] == "completed"


async def test_shutdown_also_cancels_owned_turn_that_has_not_started(tmp_path: Path) -> None:
    import httpx

    from api import server
    from harness.session_store import SessionStore
    from tests.test_turn_admission import HoldingProvider

    provider = HoldingProvider()
    app = make_app(tmp_path, provider)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        sid = (await client.post("/sessions", json={"user_id": "dev"})).json()["session_id"]
        components = app.state.components
        task = asyncio.create_task(
            server._run_configured_turn(
                components=components,
                settings=app.state.settings,
                session=components.sessions[sid],
                user_id="dev",
                message="Explain",
                provider=provider,
            )
        )
        server._own_task(components, task)
        # No await: shutdown starts before the scheduled turn registers its signal.
    assert task.result().completion_status == "cancelled"
    assert provider.calls == [] and not provider.started.is_set()
    store = SessionStore(tmp_path / "sessions.db")
    try:
        assert store.get(sid).responses[0].response.completion_status == "cancelled"
    finally:
        store.close()
