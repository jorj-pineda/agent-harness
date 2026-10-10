"""Durable request identity across transports and server lifecycles."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness.runs import RunInput
from harness.session_store import SessionStore
from harness.state import Session
from providers.base import ToolCall

from .conftest import ScriptedProvider, make_response
from .test_session_persistence import make_app
from .test_stream import _parse_sse


class HeldProvider(ScriptedProvider):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    async def chat(self, messages, **kwargs):
        self.started.set()
        while not self.release.is_set():  # noqa: ASYNC110 - controlled from TestClient thread
            await asyncio.sleep(0.001)
        return await super().chat(messages, **kwargs)


def test_async_submission_and_http_sse_retries_share_one_real_edit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    provider = HeldProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="write",
                    name="write_file",
                    arguments={"path": "a.py", "content": "value = 1\n"},
                )
            ]
        ),
        make_response(content="Done"),
    )
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        payload = {
            "user_id": "dev",
            "session_id": sid,
            "message": "Write a.py",
            "request_id": "edit-1",
        }
        submission = client.post("/runs", json=payload)
        assert submission.status_code == 202
        run_id = submission.json()["run_id"]
        assert provider.started.wait(2)
        assert client.post("/runs", json=payload).json()["run_id"] == run_id
        assert (
            client.get(f"/runs/{run_id}", params={"user_id": "dev"}).json()["status"] == "running"
        )
        busy_payload = {**payload, "request_id": "busy"}
        busy = client.post("/chat", json=busy_payload).json()
        assert busy["completion_status"] == "blocked"
        assert client.post("/chat", json=busy_payload).json() == busy
        with ThreadPoolExecutor(max_workers=2) as pool:
            http = pool.submit(client.post, "/chat", json=payload)
            stream = pool.submit(client.get, "/chat/stream", params=payload)
            provider.release.set()
            answer = http.result(timeout=5).json()
            events = _parse_sse(stream.result(timeout=5).text)
        assert answer["run_id"] == run_id
        assert json.loads(events[-1]["data"])["response"] == answer
        assert len(provider.calls) == 2
        assert (repo / "a.py").read_text() == "value = 1\n"
        assert answer["workspace_changes"]["added"] == ["a.py"]
        record = client.get(f"/runs/{run_id}", params={"user_id": "dev"}).json()
        assert record["status"] == "finished"
        assert record["response"] == answer
        assert client.post("/runs", json=payload).status_code == 200
        archive = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        assert len(archive["responses"]) == 1
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted)) as client:
        assert client.post("/chat", json=payload).json() == answer
        assert client.post("/runs", json=payload).json() == record
        events = _parse_sse(client.get("/chat/stream", params=payload).text)
        assert len(events) == 1  # Final result only; tool events are not replayed.
        assert client.get(f"/runs/{run_id}", params={"user_id": "dev"}).json() == record
        assert client.get(f"/runs/{run_id}", params={"user_id": "other"}).status_code == 403
        assert client.get("/runs/missing", params={"user_id": "dev"}).status_code == 404
        assert client.get("/runs", params={"user_id": "other"}).json() == []
        listing = client.get("/runs", params={"user_id": "dev", "session_id": sid}).json()
        assert {item["run_id"] for item in listing} == {run_id, busy["run_id"]}
        assert client.get("/runs", params={"user_id": "dev", "offset": 2}).json() == []
        assert client.get("/runs", params={"user_id": "dev", "limit": 101}).status_code == 422
        assert restarted.calls == []


@pytest.mark.parametrize(
    "change", [{"message": "Different"}, {"session_id": "other"}, {"provider": "missing"}]
)
def test_request_id_conflicts_before_resolving_new_input(tmp_path: Path, change: dict) -> None:
    provider = ScriptedProvider()
    provider.script(make_response(content="Done"))
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        payload = {"user_id": "dev", "session_id": sid, "message": "Explain", "request_id": "one"}
        assert client.post("/chat", json=payload).status_code == 200
        for endpoint in ("/chat", "/runs"):
            assert client.post(endpoint, json={**payload, **change}).status_code == 409
        assert client.get("/chat/stream", params={**payload, **change}).status_code == 409
        assert len(provider.calls) == 1


def test_failed_run_retry_does_not_dispatch_again(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider), raise_server_exceptions=False) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        payload = {
            "user_id": "dev",
            "session_id": sid,
            "message": "Explain",
            "request_id": "failure",
        }
        assert client.post("/chat", json=payload).status_code == 500
        record = client.post("/runs", json=payload).json()
        assert record["status"] == "failed"
        assert record["response"] is None
        assert client.post("/chat", json=payload).status_code == 409
        assert len(provider.calls) == 1


def test_startup_marks_abandoned_run_interrupted_without_execution(tmp_path: Path) -> None:
    session = Session(user_id="dev")
    payload = {
        "user_id": "dev",
        "session_id": session.session_id,
        "message": "Fix it",
        "request_id": "lost",
    }
    store = SessionStore(tmp_path / "sessions.db")
    store.save(session)
    run, _ = store.create_run(RunInput(**payload), provider="scripted", configured_model=None)
    store.close()
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:
        record = client.get(f"/runs/{run.run_id}", params={"user_id": "dev"}).json()
        assert record["status"] == "interrupted"
        assert record["finished_at"] is not None
        assert "workspace edits may exist" in record["error"]
        assert client.post("/runs", json=payload).json() == record
        assert client.post("/chat", json=payload).status_code == 409
        assert provider.calls == []


@pytest.mark.parametrize("request_id", ["", "has spaces", "x" * 129])
def test_invalid_request_identity_is_rejected(tmp_path: Path, request_id: str) -> None:
    with TestClient(make_app(tmp_path, ScriptedProvider())) as client:
        payload = {
            "user_id": "dev",
            "session_id": "missing",
            "message": "Fix",
            "request_id": request_id,
        }
        assert client.post("/runs", json=payload).status_code == 422
        assert client.post("/chat", json=payload).status_code == 422
        assert client.get("/chat/stream", params=payload).status_code == 422


def test_legacy_requests_remain_distinct_and_run_key_is_user_scoped(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    provider.script(*(make_response(content="Done") for _ in range(4)))
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        payload = {"user_id": "dev", "session_id": sid, "message": "Explain"}
        first = client.post("/chat", json=payload).json()
        second = client.post("/chat", json=payload).json()
        assert first["run_id"] != second["run_id"]
        assert client.post("/runs", json=payload).status_code == 422
        third = client.post("/chat", json={**payload, "request_id": "same"}).json()
        other = client.post("/sessions", json={"user_id": "other"}).json()["session_id"]
        fourth = client.post(
            "/chat", json={**payload, "user_id": "other", "session_id": other, "request_id": "same"}
        ).json()
        assert third["run_id"] != fourth["run_id"]
        assert len(provider.calls) == 4


def test_run_storage_failure_prevents_provider_dispatch(tmp_path: Path, monkeypatch) -> None:
    import sqlite3

    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]

        def fail(*args, **kwargs):
            raise sqlite3.OperationalError("disk full")

        monkeypatch.setattr(SessionStore, "create_run", fail)
        payload = {"user_id": "dev", "session_id": sid, "message": "Fix", "request_id": "one"}
        assert client.post("/chat", json=payload).status_code == 503
        assert client.post("/runs", json=payload).status_code == 503
        assert client.get("/chat/stream", params=payload).status_code == 503
        assert provider.calls == []


def test_cancelled_run_is_replayable_without_reusing_session(tmp_path: Path) -> None:
    provider = HeldProvider()
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        payload = {
            "user_id": "dev",
            "session_id": sid,
            "message": "Explain",
            "request_id": "cancel",
        }
        run = client.post("/runs", json=payload).json()
        assert provider.started.wait(2)
        assert client.post(f"/sessions/{sid}/cancel", json={"user_id": "dev"}).status_code == 202
        response = client.post("/chat", json=payload).json()
        assert response["completion_status"] == "cancelled"
        saved = client.get(f"/runs/{run['run_id']}", params={"user_id": "dev"}).json()
        assert saved["status"] == "cancelled"
        assert saved["response"] == response
        assert client.post("/chat", json=payload).json() == response
        assert client.post("/chat", json={**payload, "request_id": "new"}).status_code == 409
    with TestClient(make_app(tmp_path, ScriptedProvider())) as client:
        assert client.post("/chat", json=payload).json() == response
