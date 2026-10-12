import sqlite3

import pytest
from fastapi.testclient import TestClient

from providers.base import ToolCall

from .conftest import ScriptedProvider, make_response
from .test_session_persistence import make_app


@pytest.mark.parametrize("transport", ["http", "stream"])
def test_opt_in_turn_checkpoints_bind_before_after_and_replay_after_restart(tmp_path, transport):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("user = 1\n")
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "a.py", "content": "user = 1\nagent = 2\n"},
                )
            ]
        ),
        make_response(content="Done"),
    )
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post(
            "/sessions",
            json={"user_id": "dev", "workspace_root": str(root), "checkpoint_turns": True},
        ).json()["session_id"]
        request = {
            "user_id": "dev",
            "session_id": sid,
            "message": "Update a.py",
            "request_id": "once",
        }
        if transport == "http":
            assert client.post("/chat", json=request).status_code == 200
        else:
            assert "turn_done" in client.get("/chat/stream", params=request).text
        run = client.get("/runs", params={"user_id": "dev"}).json()[0]
        before, after = run["before_checkpoint_id"], run["after_checkpoint_id"]
        assert before and after and before != after
        endpoint = f"/sessions/{sid}/checkpoints"
        records = client.get(endpoint, params={"user_id": "dev"}).json()
        assert len(records) == 2
        b = client.get(endpoint + "/" + before, params={"user_id": "dev"}).json()
        a = client.get(endpoint + "/" + after, params={"user_id": "dev"}).json()
        assert b["kind"] == "turn_before" and a["kind"] == "turn_after"
        assert a["paired_before_id"] == before
        assert a["run_id"] == b["run_id"] == run["run_id"]
        assert a["files"]["a.py"]["sha256"] != b["files"]["a.py"]["sha256"]
        assert a["restore_supported"] is False
    with sqlite3.connect(tmp_path / "sessions.db") as db:
        assert (
            db.execute(
                "SELECT content FROM checkpoint_files WHERE checkpoint_id=?", (before,)
            ).fetchone()[0]
            == b"user = 1\n"
        )
        assert (
            db.execute(
                "SELECT content FROM checkpoint_files WHERE checkpoint_id=?", (after,)
            ).fetchone()[0]
            == b"user = 1\nagent = 2\n"
        )
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted)) as client:
        replay = client.post("/chat", json=request)
        assert replay.status_code == 200 and replay.json()["run_id"] == run["run_id"]
        assert len(client.get(endpoint, params={"user_id": "dev"}).json()) == 2
        assert client.get("/runs/" + run["run_id"], params={"user_id": "dev"}).json() == run
    assert restarted.calls == []
    assert (root / "a.py").read_text() == "user = 1\nagent = 2\n"


@pytest.mark.parametrize("phase", ["before", "after"])
def test_checkpoint_failure_stops_execution_and_preserves_evidence(tmp_path, monkeypatch, phase):
    import api.server

    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("user = 1\n")
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "a.py", "content": "user = 1\nagent = 2\n"},
                )
            ]
        ),
        make_response(content="Done"),
    )
    real_capture = api.server.capture_checkpoint
    captures = 0

    def fail_capture(*args, **kwargs):
        nonlocal captures
        captures += 1
        if captures == (1 if phase == "before" else 2):
            raise ValueError("fixture-private-marker")
        return real_capture(*args, **kwargs)

    monkeypatch.setattr("api.server.capture_checkpoint", fail_capture)
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post(
            "/sessions",
            json={"user_id": "dev", "workspace_root": str(root), "checkpoint_turns": True},
        ).json()["session_id"]
        saved = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        result = client.post(
            "/chat", json={"user_id": "dev", "session_id": sid, "message": "Update a.py"}
        )
        assert result.status_code == 503 and "fixture-private-marker" not in result.text
        run = client.get("/runs", params={"user_id": "dev"}).json()[0]
        assert run["status"] == "failed" and run["response"] is None
        assert run["after_checkpoint_id"] is None
        assert bool(run["before_checkpoint_id"]) == (phase == "after")
        assert len(provider.calls) == (0 if phase == "before" else 2)
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
        assert (
            client.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Continue"}
            ).status_code
            == 409
        )
    assert (root / "a.py").read_text() == (
        "user = 1\n" if phase == "before" else "user = 1\nagent = 2\n"
    )


def test_default_turns_do_not_create_checkpoints(tmp_path):
    provider = ScriptedProvider()
    provider.script(make_response(content="Hello"))
    with TestClient(make_app(tmp_path, provider)) as client:
        assert (
            client.post("/sessions", json={"user_id": "dev", "checkpoint_turns": True}).status_code
            == 400
        )
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        assert (
            client.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Inspect code"}
            ).status_code
            == 200
        )
        run = client.get("/runs", params={"user_id": "dev"}).json()[0]
        assert run["before_checkpoint_id"] is run["after_checkpoint_id"] is None
        assert client.get(f"/sessions/{sid}/checkpoints", params={"user_id": "dev"}).json() == []


def test_cancel_during_before_capture_settles_then_archives_without_inference(
    tmp_path, monkeypatch
):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import api.server

    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("user = 1\n")
    started, release = threading.Event(), threading.Event()
    original_capture = api.server.capture_checkpoint

    def held_capture(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return original_capture(*args, **kwargs)

    monkeypatch.setattr("api.server.capture_checkpoint", held_capture)
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post(
            "/sessions",
            json={"user_id": "dev", "workspace_root": str(root), "checkpoint_turns": True},
        ).json()["session_id"]
        with ThreadPoolExecutor() as pool:
            pending = pool.submit(
                client.post,
                "/chat",
                json={"user_id": "dev", "session_id": sid, "message": "Inspect a.py"},
            )
            try:
                assert started.wait(5)
                assert (
                    client.post(f"/sessions/{sid}/cancel", json={"user_id": "dev"}).status_code
                    == 202
                )
            finally:
                release.set()
            result = pending.result()
        assert result.status_code == 200
        assert result.json()["completion_status"] == "cancelled"
        run = client.get("/runs", params={"user_id": "dev"}).json()[0]
        assert run["status"] == "cancelled"
        assert run["before_checkpoint_id"] and run["after_checkpoint_id"]
        assert (root / "a.py").read_text() == "user = 1\n"
    assert provider.calls == []
