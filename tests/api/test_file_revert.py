import sqlite3

import pytest
from fastapi.testclient import TestClient

from providers.base import ToolCall

from .conftest import ScriptedProvider, make_response
from .test_session_persistence import make_app


def prepared(tmp_path, client, provider):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("user = 1\n")
    (root / "other.py").write_text("unrelated user work\n")
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
    sid = client.post(
        "/sessions", json={"user_id": "dev", "workspace_root": str(root), "checkpoint_turns": True}
    ).json()["session_id"]
    assert (
        client.post(
            "/chat", json={"user_id": "dev", "session_id": sid, "message": "Update a.py"}
        ).status_code
        == 200
    )
    run = client.get("/runs", params={"user_id": "dev"}).json()[0]
    return root, sid, run


def test_file_revert_preserves_other_work_retires_session_and_never_replays(tmp_path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:
        root, sid, run = prepared(tmp_path, client, provider)
        (root / "other.py").write_text("later unrelated user edit\n")
        saved = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        body = {"user_id": "dev", "path": "a.py", "request_id": "undo-once"}
        endpoint = f"/runs/{run['run_id']}/revert-file"
        result = client.post(endpoint, json=body)
        assert result.status_code == 200, result.text
        record = result.json()
        assert record["status"] == "applied"
        assert (root / "a.py").read_text() == "user = 1\n"
        assert (root / "other.py").read_text() == "later unrelated user edit\n"
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
        assert client.get("/runs/" + run["run_id"], params={"user_id": "dev"}).json() == run
        assert (
            client.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Continue"}
            ).status_code
            == 409
        )
        assert client.post(endpoint, json={**body, "path": "other.py"}).status_code == 409
        assert (
            client.get(
                "/file-reverts/" + record["revert_id"], params={"user_id": "other"}
            ).status_code
            == 403
        )
    (root / "a.py").write_text("later target user edit\n")
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted)) as client:
        assert client.post(endpoint, json=body).json() == record
        assert (root / "a.py").read_text() == "later target user edit\n"
        assert (
            client.get("/file-reverts", params={"user_id": "dev", "request_id": "undo-once"}).json()
            == record
        )
    assert restarted.calls == []


def test_journal_storage_failures_never_invent_success_or_repeat_edits(tmp_path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:
        root, _sid, run = prepared(tmp_path, client, provider)
        endpoint = f"/runs/{run['run_id']}/revert-file"
        body = {"user_id": "dev", "path": "a.py", "request_id": "undo-once"}
        with sqlite3.connect(tmp_path / "sessions.db") as db:
            db.execute(
                "CREATE TRIGGER deny_pending BEFORE INSERT ON file_reverts "
                "BEGIN SELECT RAISE(ABORT, 'private-marker'); END"
            )
        failed = client.post(endpoint, json=body)
        assert failed.status_code == 503 and "private-marker" not in failed.text
        assert (root / "a.py").read_text() == "user = 1\nagent = 2\n"
        assert (
            client.get("/file-reverts", params={"user_id": "dev", "request_id": "undo-once"}).json()
            is None
        )
        with sqlite3.connect(tmp_path / "sessions.db") as db:
            db.execute("DROP TRIGGER deny_pending")
            db.execute(
                "CREATE TRIGGER deny_outcome BEFORE UPDATE ON file_reverts "
                "BEGIN SELECT RAISE(ABORT, 'private-marker'); END"
            )
        failed = client.post(endpoint, json=body)
        assert failed.status_code == 503 and "private-marker" not in failed.text
        assert (root / "a.py").read_text() == "user = 1\n"
        pending = client.get(
            "/file-reverts", params={"user_id": "dev", "request_id": "undo-once"}
        ).json()
        assert pending["status"] == "pending"
        assert client.post(endpoint, json=body).status_code == 202
        with sqlite3.connect(tmp_path / "sessions.db") as db:
            db.execute("DROP TRIGGER deny_outcome")
    (root / "a.py").write_text("new user work\n")
    with TestClient(make_app(tmp_path, ScriptedProvider())) as client:
        recovered = client.get(
            "/file-reverts/" + pending["revert_id"], params={"user_id": "dev"}
        ).json()
        assert recovered["status"] == "interrupted"
        assert client.post(endpoint, json=body).status_code == 409
        assert (root / "a.py").read_text() == "new user work\n"


def test_later_edits_and_corrupt_saved_bytes_fail_without_mutating(tmp_path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:
        root, _sid, run = prepared(tmp_path, client, provider)
        endpoint = f"/runs/{run['run_id']}/revert-file"
        body = {"user_id": "dev", "path": "a.py", "request_id": "later-edit"}
        (root / "a.py").write_text("later user edit\n")
        rejected = client.post(endpoint, json=body)
        assert rejected.status_code == 409
        assert rejected.json()["status"] == "rejected"
        assert (root / "a.py").read_text() == "later user edit\n"
        assert (
            client.post(
                endpoint, json={**body, "path": "../escape", "request_id": "bad-path"}
            ).status_code
            == 422
        )
        with sqlite3.connect(tmp_path / "sessions.db") as db:
            db.execute(
                "UPDATE checkpoint_files SET content=? WHERE checkpoint_id=?",
                (b"corrupt", run["before_checkpoint_id"]),
            )
        failed = client.post(endpoint, json={**body, "request_id": "corrupt-bytes"})
        assert failed.status_code == 503
        assert (
            client.get(
                "/file-reverts", params={"user_id": "dev", "request_id": "corrupt-bytes"}
            ).json()
            is None
        )
        assert (root / "a.py").read_text() == "later user edit\n"


async def test_disconnected_revert_stays_owned_and_keeps_admission_until_settled(
    tmp_path, monkeypatch
):
    import asyncio
    import threading

    import httpx

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
    started, release = threading.Event(), threading.Event()
    real_revert = api.server.revert_file

    def held_revert(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return real_revert(*args, **kwargs)

    monkeypatch.setattr("api.server.revert_file", held_revert)
    app = make_app(tmp_path, provider)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        sid = (
            await client.post(
                "/sessions",
                json={"user_id": "dev", "workspace_root": str(root), "checkpoint_turns": True},
            )
        ).json()["session_id"]
        await client.post(
            "/chat", json={"user_id": "dev", "session_id": sid, "message": "Update a.py"}
        )
        run = (await client.get("/runs", params={"user_id": "dev"})).json()[0]
        endpoint = f"/runs/{run['run_id']}/revert-file"
        body = {"user_id": "dev", "path": "a.py", "request_id": "owned-revert"}
        pending = asyncio.create_task(client.post(endpoint, json=body))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            replay = await client.post(endpoint, json=body)
            assert replay.status_code == 202 and replay.json()["status"] == "pending"
            assert (
                await client.post(endpoint, json={**body, "request_id": "another-revert"})
            ).status_code == 409
        finally:
            release.set()
        await asyncio.gather(*tuple(app.state.components.tasks))
        record = (
            await client.get(
                "/file-reverts", params={"user_id": "dev", "request_id": "owned-revert"}
            )
        ).json()
        assert record["status"] == "applied"
        assert (root / "a.py").read_text() == "user = 1\n"
    assert len(provider.calls) == 2
