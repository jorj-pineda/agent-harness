import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from harness.checkpoints import capture_checkpoint

from .conftest import ScriptedProvider
from .test_session_persistence import make_app


def test_checkpoint_metadata_survives_restart_without_restoring_history(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("pre-existing user edit\n")
    (root / ".env").write_text("fixture-private-marker")
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(root)}).json()[
            "session_id"
        ]
        saved = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        endpoint = f"/sessions/{sid}/checkpoints"
        result = client.post(endpoint, json={"user_id": "dev"})
        assert result.status_code == 201
        record = result.json()
        assert record["file_count"] == 1
        assert record["restore_supported"] is False
        assert "pre-existing" not in result.text and "fixture-private-marker" not in result.text
        assert ".env" not in record["files"]
        detail = endpoint + "/" + record["checkpoint_id"]
        assert client.get(detail, params={"user_id": "other"}).status_code == 403
        assert client.get(endpoint, params={"user_id": "dev", "limit": 0}).status_code == 422
        assert client.post(endpoint, json={"user_id": "other"}).status_code == 403
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
    (root / "a.py").write_text("later external edit\n")
    with TestClient(make_app(tmp_path, provider)) as client:
        assert client.get(detail, params={"user_id": "dev"}).json() == record
        listed = client.get(endpoint, params={"user_id": "dev"}).json()
        assert listed[0]["checkpoint_id"] == record["checkpoint_id"] and "files" not in listed[0]
        other = client.post(
            "/sessions", json={"user_id": "dev", "workspace_root": str(root)}
        ).json()["session_id"]
        assert (
            client.get(
                f"/sessions/{other}/checkpoints/{record['checkpoint_id']}",
                params={"user_id": "dev"},
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Continue"}
            ).status_code
            == 409
        )
        assert (root / "a.py").read_text() == "later external edit\n"
        assert client.get("/runs", params={"user_id": "dev"}).json() == []
    assert provider.calls == []


def test_capture_holds_admission_and_storage_failure_leaves_no_checkpoint(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a").write_bytes(b"user content")
    started, release = threading.Event(), threading.Event()

    def held_capture(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return capture_checkpoint(*args, **kwargs)

    monkeypatch.setattr("api.server.capture_checkpoint", held_capture)
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(root)}).json()[
            "session_id"
        ]
        endpoint = f"/sessions/{sid}/checkpoints"
        with ThreadPoolExecutor() as pool:
            pending = pool.submit(client.post, endpoint, json={"user_id": "dev"})
            try:
                assert started.wait(5)
                assert client.post(endpoint, json={"user_id": "dev"}).status_code == 409
                blocked = client.post(
                    "/chat", json={"user_id": "dev", "session_id": sid, "message": "Read a"}
                )
                assert blocked.json()["completion_status"] == "blocked"
                assert provider.calls == []
            finally:
                release.set()
            assert pending.result().status_code == 201
        with sqlite3.connect(tmp_path / "sessions.db") as conn:
            conn.execute(
                "CREATE TRIGGER deny_capture BEFORE INSERT ON checkpoints "
                "BEGIN SELECT RAISE(ABORT, 'fixture-private-marker'); END"
            )
        failed = client.post(endpoint, json={"user_id": "dev"})
        assert failed.status_code == 503
        assert "fixture-private-marker" not in failed.text
        assert len(client.get(endpoint, params={"user_id": "dev"}).json()) == 1
        assert (root / "a").read_bytes() == b"user content"


async def test_disconnected_capture_remains_owned_until_read_and_save_settle(tmp_path, monkeypatch):
    import asyncio

    import httpx

    root = tmp_path / "repo"
    root.mkdir()
    (root / "a").write_bytes(b"user content")
    started, release = threading.Event(), threading.Event()

    def held_capture(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return capture_checkpoint(*args, **kwargs)

    monkeypatch.setattr("api.server.capture_checkpoint", held_capture)
    provider = ScriptedProvider()
    app = make_app(tmp_path, provider)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        sid = (
            await client.post("/sessions", json={"user_id": "dev", "workspace_root": str(root)})
        ).json()["session_id"]
        endpoint = f"/sessions/{sid}/checkpoints"
        pending = asyncio.create_task(client.post(endpoint, json={"user_id": "dev"}))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert (await client.post(endpoint, json={"user_id": "dev"})).status_code == 409
        finally:
            release.set()
        await asyncio.gather(*tuple(app.state.components.tasks))
        assert len((await client.get(endpoint, params={"user_id": "dev"})).json()) == 1
    assert provider.calls == []
