import hashlib

from fastapi.testclient import TestClient

from .conftest import ScriptedProvider
from .test_session_persistence import make_app


def test_owner_note_detail_complete_utf8_text_and_restart(tmp_path):
    provider = ScriptedProvider()
    text = "é" * 2048
    with TestClient(make_app(tmp_path, provider)) as client:

        def seed():
            store = client.app.state.components.fact_store
            store.add("dev", text, source_turn_id="source", workspace_root="/project/a")
            store.add("other", "private-marker")

        client.portal.call(seed)
        before = client.get("/memory", params={"user_id": "dev"}).json()
        note = before["notes"][0]
        assert note["preview_truncated"]
    with TestClient(make_app(tmp_path, provider)) as client:
        response = client.get(f"/memory/{note['id']}", params={"user_id": "dev"})
        assert response.status_code == 200
        detail = response.json()
        assert detail["text"] == text
        assert detail["note"] == note
        assert detail["note"]["fact_sha256"] == hashlib.sha256(text.encode()).hexdigest()
        assert client.get(f"/memory/{note['id']}", params={"user_id": "other"}).status_code == 404
        assert client.get("/memory/999", params={"user_id": "dev"}).status_code == 404
        assert client.get("/memory/0", params={"user_id": "dev"}).status_code == 422
        assert client.get("/memory/1").status_code == 422
        assert client.get("/memory", params={"user_id": "dev"}).json() == before
        assert client.get("/runs", params={"user_id": "dev"}).json() == []
    assert provider.calls == []


def test_legacy_note_detail_overflow_and_lookup_failure_are_private(tmp_path, monkeypatch):
    import sqlite3

    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:

        def seed():
            store = client.app.state.components.fact_store
            with store._conn:
                store._conn.execute(
                    "INSERT INTO memory_facts (id, user_id, fact, created_at) VALUES (1, 'dev', ?, 'old')",
                    ("private-marker" + "é" * 2042,),
                )

        client.portal.call(seed)
        response = client.get("/memory/1", params={"user_id": "dev"})
        assert response.status_code == 413
        assert "private-marker" not in response.text
        assert client.get("/memory/1", params={"user_id": "other"}).status_code == 404
        assert client.get("/memory", params={"user_id": "dev"}).json()["notes"][0][
            "preview_truncated"
        ]

        def fail(*args):
            raise sqlite3.OperationalError("private-marker")

        monkeypatch.setattr(client.app.state.components.fact_store, "get_for_review", fail)
        response = client.get("/memory/1", params={"user_id": "dev"})
        assert response.status_code == 503 and "private-marker" not in response.text
    assert provider.calls == []


def test_note_detail_global_and_ambiguous_unsupported_ids(tmp_path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:
        client.portal.call(lambda: client.app.state.components.fact_store.add("dev", "global note"))
        note = client.get("/memory", params={"user_id": "dev"}).json()["notes"][0]
        detail = client.get(f"/memory/{note['id']}", params={"user_id": "dev"}).json()
        assert detail["text"] == "global note" and detail["note"]["workspace_root"] is None

        def corrupt():
            store = client.app.state.components.fact_store
            with store._conn:
                store._conn.execute(
                    "INSERT INTO workspace_memory_facts (id, user_id, workspace_root, fact, created_at) VALUES (?, 'dev', '/project/a', 'private-marker', 'old')",
                    (note["id"],),
                )

        client.portal.call(corrupt)
        response = client.get(f"/memory/{note['id']}", params={"user_id": "dev"})
        assert response.status_code == 409
        assert "private-marker" not in response.text
    assert provider.calls == []
