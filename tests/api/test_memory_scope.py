import hashlib
import sqlite3

from fastapi.testclient import TestClient

from memory import FactStore
from memory.store import FactConflictError

from .conftest import ScriptedProvider
from .test_session_persistence import make_app


def move(client, note, target, *, source=None, owner="dev", digest=None):
    return client.put(
        f"/memory/{note['id']}/scope",
        json={
            "user_id": owner,
            "expected_fact_sha256": digest or note["fact_sha256"],
            "expected_workspace_root": source,
            "workspace_root": target,
        },
    )


def test_scope_moves_survive_restart_preserve_metadata_and_update_future_injection(tmp_path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:
        client.portal.call(
            lambda: client.app.state.components.fact_store.add(
                "dev", "repository convention", source_turn_id="original"
            )
        )
        original = client.get("/memory", params={"user_id": "dev"}).json()["notes"][0]
        assert move(client, original, "/project/a", owner="other").status_code == 404
        assert move(client, original, "/project/a", digest="0" * 64).status_code == 409
        moved = move(client, original, "/project/a").json()["note"]
        assert moved == {**original, "workspace_root": "/project/a"}
        assert move(client, original, "/project/b").status_code == 409

        def prompts():
            store = client.app.state.components.fact_store
            return [
                store.format_for_system_prompt("dev", workspace_root=root)
                for root in [None, "/project/a", "/project/b"]
            ]

        assert client.portal.call(prompts) == [
            "",
            "Engineering memory for this developer (persists across sessions):\n- repository convention",
            "",
        ]
        assert client.get("/runs", params={"user_id": "dev"}).json() == []
    with TestClient(make_app(tmp_path, provider)) as client:
        note = client.get("/memory", params={"user_id": "dev"}).json()["notes"][0]
        assert note == moved
        assert move(client, note, "/project/a", source="/project/a").json()["note"] == moved
        b = move(client, note, "/project/b", source="/project/a").json()["note"]
        assert b["id"] == original["id"] and b["workspace_root"] == "/project/b"
        assert move(client, b, None, source="/project/b").json()["note"] == original
    assert provider.calls == []


def test_invalid_scope_requests_conflicts_and_failures_preserve_rows(tmp_path, monkeypatch):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:

        def seed():
            store = client.app.state.components.fact_store
            store.add("dev", "same")
            store.add("dev", "same", workspace_root="/a")

        client.portal.call(seed)
        before = client.get("/memory", params={"user_id": "dev"}).json()
        note = before["notes"][-1]
        assert move(client, note, "/a").status_code == 409
        for target in ["relative", "/a/../b", "/a/"]:
            assert move(client, note, target).status_code == 422
        payload = {"user_id": "dev", "expected_fact_sha256": note["fact_sha256"]}
        assert client.put(f"/memory/{note['id']}/scope", json=payload).status_code == 422
        assert move(client, note, None, source="relative").status_code == 422

        def reject(*args):
            raise sqlite3.OperationalError("private-marker")

        monkeypatch.setattr(client.app.state.components.fact_store, "reclassify", reject)
        failed = move(client, note, "/b")
        assert failed.status_code == 503 and "private-marker" not in failed.text
        assert client.get("/memory", params={"user_id": "dev"}).json() == before
    assert provider.calls == []


def test_scope_transaction_rolls_back_and_competing_connection_rejects_stale_review(tmp_path):
    path = tmp_path / "notes.db"
    with FactStore(path) as store, FactStore(path) as other:
        store.add("dev", "note", source_turn_id="source")
        note = store.list("dev")[0]
        digest = hashlib.sha256(note.fact.encode()).hexdigest()
        with store._conn:
            store._conn.execute(
                "CREATE TRIGGER reject_move BEFORE INSERT ON workspace_memory_facts BEGIN SELECT RAISE(ABORT, 'private-marker'); END"
            )
        import pytest

        with pytest.raises(sqlite3.IntegrityError):
            store.reclassify("dev", note.id, digest, None, "/a")
        assert store.list("dev") == [note]
        with store._conn:
            store._conn.execute("DROP TRIGGER reject_move")
        moved = other.reclassify("dev", note.id, digest, None, "/a")
        with pytest.raises(FactConflictError):
            store.reclassify("dev", note.id, digest, None, "/b")
        assert store.list("dev") == [moved]
        store.add("dev", "next")
        assert store.list("dev")[0].id == note.id + 1
