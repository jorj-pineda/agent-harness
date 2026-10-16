import hashlib
import sqlite3

import pytest
from fastapi.testclient import TestClient

from memory import FactStore
from memory.store import FactConflictError, MemoryScopePrecondition

from .conftest import ScriptedProvider
from .test_session_persistence import make_app


@pytest.mark.parametrize("method", ["DELETE", "PUT"])
def test_explicit_scope_guards_and_legacy_omission(method, tmp_path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:

        def seed():
            store = client.app.state.components.fact_store
            store.add("dev", "global")
            store.add("dev", "workspace", workspace_root="/a")

        client.portal.call(seed)
        notes = client.get("/memory", params={"user_id": "dev"}).json()["notes"]
        scoped, global_note = notes

        def mutate(note, **extra):
            return client.request(
                method,
                f"/memory/{note['id']}",
                json={
                    "user_id": "dev",
                    "expected_fact_sha256": note["fact_sha256"],
                    **({"replacement": "corrected"} if method == "PUT" else {}),
                    **extra,
                },
            )

        for note, wrong in [(scoped, None), (scoped, "/b"), (global_note, "/a")]:
            assert mutate(note, expected_workspace_root=wrong).status_code == 409
        assert mutate(scoped, expected_workspace_root="relative").status_code == 422
        assert client.get("/memory", params={"user_id": "dev"}).json()["notes"] == notes
        assert mutate(global_note, expected_workspace_root=None).status_code == 200
        moved = client.put(
            f"/memory/{scoped['id']}/scope",
            json={
                "user_id": "dev",
                "expected_fact_sha256": scoped["fact_sha256"],
                "expected_workspace_root": "/a",
                "workspace_root": "/b",
            },
        )
        assert moved.status_code == 200
        assert mutate(scoped, expected_workspace_root="/a").status_code == 409
        assert mutate(scoped).status_code == 200  # Omission retains original text-only contract.
        assert client.get("/runs", params={"user_id": "dev"}).json() == []
    assert provider.calls == []


@pytest.mark.parametrize("action", ["delete", "correct"])
def test_final_sql_scope_guard_rejects_move_after_hash_read(action, tmp_path, monkeypatch):
    with FactStore(tmp_path / "memory.db") as store:
        store.add("dev", "unchanged text", workspace_root="/a")
        original = store.list("dev")[0]
        digest = hashlib.sha256(original.fact.encode()).hexdigest()
        sha256 = hashlib.sha256

        def move_after_read(raw):
            with sqlite3.connect(store.path) as other:
                other.execute(
                    "UPDATE workspace_memory_facts SET workspace_root='/b' WHERE id=?",
                    (original.id,),
                )
            return sha256(raw)

        monkeypatch.setattr("memory.store.hashlib.sha256", move_after_read)
        with pytest.raises(FactConflictError, match="changed during"):
            if action == "delete":
                store.delete(
                    "dev", original.id, digest, expected_scope=MemoryScopePrecondition("/a")
                )
            else:
                store.correct(
                    "dev",
                    original.id,
                    digest,
                    "replacement",
                    expected_scope=MemoryScopePrecondition("/a"),
                )
        assert store.list("dev") == [original.model_copy(update={"workspace_root": "/b"})]


@pytest.mark.parametrize("action", ["delete", "correct"])
def test_matching_workspace_scope_succeeds(action, tmp_path):
    with FactStore(tmp_path / "memory.db") as store:
        store.add("dev", "note", workspace_root="/a")
        note = store.list("dev")[0]
        digest = hashlib.sha256(b"note").hexdigest()
        if action == "delete":
            assert store.delete(
                "dev", note.id, digest, expected_scope=MemoryScopePrecondition("/a")
            )
            assert store.list("dev") == []
        else:
            corrected = store.correct(
                "dev", note.id, digest, "new", expected_scope=MemoryScopePrecondition("/a")
            )
            assert corrected.workspace_root == "/a" and corrected.fact == "new"
