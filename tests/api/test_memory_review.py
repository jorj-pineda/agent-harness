import hashlib

from fastapi.testclient import TestClient

from providers.base import ToolCall

from .conftest import ScriptedProvider, make_response
from .test_session_persistence import make_app


def delete(client, note, *, user_id="dev", digest=None):
    return client.request(
        "DELETE",
        f"/memory/{note['id']}",
        json={"user_id": user_id, "expected_fact_sha256": digest or note["fact_sha256"]},
    )


def test_memory_paging_preview_ownership_and_restart(tmp_path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:

        def seed():
            store = client.app.state.components.fact_store
            for text in ["oldest", "middle", "€" * 500]:
                store.add("dev", text)
            store.add("other", "private other user")

        client.portal.call(seed)
        page = client.get("/memory", params={"user_id": "dev", "limit": 2}).json()
        note = page["notes"][0]
        assert note["preview"] == "€" * 341
        assert len(note["preview"].encode()) <= 1024
        assert note["preview_truncated"]
        assert note["fact_bytes"] == 1500
        assert note["fact_sha256"] == hashlib.sha256(("€" * 500).encode()).hexdigest()
        assert note["source_turn_id"] is None
        assert "private other user" not in str(page)
        next_page = client.get(
            "/memory",
            params={
                "user_id": "dev",
                "before_id": page["next_before_id"],
                "limit": 2,
            },
        ).json()
        assert [n["preview"] for n in next_page["notes"]] == ["oldest"]
        assert next_page["next_before_id"] is None
        assert delete(client, note, user_id="other").status_code == 404
        assert delete(client, note, digest="0" * 64).status_code == 409
        assert delete(client, note).status_code == 200
        assert delete(client, note).status_code == 404
        assert client.get("/runs", params={"user_id": "dev"}).json() == []
    with TestClient(make_app(tmp_path, provider)) as client:
        assert [
            n["preview"] for n in client.get("/memory", params={"user_id": "dev"}).json()["notes"]
        ] == ["middle", "oldest"]
        assert (
            client.get("/memory", params={"user_id": "other"}).json()["notes"][0]["preview"]
            == "private other user"
        )
    assert provider.calls == []


def test_deletion_changes_future_prompt_without_erasing_saved_evidence(tmp_path):
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="note", name="remember_fact", arguments={"fact": "Incorrect convention"}
                )
            ]
        ),
        make_response(content="Stored note"),
        make_response(content="New answer"),
    )
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        assert (
            client.post(
                "/chat", json={"session_id": sid, "user_id": "dev", "message": "Remember a note"}
            ).status_code
            == 200
        )
        archived = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        note = client.get("/memory", params={"user_id": "dev"}).json()["notes"][0]
        assert delete(client, note).status_code == 200
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archived
        assert "Incorrect convention" in str(archived)
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        assert (
            client.post(
                "/chat", json={"session_id": sid, "user_id": "dev", "message": "Explain"}
            ).status_code
            == 200
        )
        assert "Incorrect convention" not in provider.calls[-1][0][0].content
        assert client.get("/memory", params={"user_id": "dev"}).json()["notes"] == []


def test_invalid_memory_requests_and_delete_failure_preserve_note(tmp_path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:

        def seed():
            store = client.app.state.components.fact_store
            store.add("dev", "keep")
            store._conn.execute(
                "CREATE TRIGGER reject_delete BEFORE DELETE ON memory_facts BEGIN SELECT RAISE(ABORT, 'private-marker'); END"
            )

        client.portal.call(seed)
        note = client.get("/memory", params={"user_id": "dev"}).json()["notes"][0]
        for params in [
            {"user_id": ""},
            {"user_id": "dev", "limit": 51},
            {"user_id": "dev", "before_id": 0},
        ]:
            assert client.get("/memory", params=params).status_code == 422
        assert delete(client, note, digest="bad").status_code == 422
        failed = delete(client, note)
        assert failed.status_code == 503
        assert "private-marker" not in failed.text
        assert client.get("/memory", params={"user_id": "dev"}).json()["notes"] == [note]
    assert provider.calls == []


def test_legacy_oversized_note_has_bounded_preview_and_can_be_deleted(tmp_path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:

        def seed():
            store = client.app.state.components.fact_store
            with store._conn:
                store._conn.execute(
                    "INSERT INTO memory_facts (user_id, fact, created_at) VALUES (?, ?, ?)",
                    ("dev", "legacy" * 10000, "2026-10-14"),
                )

        client.portal.call(seed)
        page = client.get("/memory", params={"user_id": "dev"}).json()
        note = page["notes"][0]
        assert len(note["preview"].encode()) == 1024
        assert note["preview_truncated"] and note["fact_bytes"] == 60000
        assert delete(client, note).status_code == 200
        assert client.get("/memory", params={"user_id": "dev"}).json()["notes"] == []
    assert provider.calls == []


def correct(client, note, text, *, user_id="dev", digest=None):
    return client.put(
        f"/memory/{note['id']}",
        json={
            "user_id": user_id,
            "expected_fact_sha256": digest or note["fact_sha256"],
            "replacement": text,
        },
    )


def test_owner_correction_and_restart_preserve_old_run_evidence(tmp_path):
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(id="note", name="remember_fact", arguments={"fact": "Wrong convention"})
            ]
        ),
        make_response(content="Stored"),
        make_response(content="Reviewed"),
    )
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        client.post("/chat", json={"user_id": "dev", "session_id": sid, "message": "Remember"})
        saved = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        note = client.get("/memory", params={"user_id": "dev"}).json()["notes"][0]
        assert correct(client, note, "Right convention", user_id="other").status_code == 404
        assert correct(client, note, "Right convention", digest="0" * 64).status_code == 409
        for text in [" ", "x" * 4097, "é" * 2049, "x" * 8193]:
            assert correct(client, note, text).status_code == 422
        response = correct(client, note, "  Right convention  ")
        assert response.status_code == 200, response.text
        changed = response.json()["note"]
        assert changed["id"] == note["id"] and changed["created_at"] == note["created_at"]
        assert changed["updated_at"] is not None and changed["source_turn_id"] is None
        assert changed["preview"] == "Right convention"
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
        assert correct(client, note, "Stale replacement").status_code == 409
    with TestClient(make_app(tmp_path, provider)) as client:
        assert client.get("/memory", params={"user_id": "dev"}).json()["notes"] == [changed]
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        client.post("/chat", json={"user_id": "dev", "session_id": sid, "message": "Explain"})
        prompt = provider.calls[-1][0][0].content
        assert "Right convention" in prompt and "Wrong convention" not in prompt


def test_memory_correction_conflict_and_storage_failure_are_private_and_atomic(tmp_path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:

        def seed():
            store = client.app.state.components.fact_store
            store.add("dev", "original", source_turn_id="old-turn")
            store.add("dev", "existing")

        client.portal.call(seed)
        note = client.get("/memory", params={"user_id": "dev"}).json()["notes"][-1]
        assert correct(client, note, "existing").status_code == 409

        def fail():
            client.app.state.components.fact_store._conn.execute(
                "CREATE TRIGGER reject_correction BEFORE UPDATE ON memory_facts BEGIN SELECT RAISE(ABORT, 'private-marker'); END"
            )

        client.portal.call(fail)
        failed = correct(client, note, "replacement")
        assert failed.status_code == 409 and "private-marker" not in failed.text
        assert client.get("/memory", params={"user_id": "dev"}).json()["notes"][-1] == note
    assert provider.calls == []


def test_api_remember_source_turn_survives_restart_and_deduplication(tmp_path):
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(id="first", name="remember_fact", arguments={"fact": "convention"})
            ]
        ),
        make_response(content="Stored"),
        make_response(
            tool_calls=[
                ToolCall(id="duplicate", name="remember_fact", arguments={"fact": "convention"})
            ]
        ),
        make_response(content="Already known"),
    )
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        for message in ["Remember", "Remember again"]:
            assert (
                client.post(
                    "/chat", json={"user_id": "dev", "session_id": sid, "message": message}
                ).status_code
                == 200
            )
        archive = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        note = client.get("/memory", params={"user_id": "dev"}).json()["notes"][0]
        assert note["source_turn_id"] == archive["session"]["turns"][0]["turn_id"]
        assert note["source_turn_id"] != archive["session"]["turns"][1]["turn_id"]
        assert client.get("/memory", params={"user_id": "other"}).json()["notes"] == []
    with TestClient(make_app(tmp_path, ScriptedProvider())) as client:
        assert client.get("/memory", params={"user_id": "dev"}).json()["notes"] == [note]
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archive


def test_runtime_injection_is_workspace_scoped_across_restart(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(id="local", name="remember_fact", arguments={"fact": "A convention"}),
                ToolCall(
                    id="global",
                    name="remember_fact",
                    arguments={"fact": "Global preference", "scope": "global"},
                ),
            ]
        ),
        make_response(content="Stored"),
        make_response(content="B answer"),
        make_response(content="A answer"),
    )
    with TestClient(make_app(tmp_path, provider)) as client:
        ids = [
            client.post("/sessions", json={"user_id": "dev", "workspace_root": str(root)}).json()[
                "session_id"
            ]
            for root in [a, b]
        ]
        for sid, message in [(ids[0], "Remember"), (ids[1], "Explain B"), (ids[0], "Explain A")]:
            assert (
                client.post(
                    "/chat", json={"user_id": "dev", "session_id": sid, "message": message}
                ).status_code
                == 200
            )
        assert "A convention" not in provider.calls[2][0][0].content
        assert "Global preference" in provider.calls[2][0][0].content
        assert "A convention" in provider.calls[3][0][0].content
        notes = client.get("/memory", params={"user_id": "dev"}).json()["notes"]
        assert {n["workspace_root"] for n in notes} == {str(a), None}
    restarted = ScriptedProvider()
    restarted.script(make_response(content="B after restart"))
    with TestClient(make_app(tmp_path, restarted)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(b)}).json()[
            "session_id"
        ]
        client.post("/chat", json={"user_id": "dev", "session_id": sid, "message": "Explain"})
        assert "A convention" not in restarted.calls[0][0][0].content
        assert "Global preference" in restarted.calls[0][0][0].content


def test_namespace_review_filters_before_paging_and_survives_restart(tmp_path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:

        def seed():
            store = client.app.state.components.fact_store
            store.add("dev", "global")
            store.add("dev", "older A", workspace_root="/project/a")
            store.add("dev", "B", workspace_root="/project/b")
            store.add("dev", "newer A", workspace_root="/project/a")
            store.add("other", "private A", workspace_root="/project/a")

        client.portal.call(seed)
    with TestClient(make_app(tmp_path, provider)) as client:
        params = {
            "user_id": "dev",
            "namespace": "workspace",
            "workspace_root": "/project/a",
            "limit": 1,
        }
        first = client.get("/memory", params=params).json()
        assert [n["preview"] for n in first["notes"]] == ["newer A"]
        second = client.get(
            "/memory", params={**params, "before_id": first["next_before_id"]}
        ).json()
        assert [n["preview"] for n in second["notes"]] == ["older A"]
        assert second["next_before_id"] is None
        global_page = client.get("/memory", params={"user_id": "dev", "namespace": "global"}).json()
        assert [n["preview"] for n in global_page["notes"]] == ["global"]
        assert len(client.get("/memory", params={"user_id": "dev"}).json()["notes"]) == 4
        for extra in [
            {"namespace": "invalid"},
            {"namespace": "workspace"},
            {"namespace": "workspace", "workspace_root": "relative"},
            {"namespace": "workspace", "workspace_root": "/project/../a"},
            {"namespace": "global", "workspace_root": "/project/a"},
            {"workspace_root": "/project/a"},
        ]:
            assert client.get("/memory", params={"user_id": "dev", **extra}).status_code == 422
        assert client.get("/runs", params={"user_id": "dev"}).json() == []
    assert provider.calls == []
