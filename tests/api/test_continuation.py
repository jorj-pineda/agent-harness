import pytest
from fastapi.testclient import TestClient

from providers.base import ToolCall

from .conftest import ScriptedProvider, make_response
from .test_session_persistence import make_app

SETTINGS = {"max_context_tokens": 8192, "ollama_num_ctx": 8192, "ollama_model": "gemma4:12b"}


def known_provider():
    provider = ScriptedProvider()
    provider.name = "ollama"
    return provider


def source(tmp_path, *, known=True):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("user = 1\n")
    provider = known_provider() if known else ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="old-edit",
                    name="write_file",
                    arguments={"path": "a.py", "content": "user = 1\nagent = 2\n"},
                )
            ]
        ),
        make_response(content="Done"),
    )
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:
        sid = client.post(
            "/sessions",
            json={"user_id": "dev", "workspace_root": str(root), "checkpoint_turns": True},
        ).json()["session_id"]
        assert (
            client.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Update a.py"}
            ).status_code
            == 200
        )
        archived = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
    return root, sid, archived


def body(archived):
    return {
        "user_id": "dev",
        "request_id": "continue-once",
        "expected_turn_id": archived["session"]["turns"][-1]["turn_id"],
        "message": "Continue: update a.py",
        "acknowledge_uncaptured_state": True,
    }


def test_continuation_forks_after_restart_without_unchecking_source_or_replaying(tmp_path):
    root, sid, archived = source(tmp_path)
    provider = known_provider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="new-edit",
                    name="write_file",
                    arguments={"path": "a.py", "content": "user = 1\nagent = 2\ncontinued = 3\n"},
                )
            ]
        ),
        make_response(content="Continued"),
    )
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:
        request = body(archived)
        endpoint = f"/sessions/{sid}/continue"
        result = client.post(endpoint, json=request)
        assert result.status_code == 202, result.text
        run = result.json()
        fork = run["request"]["session_id"]
        assert fork != sid
        # Joining this app-owned task via its exact continuation input is read-only.
        client.portal.call(async_wait, client.app.state.components, run["run_id"])
        completed = client.get("/runs/" + run["run_id"], params={"user_id": "dev"}).json()
        assert completed["status"] == "finished", completed
        assert completed["response"]["completion_status"] == "completed"
        assert completed["before_checkpoint_id"] and completed["after_checkpoint_id"]
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archived
        saved_fork = client.get(f"/sessions/{fork}", params={"user_id": "dev"}).json()
        assert len(saved_fork["responses"]) == len(archived["responses"]) + 1
        assert saved_fork["session"]["continuation_source"]["session_id"] == sid
        assert (
            client.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Continue"}
            ).status_code
            == 409
        )
        assert client.post(endpoint, json=request).json() == completed
        assert (
            client.post(endpoint, json={**request, "message": "Different input"}).status_code == 409
        )
        assert len(provider.calls) == 2
        assert any(message.role == "tool" for message in provider.calls[0][0])
    restarted = known_provider()
    with TestClient(make_app(tmp_path, restarted, **SETTINGS)) as client:
        assert client.post(endpoint, json=request).json() == completed
    assert restarted.calls == []
    assert (root / "a.py").read_text() == "user = 1\nagent = 2\ncontinued = 3\n"


async def async_wait(components, run_id):
    task = components.run_tasks.get(run_id)
    if task is not None:
        await task


@pytest.mark.parametrize("failure", ["workspace", "configuration", "context", "unknown"])
def test_guard_rejections_do_not_infer_or_edit_source(tmp_path, failure):
    root, sid, archived = source(tmp_path, known=failure != "unknown")
    provider = known_provider() if failure != "unknown" else ScriptedProvider()
    overrides = dict(SETTINGS)
    request = body(archived)
    if failure == "workspace":
        (root / "a.py").write_text("later user edit\n")
    elif failure == "configuration":
        overrides["max_tool_calls_per_turn"] = 1
    elif failure == "context":
        request["message"] = "z" * 50_000
    expected_bytes = (root / "a.py").read_bytes()
    with TestClient(make_app(tmp_path, provider, **overrides)) as client:
        submitted = client.post(f"/sessions/{sid}/continue", json=request)
        assert submitted.status_code == 202, submitted.text
        rid = submitted.json()["run_id"]
        client.portal.call(async_wait, client.app.state.components, rid)
        run = client.get("/runs/" + rid, params={"user_id": "dev"}).json()
        assert run["status"] == "finished"
        assert run["response"]["completion_status"] == "blocked", run
        assert run["before_checkpoint_id"] is None
        assert run["execution_profile"] is None
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archived
        fork = run["request"]["session_id"]
        assert (
            client.post(
                "/chat", json={"user_id": "dev", "session_id": fork, "message": "Bypass guard"}
            ).status_code
            == 409
        )
    assert provider.calls == []
    assert (root / "a.py").read_bytes() == expected_bytes


def test_continuation_requires_explicit_scope_latest_turn_and_ownership(tmp_path):
    _root, sid, archived = source(tmp_path)
    provider = known_provider()
    request = body(archived)
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:
        endpoint = f"/sessions/{sid}/continue"
        assert (
            client.post(
                endpoint,
                json={k: v for k, v in request.items() if k != "acknowledge_uncaptured_state"},
            ).status_code
            == 422
        )
        assert (
            client.post(
                endpoint, json={**request, "acknowledge_uncaptured_state": False}
            ).status_code
            == 422
        )
        assert client.post(endpoint, json={**request, "user_id": "other"}).status_code == 403
        assert (
            client.post(endpoint, json={**request, "expected_turn_id": "old-turn"}).status_code
            == 409
        )
        assert len(client.get("/sessions", params={"user_id": "dev"}).json()) == 1
    assert provider.calls == []


def submit_and_finish(client, sid, request):
    result = client.post(f"/sessions/{sid}/continue", json=request)
    assert result.status_code == 202, result.text
    rid = result.json()["run_id"]
    client.portal.call(async_wait, client.app.state.components, rid)
    return client.get(f"/runs/{rid}", params={"user_id": "dev"}).json()


def test_live_fork_rechecks_each_turn_and_retires_after_changed_workspace(tmp_path):
    root, sid, archived = source(tmp_path)
    provider = known_provider()
    provider.script(make_response(content="Reviewed"), make_response(content="Reviewed again"))
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:
        run = submit_and_finish(client, sid, body(archived))
        fork = run["request"]["session_id"]
        request = {"user_id": "dev", "session_id": fork, "message": "Explain a.py"}
        second = client.post("/chat", json=request)
        assert second.status_code == 200, second.text
        assert second.json()["completion_status"] == "completed"
        assert len(provider.calls) == 2
        before = client.get(f"/sessions/{fork}", params={"user_id": "dev"}).json()
        (root / "a.py").write_text("later user work\n")
        blocked = client.post("/chat", json=request)
        assert blocked.status_code == 200, blocked.text
        assert blocked.json()["completion_status"] == "blocked"
        assert len(provider.calls) == 2
        assert client.get(f"/sessions/{fork}", params={"user_id": "dev"}).json() == before
        assert client.post("/chat", json=request).status_code == 409
    assert (root / "a.py").read_text() == "later user work\n"


def test_newer_failed_source_run_blocks_even_when_workspace_is_unchanged(tmp_path):
    from harness.runs import RunInput

    _root, sid, archived = source(tmp_path)
    provider = known_provider()
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:

        def failed_work():
            store = client.app.state.components.session_store
            record, _ = store.create_run(
                RunInput(
                    user_id="dev", session_id=sid, message="Unsaved work", request_id="failed"
                ),
                provider="ollama",
                configured_model="gemma4:12b",
            )
            store.fail_run(record.run_id)

        client.portal.call(failed_work)
        run = submit_and_finish(client, sid, body(archived))
        assert run["response"]["completion_status"] == "blocked"
        assert "newer unfinished/failed" in run["response"]["completion_reason"]
        assert run["before_checkpoint_id"] is None
    assert provider.calls == []


def test_busy_workspace_blocks_and_cannot_bypass_guard(tmp_path):
    from harness.admission import turn_admission

    root, sid, archived = source(tmp_path)
    provider = known_provider()
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:
        with turn_admission.claim("other-session", str(root)) as reason:
            assert reason is None
            run = submit_and_finish(client, sid, body(archived))
        assert run["response"]["completion_status"] == "blocked"
        assert run["before_checkpoint_id"] is None
        fork = run["request"]["session_id"]
        assert (
            client.post(
                "/chat", json={"user_id": "dev", "session_id": fork, "message": "Go"}
            ).status_code
            == 409
        )
    assert provider.calls == []


def test_fork_and_run_storage_are_atomic_before_any_execution(tmp_path):
    _root, sid, archived = source(tmp_path)
    provider = known_provider()
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:

        def reject_fork():
            store = client.app.state.components.session_store
            store._conn.execute(
                "CREATE TRIGGER reject_fork BEFORE INSERT ON sessions "
                "BEGIN SELECT RAISE(ABORT, 'unavailable'); END"
            )

        client.portal.call(reject_fork)
        result = client.post(f"/sessions/{sid}/continue", json=body(archived))
        assert result.status_code == 503, result.text
        assert (
            client.get("/runs", params={"user_id": "dev", "request_id": "continue-once"}).json()
            == []
        )
        assert len(client.get("/sessions", params={"user_id": "dev"}).json()) == 1
        assert client.app.state.components.sessions == {}
    assert provider.calls == []


def test_invalid_provider_creates_no_fork_or_run(tmp_path):
    _root, sid, archived = source(tmp_path)
    provider = known_provider()
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:
        result = client.post(
            f"/sessions/{sid}/continue", json={**body(archived), "provider": "missing"}
        )
        assert result.status_code == 400
        assert len(client.get("/sessions", params={"user_id": "dev"}).json()) == 1
        assert client.app.state.components.sessions == {}
    assert provider.calls == []


@pytest.mark.parametrize("corruption", ["transcript", "missing-checkpoint"])
def test_damaged_saved_evidence_cannot_grant_continuation(tmp_path, corruption):
    import json

    _root, sid, archived = source(tmp_path)
    provider = known_provider()
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:

        def damage():
            store = client.app.state.components.session_store
            if corruption == "transcript":
                damaged = dict(archived)
                damaged["session"]["messages"].append(
                    {"role": "tool", "content": "unmatched", "tool_call_id": "missing"}
                )
                with store._conn:
                    store._conn.execute(
                        "UPDATE sessions SET archive=? WHERE session_id=?",
                        (json.dumps(damaged), sid),
                    )
            else:
                old_run = store.get_run(archived["responses"][-1]["response"]["run_id"])
                with store._conn:
                    store._conn.execute(
                        "DELETE FROM checkpoints WHERE checkpoint_id=?",
                        (old_run.after_checkpoint_id,),
                    )

        client.portal.call(damage)
        run = submit_and_finish(client, sid, body(archived))
        assert run["response"]["completion_status"] == "blocked", run
        assert run["before_checkpoint_id"] is None
        assert run["execution_profile"] is None
    assert provider.calls == []


def test_failed_related_fork_blocks_another_fork_from_original_source(tmp_path):
    from harness.runs import RunInput

    _root, sid, archived = source(tmp_path)
    provider = known_provider()
    provider.script(make_response(content="Reviewed"))
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:
        first = submit_and_finish(client, sid, body(archived))
        fork = first["request"]["session_id"]

        def unfinished_fork():
            store = client.app.state.components.session_store
            record, _ = store.create_run(
                RunInput(user_id="dev", session_id=fork, message="Unsaved", request_id="lost-fork"),
                provider="ollama",
                configured_model="gemma4:12b",
            )
            store.fail_run(record.run_id)

        client.portal.call(unfinished_fork)
        second = submit_and_finish(client, sid, {**body(archived), "request_id": "another-fork"})
        assert second["response"]["completion_status"] == "blocked", second
        assert second["before_checkpoint_id"] is None
        assert len(provider.calls) == 1
