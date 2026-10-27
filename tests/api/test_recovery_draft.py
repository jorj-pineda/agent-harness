"""Draft review uses current shared setup under admission without submitting any work."""

import hashlib
import sqlite3

import pytest
from fastapi.testclient import TestClient

from harness.admission import turn_admission
from providers.base import ChatMessage

from .conftest import ScriptedProvider, make_response
from .test_recovery_context import seed
from .test_runs import HeldProvider
from .test_session_persistence import make_app


def proposal(body):
    return {**body, "message": "Finish the original bounded task"}


@pytest.mark.parametrize("transport", ["http", "stream"])
def test_draft_restart_preserves_identity_user_text_and_edits_with_current_instruction_refresh(
    tmp_path, transport
):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "AGENTS.md").write_text("Original conventions\n")
    provider = ScriptedProvider()
    with TestClient(
        make_app(
            tmp_path,
            provider,
            max_tool_iterations=1,
            max_context_tokens=16384,
            load_root_agents=True,
        )
    ) as client:
        sid, rid, body = seed(client, root, provider, transport=transport)
        body = proposal(body)
        endpoint = f"/runs/{rid}/recovery-draft"
        run = client.get(f"/runs/{rid}", params={"user_id": "dev"}).json()
        archive = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        cid = run["after_checkpoint_id"]
        checkpoint = client.get(
            f"/sessions/{sid}/checkpoints/{cid}", params={"user_id": "dev"}
        ).json()
        first = client.post(endpoint, json=body)
        assert first.status_code == 200, first.text
        assert first.headers["cache-control"] == "no-store" and len(first.content) <= 131072
        data = first.json()
        assert data["context"]["status"] == "estimated_fit"
        assert data["configuration"]["status"] in {"matching", "unknown"}
        assert (
            data["budget_scope"] == "hypothetical-new-turn" and data["execution_supported"] is False
        )
        assert [m["content"] for m in data["messages"][1 : data["evidence_message_index"]]] == [
            m["content"] for m in archive["session"]["messages"] if m["role"] == "user"
        ]
    # Current work differs from captured source; draft construction is deliberately not a workspace lease.
    (root / "a.txt").write_text("later user edit\n")
    restarted = ScriptedProvider()
    with TestClient(
        make_app(
            tmp_path,
            restarted,
            max_tool_iterations=1,
            max_context_tokens=16384,
            load_root_agents=True,
        )
    ) as client:
        again = client.post(endpoint, json=body)
        assert again.content == first.content
        (root / "AGENTS.md").write_text("New conventions\n")
        refreshed = client.post(endpoint, json=body)
        assert refreshed.status_code == 200
        assert refreshed.json()["messages_sha256"] != data["messages_sha256"]
        assert "New conventions" in refreshed.json()["messages"][0]["content"]
        assert refreshed.json()["configuration"]["status"] == "changed"
        assert "repository_instructions" in refreshed.json()["configuration"]["changed_fields"]
        assert client.get(f"/runs/{rid}", params={"user_id": "dev"}).json() == run
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archive
        assert (
            client.get(f"/sessions/{sid}/checkpoints/{cid}", params={"user_id": "dev"}).json()
            == checkpoint
        )
        assert len(client.get("/runs", params={"user_id": "dev"}).json()) == 1
    assert (root / "a.txt").read_text() == "later user edit\n"
    assert len(provider.calls) == 1 and restarted.calls == []


def test_owner_active_and_busy_guards_precede_current_resource_construction(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider, max_tool_iterations=1)) as client:
        _, rid, body = seed(client, root, provider)
        body = proposal(body)
        endpoint = f"/runs/{rid}/recovery-draft"

        def forbidden(*args, **kwargs):
            pytest.fail("No current resources before ownership/admission")

        monkeypatch.setattr("api.server._build_turn_registry", forbidden)
        assert client.post(endpoint, json={**body, "user_id": "other"}).status_code == 403
        assert client.post("/runs/missing/recovery-draft", json=body).status_code == 404
        with turn_admission.claim("other-owned-task", str(root)) as busy:
            assert busy is None
            result = client.post(endpoint, json=body)
        assert result.status_code == 409 and result.json()["detail"]["code"] == "review_busy"


def test_active_run_refuses_before_archive_or_provider_review(tmp_path, monkeypatch):
    provider = HeldProvider()
    provider.script(make_response(content="Done"))
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        record = client.post(
            "/runs",
            json={"user_id": "dev", "session_id": sid, "message": "Explain", "request_id": "once"},
        ).json()
        try:
            assert provider.started.wait(2)
            monkeypatch.setattr(
                "api.server._lookup_archive", lambda *args: pytest.fail("No archive review")
            )
            result = client.post(
                f"/runs/{record['run_id']}/recovery-draft",
                json={
                    "user_id": "dev",
                    "expected_record_sha256": "0" * 64,
                    "spans": [],
                    "message": "Review",
                },
            )
            assert result.status_code == 409 and result.json()["detail"]["code"] == "run_active"
        finally:
            provider.release.set()


@pytest.mark.parametrize("change", ["run_id", "profile", "configured_model", "provider"])
def test_record_archive_identity_mismatches_refuse_before_source_blobs(
    tmp_path, monkeypatch, change
):
    root = tmp_path / "repo"
    root.mkdir()
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider, max_tool_iterations=1)) as client:
        _, rid, body = seed(client, root, provider)
        store = client.app.state.components.session_store
        record = client.portal.call(store.get_run, rid)
        if change == "run_id":
            record.run_id = "wrong-run"
        elif change == "profile":
            record.execution_profile = record.execution_profile.model_copy(
                update={"prompt_version": "wrong-version"}
            )
        elif change == "configured_model":
            record.configured_model = "wrong-model"
        else:
            record.provider = "wrong-provider"
        body = {
            **proposal(body),
            "expected_record_sha256": hashlib.sha256(record.model_dump_json().encode()).hexdigest(),
        }
        monkeypatch.setattr(store, "get_run", lambda _: record)
        monkeypatch.setattr(
            store, "checkpoint_content", lambda *args: pytest.fail("No mismatched blob read")
        )
        result = client.post(f"/runs/{rid}/recovery-draft", json=body)
        assert result.status_code == 409 and result.json() == {
            "detail": {"code": "archive_mismatch"}
        }


def test_older_source_run_refuses_after_archive_advances_and_releases_admission(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    provider = ScriptedProvider()
    with TestClient(
        make_app(tmp_path, provider, max_tool_iterations=1, max_context_tokens=16384)
    ) as client:
        sid, rid, body = seed(client, root, provider)
        provider.script(make_response(content="Another response"))
        later = client.post(
            "/chat", json={"user_id": "dev", "session_id": sid, "message": "Explain"}
        ).json()["run_id"]
        stale = client.post(f"/runs/{rid}/recovery-draft", json=proposal(body))
        assert stale.status_code == 409 and stale.json()["detail"]["code"] == "archive_mismatch"
        packet = client.get(f"/runs/{later}/recovery-review", params={"user_id": "dev"}).json()
        fresh = client.post(
            f"/runs/{later}/recovery-draft",
            json={
                "user_id": "dev",
                "expected_record_sha256": packet["source_record_sha256"],
                "spans": [],
                "message": "Review next",
            },
        )
        assert fresh.status_code == 200
        assert len(client.get("/runs", params={"user_id": "dev"}).json()) == 2
    assert len(provider.calls) == 2


@pytest.mark.parametrize("change", ["message", "utf8", "extra", "user", "hash"])
def test_bad_requests_do_not_build_current_resources(tmp_path, monkeypatch, change):
    root = tmp_path / "repo"
    root.mkdir()
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider, max_tool_iterations=1)) as client:
        _, rid, body = seed(client, root, provider)
        body = proposal(body)
        if change == "message":
            body["message"] = " "
        elif change == "utf8":
            body["message"] = "🙂" * 1025
        elif change == "extra":
            body["execute"] = True
        elif change == "user":
            body.pop("user_id")
        else:
            body["expected_record_sha256"] = "bad"
        monkeypatch.setattr(
            "api.server._build_turn_registry", lambda *args: pytest.fail("No resource build")
        )
        assert client.post(f"/runs/{rid}/recovery-draft", json=body).status_code == 422


@pytest.mark.parametrize("method", ["get_run", "get", "get_checkpoint", "checkpoint_content"])
def test_private_storage_errors_release_admission_and_never_return_partial_draft(
    tmp_path, monkeypatch, method
):
    root = tmp_path / "repo"
    root.mkdir()
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider, max_tool_iterations=1)) as client:
        _, rid, body = seed(client, root, provider)
        body = proposal(body)

        def fail(*args):
            raise sqlite3.OperationalError("private storage marker")

        with monkeypatch.context() as patch:
            patch.setattr(client.app.state.components.session_store, method, fail)
            result = client.post(f"/runs/{rid}/recovery-draft", json=body)
        assert result.status_code == 503 and "private" not in result.text
        assert client.post(f"/runs/{rid}/recovery-draft", json=body).status_code == 200


def test_unavailable_current_instructions_and_oversized_history_refuse(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    provider = ScriptedProvider()
    with TestClient(
        make_app(tmp_path, provider, max_tool_iterations=1, load_root_agents=True)
    ) as client:
        _, rid, body = seed(client, root, provider)
        body = proposal(body)
        (root / "AGENTS.md").write_bytes(b"x" * 8193)
        unavailable = client.post(f"/runs/{rid}/recovery-draft", json=body)
        assert (
            unavailable.status_code == 409
            and unavailable.json()["detail"]["code"] == "instructions_unavailable"
        )
        (root / "AGENTS.md").unlink()
        store = client.app.state.components.session_store
        archive = client.portal.call(
            store.get, client.portal.call(store.get_run, rid).request.session_id
        )
        archive.session.messages.append(ChatMessage(role="user", content="a" * 32769))
        monkeypatch.setattr(store, "get", lambda _: archive)
        too_large = client.post(f"/runs/{rid}/recovery-draft", json=body)
        assert (
            too_large.status_code == 413
            and too_large.json()["detail"]["code"] == "user_history_too_large"
        )


@pytest.mark.parametrize("limit,status", [(64, "blocked"), (0, "unbounded_context")])
def test_context_status_cannot_grant_execution(tmp_path, limit, status):
    root = tmp_path / "repo"
    root.mkdir()
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider, max_tool_iterations=1)) as client:
        _, rid, body = seed(client, root, provider)
        client.app.state.settings.max_context_tokens = limit
        result = client.post(f"/runs/{rid}/recovery-draft", json=proposal(body))
        assert result.status_code == 200, result.text
        assert result.json()["context"]["status"] == status
        assert result.json()["execution_supported"] is False
        assert len(provider.calls) == 1


def test_admission_covers_complete_draft_build_and_is_released(tmp_path, monkeypatch):
    import api.server as server

    root = tmp_path / "repo"
    root.mkdir()
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider, max_tool_iterations=1)) as client:
        sid, rid, body = seed(client, root, provider)
        original = server.build_recovery_draft
        observed = []

        def build(*args, **kwargs):
            with turn_admission.claim(sid, str(root)) as busy:
                observed.append(busy)
            return original(*args, **kwargs)

        monkeypatch.setattr(server, "build_recovery_draft", build)
        assert client.post(f"/runs/{rid}/recovery-draft", json=proposal(body)).status_code == 200
        assert len(observed) == 1 and observed[0] is not None
        with turn_admission.claim(sid, str(root)) as busy:
            assert busy is None


def test_missing_current_workspace_refuses_draft_but_historical_preview_remains_available(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider, max_tool_iterations=1)) as client:
        _, rid, body = seed(client, root, provider)
        root.rename(tmp_path / "moved")
        draft = client.post(f"/runs/{rid}/recovery-draft", json=proposal(body))
        assert draft.status_code == 503, draft.text
        history = client.post(f"/runs/{rid}/recovery-context", json=body)
        assert history.status_code == 200 and history.json()["spans"][0]["content"] == "saved\r\n"
    assert len(provider.calls) == 1
