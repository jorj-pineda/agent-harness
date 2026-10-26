"""Saved recovery evidence survives lifecycle restart without granting new work."""

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from harness.recovery_review import MAX_PACKET_BYTES, build_recovery_review
from harness.runs import RunInput
from harness.state import (
    CheckRecord,
    ToolCallRecord,
    TurnResponse,
    WorkspaceChangeReport,
    WorkspaceFileDiff,
)
from providers.base import ToolCall

from .conftest import ScriptedProvider, make_response
from .test_runs import HeldProvider
from .test_session_persistence import make_app
from .test_stream import _parse_sse


@pytest.mark.parametrize("transport", ["http", "stream"])
def test_partial_patch_packet_is_identical_after_restart_and_ignores_later_user_edits(
    tmp_path, transport, monkeypatch
):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "a.py"
    target.write_text("user = 1\n")
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="write",
                    name="write_file",
                    arguments={"path": "a.py", "content": "user = 1\nagent = 2\n"},
                )
            ]
        )
    )
    with TestClient(make_app(tmp_path, provider, max_tool_iterations=1)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(root)}).json()[
            "session_id"
        ]
        payload = {"user_id": "dev", "session_id": sid, "message": "Update a.py"}
        result = (
            client.post("/chat", json=payload)
            if transport == "http"
            else client.get("/chat/stream", params=payload)
        )
        answer = (
            result.json()
            if transport == "http"
            else json.loads(_parse_sse(result.text)[-1]["data"])["response"]
        )
        rid = answer["run_id"]
        endpoint = f"/runs/{rid}/recovery-review"
        saved = client.get(f"/runs/{rid}", params={"user_id": "dev"}).json()
        archive = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        packet = client.get(endpoint, params={"user_id": "dev"})
        assert packet.status_code == 200, packet.text
        assert packet.json()["completion_status"] == "budget_exhausted"
        assert "+agent = 2" in packet.json()["diffs"][0]["diff"]["text"]
        assert packet.json()["action_count"] == 1
    target.write_text("later user work\n")
    root.rename(tmp_path / "moved")  # Original workspace no longer exists.
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted)) as client:

        def no_execution(*args, **kwargs):
            pytest.fail("Recovery review must not build tools or select a provider")

        monkeypatch.setattr("api.server._build_turn_registry", no_execution)
        monkeypatch.setattr(client.app.state.components.router, "resolve", no_execution)
        again = client.get(endpoint, params={"user_id": "dev"})
        assert again.content == packet.content
        assert client.get(f"/runs/{rid}", params={"user_id": "dev"}).json() == saved
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archive
        assert len(client.get("/runs", params={"user_id": "dev"}).json()) == 1
        assert len(client.get("/sessions", params={"user_id": "dev"}).json()) == 1
    assert (tmp_path / "moved/a.py").read_text() == "later user work\n"
    assert restarted.calls == []
    assert len(provider.calls) == 1


@pytest.mark.parametrize("state", ["failed", "interrupted"])
def test_absent_final_evidence_stays_unknown_after_restart(tmp_path, state):
    from harness.session_store import SessionStore

    store = SessionStore(tmp_path / "sessions.db")
    record, _ = store.create_run(
        RunInput(user_id="dev", session_id="s", request_id="r", message="Fix"),
        provider="scripted",
        configured_model=None,
    )
    if state == "failed":
        store.fail_run(record.run_id)
    store.close()
    # Application startup marks leftover running records interrupted.
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:
        result = client.get(f"/runs/{record.run_id}/recovery-review", params={"user_id": "dev"})
        assert result.status_code == 200, result.text
        packet = result.json()
        assert packet["run_status"] == state
        assert not packet["final_response_available"]
        assert (
            packet["diff_count"]
            is packet["action_count"]
            is packet["unresolved_check_count"]
            is None
        )
        assert packet["completion_status"] is packet["verification_status"] is None
    assert provider.calls == []


def test_owner_and_running_guards_precede_extraction(tmp_path, monkeypatch):
    provider = HeldProvider()
    provider.script(make_response(content="Done"))
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        run = client.post(
            "/runs",
            json={"user_id": "dev", "session_id": sid, "message": "Explain", "request_id": "once"},
        ).json()
        endpoint = f"/runs/{run['run_id']}/recovery-review"

        def forbidden(*args):
            pytest.fail("Must reject before extraction")

        monkeypatch.setattr("api.server.build_recovery_review", forbidden)
        try:
            assert provider.started.wait(2)
            assert client.get(endpoint, params={"user_id": "dev"}).status_code == 409
            assert client.get(endpoint, params={"user_id": "other"}).status_code == 403
            assert (
                client.get("/runs/missing/recovery-review", params={"user_id": "dev"}).status_code
                == 404
            )
            assert client.get(endpoint).status_code == 422
            assert client.get(endpoint, params={"user_id": ""}).status_code == 422
        finally:
            provider.release.set()


def test_large_saved_text_is_bounded_on_wire_and_storage_unchanged(tmp_path):
    text = '\x00🙂"\\' * 10000
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:
        store = client.app.state.components.session_store

        def seed():
            record, _ = store.create_run(
                RunInput(user_id="dev", session_id="s", request_id="r", message=text),
                provider="scripted",
                configured_model=None,
            )
            store.finish_run(
                record.run_id,
                TurnResponse(
                    run_id=record.run_id,
                    answer="",
                    provider="scripted",
                    latency_ms=0,
                    completion_reason=text,
                    check_attempts=[
                        CheckRecord(argv=[text], status="failed", relevant=True, error=text)
                        for _ in range(20)
                    ],
                    tool_calls=[ToolCallRecord(name=text, error=text) for _ in range(20)],
                    workspace_changes=WorkspaceChangeReport(
                        status="tracked",
                        diffs=[WorkspaceFileDiff(path=text, diff=text) for _ in range(20)],
                    ),
                ),
            )
            return store.get_run(record.run_id)

        saved = client.portal.call(seed)
        result = client.get(f"/runs/{saved.run_id}/recovery-review", params={"user_id": "dev"})
        assert result.status_code == 200
        assert len(result.content) <= MAX_PACKET_BYTES
        assert result.json() == build_recovery_review(saved).model_dump(mode="json")
        assert (
            result.json()["objective"]["source_sha256"] == hashlib.sha256(text.encode()).hexdigest()
        )
        assert client.portal.call(store.get_run, saved.run_id) == saved
    assert provider.calls == []


def test_oversized_metadata_has_explicit_failure_not_partial_packet(tmp_path, monkeypatch):
    provider = ScriptedProvider()
    provider.script(make_response(content="Done"))
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        rid = client.post(
            "/chat", json={"user_id": "dev", "session_id": sid, "message": "Explain"}
        ).json()["run_id"]

        def oversized(record):
            raise ValueError("private metadata must not leak")

        monkeypatch.setattr("api.server.build_recovery_review", oversized)
        result = client.get(f"/runs/{rid}/recovery-review", params={"user_id": "dev"})
        assert result.status_code == 413
        assert "private" not in result.text
        assert "original run" in result.text
    assert len(provider.calls) == 1
