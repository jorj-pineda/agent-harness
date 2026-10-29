"""Explicit read-only assembly, with real saved file tools and unchanged restart evidence."""

import hashlib
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from harness.runs import RunInput
from providers.base import ToolCall

from .conftest import ScriptedProvider, make_response
from .test_runs import HeldProvider
from .test_session_persistence import make_app
from .test_stream import _parse_sse


def seed(
    client, root, provider, raw="saved\r\nsecond\nlast", transport="http", *, known_usage=False
):
    (root / "a.txt").write_text("pre-existing user work\n")
    provider.script(
        make_response(
            prompt_tokens=10 if known_usage else None,
            completion_tokens=2 if known_usage else None,
            tool_calls=[
                ToolCall(id="write", name="write_file", arguments={"path": "a.txt", "content": raw})
            ],
        )
    )
    sid = client.post(
        "/sessions",
        json={
            "user_id": "dev",
            "workspace_root": str(root),
            "checkpoint_turns": True,
        },
    ).json()["session_id"]
    payload = {"user_id": "dev", "session_id": sid, "message": "Update a.txt"}
    if transport == "http":
        answer = client.post("/chat", json=payload).json()
    else:
        result = client.get("/chat/stream", params=payload)
        answer = json.loads(_parse_sse(result.text)[-1]["data"])["response"]
    rid = answer["run_id"]
    review = client.get(f"/runs/{rid}/recovery-review", params={"user_id": "dev"}).json()
    request = {
        "user_id": "dev",
        "expected_record_sha256": review["source_record_sha256"],
        "spans": [
            {
                "path": "a.txt",
                "expected_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "start_line": 1,
                "end_line": 1,
            }
        ],
    }
    return sid, rid, request


@pytest.mark.parametrize("transport", ["http", "stream"])
def test_context_preview_survives_restart_without_workspace_and_preserves_all_evidence(
    tmp_path, monkeypatch, transport
):
    root = tmp_path / "repo"
    root.mkdir()
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider, max_tool_iterations=1)) as client:
        sid, rid, payload = seed(client, root, provider, transport=transport)
        endpoint = f"/runs/{rid}/recovery-context"
        run = client.get(f"/runs/{rid}", params={"user_id": "dev"}).json()
        archive = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        cid = run["after_checkpoint_id"]
        checkpoint = client.get(
            f"/sessions/{sid}/checkpoints/{cid}", params={"user_id": "dev"}
        ).json()
        first = client.post(endpoint, json=payload)
        assert first.status_code == 200, first.text
        assert first.headers["cache-control"] == "no-store"
        assert first.json()["review"]["completion_status"] == "budget_exhausted"
        assert first.json()["spans"][0]["content"] == "saved\r\n"
        assert first.json()["execution_supported"] is False
        assert len(first.content) <= 65536
    (root / "a.txt").write_text("later user work\n")
    root.rename(tmp_path / "moved")
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted)) as client:

        def forbidden(*args, **kwargs):
            pytest.fail("No workspace, tools or provider resolution")

        monkeypatch.setattr("api.server._build_turn_registry", forbidden)
        monkeypatch.setattr(client.app.state.components.router, "resolve", forbidden)
        again = client.post(endpoint, json=payload)
        assert again.content == first.content
        assert client.get(f"/runs/{rid}", params={"user_id": "dev"}).json() == run
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archive
        assert (
            client.get(f"/sessions/{sid}/checkpoints/{cid}", params={"user_id": "dev"}).json()
            == checkpoint
        )
        assert len(client.get("/runs", params={"user_id": "dev"}).json()) == 1
    assert (tmp_path / "moved/a.txt").read_text() == "later user work\n"
    assert restarted.calls == [] and len(provider.calls) == 1


def test_owner_and_active_guards_precede_assembly(tmp_path, monkeypatch):
    provider = HeldProvider()
    provider.script(make_response(content="Done"))
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        record = client.post(
            "/runs",
            json={
                "user_id": "dev",
                "session_id": sid,
                "message": "Explain",
                "request_id": "once",
            },
        ).json()
        endpoint = f"/runs/{record['run_id']}/recovery-context"
        body = {"user_id": "dev", "expected_record_sha256": "0" * 64, "spans": []}
        try:
            assert provider.started.wait(2)
            result = client.post(endpoint, json=body)
            assert result.status_code == 409 and result.json()["detail"]["code"] == "run_active"

            def forbidden(*args):
                pytest.fail("Owner lookup must reject before assembly")

            monkeypatch.setattr("api.server.build_recovery_context", forbidden)
            assert client.post(endpoint, json={**body, "user_id": "other"}).status_code == 403
            assert client.post("/runs/missing/recovery-context", json=body).status_code == 404
        finally:
            provider.release.set()


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_missing_final_zero_span_preview_stays_unknown(tmp_path, status):
    from harness.session_store import SessionStore

    store = SessionStore(tmp_path / "sessions.db")
    record, _ = store.create_run(
        RunInput(user_id="dev", session_id="s", request_id="r", message="Fix"),
        provider="scripted",
        configured_model=None,
    )
    if status == "failed":
        store.fail_run(record.run_id)
    store.close()
    with TestClient(make_app(tmp_path, ScriptedProvider())) as client:
        review = client.get(
            f"/runs/{record.run_id}/recovery-review", params={"user_id": "dev"}
        ).json()
        result = client.post(
            f"/runs/{record.run_id}/recovery-context",
            json={
                "user_id": "dev",
                "expected_record_sha256": review["source_record_sha256"],
                "spans": [],
            },
        )
        assert result.status_code == 200
        assert result.json()["review"] == review
        assert result.json()["review"]["action_count"] is None
        assert result.json()["spans"] == []


@pytest.mark.parametrize("change", ["hash", "path", "range", "overlap", "count", "extra"])
def test_bad_requests_or_stale_record_refuse_before_source_reads(tmp_path, monkeypatch, change):
    root = tmp_path / "repo"
    root.mkdir()
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider, max_tool_iterations=1)) as client:
        _, rid, body = seed(client, root, provider)
        if change == "hash":
            body["expected_record_sha256"] = "0" * 64
        elif change == "path":
            body["spans"][0]["path"] = "../a.txt"
        elif change == "range":
            body["spans"][0]["start_line"] = True
        elif change == "overlap":
            body["spans"] *= 2
        elif change == "count":
            body["spans"] *= 5
        else:
            body["execute"] = True

        def forbidden(*args):
            pytest.fail("No blob read")

        monkeypatch.setattr(
            client.app.state.components.session_store, "checkpoint_content", forbidden
        )
        result = client.post(f"/runs/{rid}/recovery-context", json=body)
        assert result.status_code == (409 if change == "hash" else 422)


@pytest.mark.parametrize(
    "raw,status,code",
    [
        ("\x01" * 8192, 413, "response_too_large"),
        ("a" * 8193, 413, "span_too_large"),
        ("", 422, "range_unavailable"),
    ],
)
def test_source_refusals_are_fixed_and_no_partial_content(tmp_path, raw, status, code):
    root = tmp_path / "repo"
    root.mkdir()
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider, max_tool_iterations=1)) as client:
        _, rid, body = seed(client, root, provider, raw=raw)
        result = client.post(f"/runs/{rid}/recovery-context", json=body)
        assert result.status_code == status
        assert result.json() == {"detail": {"code": code}}


@pytest.mark.parametrize("method", ["get_run", "get_checkpoint", "checkpoint_content"])
def test_storage_errors_are_generic(tmp_path, monkeypatch, method):
    root = tmp_path / "repo"
    root.mkdir()
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider, max_tool_iterations=1)) as client:
        _, rid, body = seed(client, root, provider)

        def fail(*args):
            raise sqlite3.OperationalError("private marker")

        monkeypatch.setattr(client.app.state.components.session_store, method, fail)
        result = client.post(f"/runs/{rid}/recovery-context", json=body)
        assert result.status_code == 503 and "private" not in result.text


@pytest.mark.parametrize(
    "line,code",
    [
        ("a" * 8191 + "\n", "source_budget_exceeded"),
        ("\x01" * 4800 + "\n", "context_too_large"),
    ],
)
def test_aggregate_refusals_do_not_return_partial_preview(tmp_path, line, code):
    root = tmp_path / "repo"
    root.mkdir()
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider, max_tool_iterations=1)) as client:
        _, rid, body = seed(client, root, provider, raw=line * 3)
        body["spans"] = [{**body["spans"][0], "start_line": i, "end_line": i} for i in range(1, 4)]
        before = client.get(f"/runs/{rid}", params={"user_id": "dev"}).content
        result = client.post(f"/runs/{rid}/recovery-context", json=body)
        assert result.status_code == 413 and result.json() == {"detail": {"code": code}}
        assert client.get(f"/runs/{rid}", params={"user_id": "dev"}).content == before


def test_missing_checkpoint_and_wrong_file_hash_refuse_without_new_work(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider, max_tool_iterations=1)) as client:
        _, rid, body = seed(client, root, provider)
        endpoint = f"/runs/{rid}/recovery-context"
        body["spans"][0]["expected_sha256"] = "0" * 64
        result = client.post(endpoint, json=body)
        assert result.status_code == 409 and result.json()["detail"]["code"] == "hash_mismatch"
        monkeypatch.setattr(
            client.app.state.components.session_store, "get_checkpoint", lambda _: None
        )
        result = client.post(endpoint, json=body)
        assert (
            result.status_code == 409 and result.json()["detail"]["code"] == "checkpoint_mismatch"
        )
    assert len(provider.calls) == 1
