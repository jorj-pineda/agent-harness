import hashlib
import sqlite3

import pytest
from fastapi.testclient import TestClient

from providers.base import ToolCall

from .conftest import ScriptedProvider, make_response
from .test_session_persistence import make_app


def capture(client, root):
    sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(root)}).json()[
        "session_id"
    ]
    record = client.post(f"/sessions/{sid}/checkpoints", json={"user_id": "dev"}).json()
    return sid, record, f"/sessions/{sid}/checkpoints/{record['checkpoint_id']}/span"


def params(raw=b"saved\n", **changes):
    return {
        "user_id": "dev",
        "path": "a.txt",
        "expected_sha256": hashlib.sha256(raw).hexdigest(),
        "start_line": 1,
        "end_line": 1,
        **changes,
    }


@pytest.mark.parametrize("kind", ["manual", "turn_after"])
def test_exact_owned_span_survives_restart_without_workspace_or_new_work(
    tmp_path, monkeypatch, kind
):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "a.txt"
    raw = "first\r\n🙂second\nlast".encode()
    target.write_bytes(raw)
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider, max_tool_iterations=1)) as client:
        if kind == "manual":
            sid, record, endpoint = capture(client, root)
        else:
            target.write_text("before\n")
            provider.script(
                make_response(
                    tool_calls=[
                        ToolCall(
                            id="write",
                            name="write_file",
                            arguments={"path": "a.txt", "content": raw.decode()},
                        )
                    ]
                )
            )
            sid = client.post(
                "/sessions",
                json={"user_id": "dev", "workspace_root": str(root), "checkpoint_turns": True},
            ).json()["session_id"]
            answer = client.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Update a.txt"}
            ).json()
            run = client.get(f"/runs/{answer['run_id']}", params={"user_id": "dev"}).json()
            cid = run["after_checkpoint_id"]
            endpoint = f"/sessions/{sid}/checkpoints/{cid}/span"
            record = client.get(
                f"/sessions/{sid}/checkpoints/{cid}", params={"user_id": "dev"}
            ).json()
        archive = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        runs = client.get("/runs", params={"user_id": "dev"}).json()
        query = params(raw, start_line=2, end_line=3)
        first = client.get(endpoint, params=query)
        assert first.status_code == 200, first.text
        assert first.headers["cache-control"] == "no-store"
        span = first.json()
        assert span["content"] == "🙂second\nlast"
        assert (
            span["checkpoint_kind"] == kind
            and span["file_sha256"] == record["files"]["a.txt"]["sha256"]
        )
        assert span["selected_bytes"] == len(span["content"].encode())
        assert len(first.content) <= 32768
    target.write_text("later user work\n")
    root.rename(tmp_path / "moved")
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted)) as client:

        def forbidden(*args, **kwargs):
            pytest.fail("No current workspace, tools or provider resolution")

        monkeypatch.setattr("api.server._build_turn_registry", forbidden)
        monkeypatch.setattr(client.app.state.components.router, "resolve", forbidden)
        again = client.get(endpoint, params=query)
        assert again.content == first.content
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archive
        assert client.get("/runs", params={"user_id": "dev"}).json() == runs
        assert (
            client.get(
                f"/sessions/{sid}/checkpoints/{record['checkpoint_id']}", params={"user_id": "dev"}
            ).json()
            == record
        )
    assert (tmp_path / "moved/a.txt").read_text() == "later user work\n"
    assert restarted.calls == []
    assert len(provider.calls) == (1 if kind == "turn_after" else 0)


def test_owner_and_checkpoint_scope_are_checked_before_blob_access(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_bytes(b"saved\n")
    with TestClient(make_app(tmp_path, ScriptedProvider())) as client:
        sid, record, endpoint = capture(client, root)
        other = client.post(
            "/sessions", json={"user_id": "dev", "workspace_root": str(root)}
        ).json()["session_id"]

        def forbidden(*args):
            pytest.fail("No unauthorized blob read")

        monkeypatch.setattr(
            client.app.state.components.session_store, "checkpoint_content", forbidden
        )
        assert client.get(endpoint, params=params(user_id="other")).status_code == 403
        assert client.get(endpoint.replace(sid, other), params=params()).status_code == 404
        assert (
            client.get(
                endpoint.replace(record["checkpoint_id"], "missing"), params=params()
            ).status_code
            == 404
        )
        query = params()
        query.pop("user_id")
        assert client.get(endpoint, params=query).status_code == 422


@pytest.mark.parametrize(
    "changes",
    [
        {"path": "../a.txt"},
        {"path": ".env"},
        {"path": ".git/config"},
        {"path": "a//b"},
        {"start_line": 0},
        {"start_line": 3, "end_line": 2},
        {"end_line": 201},
        {"expected_sha256": "bad"},
    ],
)
def test_bad_selections_refuse_before_saved_content_access(tmp_path, monkeypatch, changes):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_bytes(b"saved\n")
    with TestClient(make_app(tmp_path, ScriptedProvider())) as client:
        _, _, endpoint = capture(client, root)

        def forbidden(*args):
            pytest.fail("Selection must reject before blob read")

        monkeypatch.setattr(
            client.app.state.components.session_store, "checkpoint_content", forbidden
        )
        result = client.get(endpoint, params=params(**changes))
        assert result.status_code == 422, result.text
        assert "saved\\n" not in result.text


@pytest.mark.parametrize(
    "raw,changes,status,code",
    [
        (b"saved\n", {"expected_sha256": "0" * 64}, 409, "hash_mismatch"),
        (b"saved\n", {"path": "missing"}, 404, "file_not_captured"),
        (b"saved\n", {"end_line": 2}, 422, "range_unavailable"),
        (b"\xff", {}, 415, "invalid_utf8"),
        (b"a" * 8193, {}, 413, "span_too_large"),
        (b"\x01" * 8192, {}, 413, "response_too_large"),
        (b"a" * 1_048_577, {}, 413, "file_too_large"),
    ],
)
def test_fixed_refusals_do_not_return_partial_source(tmp_path, raw, changes, status, code):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_bytes(raw)
    with TestClient(make_app(tmp_path, ScriptedProvider())) as client:
        _, _, endpoint = capture(client, root)
        result = client.get(endpoint, params=params(raw, **changes))
        assert result.status_code == status, result.text
        assert result.json() == {"detail": {"code": code}}


def test_corrupt_blob_and_storage_errors_are_private_and_non_mutating(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_bytes(b"saved\n")
    with TestClient(make_app(tmp_path, ScriptedProvider())) as client:
        _, record, endpoint = capture(client, root)
        store = client.app.state.components.session_store

        def corrupt():
            store._conn.execute("UPDATE checkpoint_files SET content=?", (b"other\n",))
            store._conn.commit()

        client.portal.call(corrupt)
        result = client.get(endpoint, params=params())
        assert result.status_code == 409 and result.json() == {
            "detail": {"code": "content_unavailable"}
        }

        def fail(*args):
            raise sqlite3.OperationalError("private storage marker")

        for method in ("get", "get_checkpoint", "checkpoint_content"):
            with monkeypatch.context() as patch:
                patch.setattr(store, method, fail)
                result = client.get(endpoint, params=params())
                assert result.status_code == 503 and "private" not in result.text
        assert (
            client.portal.call(store.get_checkpoint, record["checkpoint_id"]).model_dump(
                mode="json"
            )
            == record
        )
