"""Read-only comparison to the last durable final workspace state."""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness.session_store import SessionStore
from providers.base import ToolCall

from .conftest import ScriptedProvider, make_response
from .test_session_persistence import make_app


def completed_session(root: Path, provider: ScriptedProvider):
    repo = root / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("user = 1\n")
    (repo / "gone.txt").write_text("keep")
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "a.py", "content": "user = 1\nagent = 2\n"},
                )
            ]
        ),
        make_response(content="Done"),
    )
    return repo


def test_saved_final_state_compares_after_restart_without_enabling_work(tmp_path: Path):
    provider = ScriptedProvider()
    repo = completed_session(tmp_path, provider)
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        assert (
            client.get(f"/sessions/{sid}/workspace-review", params={"user_id": "dev"}).json()[
                "status"
            ]
            == "not_recorded"
        )
        result = client.post(
            "/chat", json={"user_id": "dev", "session_id": sid, "message": "Update a.py"}
        ).json()
        assert result["workspace_changes"]["modified"] == ["a.py"]
        saved = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        review = client.get(f"/sessions/{sid}/workspace-review", params={"user_id": "dev"}).json()
        assert review["status"] == "unchanged"
        assert review["turn_id"] == saved["session"]["turns"][-1]["turn_id"]
        assert review["read_only"] is True
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted)) as client:
        endpoint = f"/sessions/{sid}/workspace-review"
        assert client.get(endpoint, params={"user_id": "dev"}).json()["status"] == "unchanged"
        (repo / "a.py").write_text("external edit\n")
        (repo / "new.txt").write_text("new")
        (repo / "gone.txt").unlink()
        review = client.get(endpoint, params={"user_id": "dev", "max_paths": 2}).json()
        assert review["status"] == "changed"
        assert review["added"] == ["new.txt"]
        assert review["modified"] == ["a.py"]
        assert review["deleted"] == []
        assert review["deleted_count"] == 1
        assert review["paths_truncated"] is True
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
        assert (
            client.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Continue"}
            ).status_code
            == 409
        )
        assert client.get(endpoint, params={"user_id": "other"}).status_code == 403
        assert (
            client.get("/sessions/missing/workspace-review", params={"user_id": "dev"}).status_code
            == 404
        )
        assert client.get(endpoint, params={"user_id": "dev", "max_paths": 0}).status_code == 422
        assert restarted.calls == []


@pytest.mark.parametrize("mode", ["disabled", "no_workspace", "over_limit", "missing", "replaced"])
def test_unavailable_state_is_not_reported_as_unchanged(tmp_path: Path, mode: str):
    provider = ScriptedProvider()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a").write_text("one")
    (repo / "b").write_text("two")
    overrides = (
        {"track_workspace_changes": False}
        if mode == "disabled"
        else {"max_tracked_files": 1}
        if mode == "over_limit"
        else {}
    )
    provider.script(make_response(content="Explained"))
    with TestClient(make_app(tmp_path, provider, **overrides)) as client:
        payload = {"user_id": "dev"}
        if mode != "no_workspace":
            payload["workspace_root"] = str(repo)
        sid = client.post("/sessions", json=payload).json()["session_id"]
        assert (
            client.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Explain"}
            ).status_code
            == 200
        )
        if mode == "missing":
            (repo / "a").unlink()
            (repo / "b").unlink()
            repo.rmdir()
        elif mode == "replaced":
            repo.rename(tmp_path / "old")
            repo.mkdir()
            (repo / "a").write_text("one")
            (repo / "b").write_text("two")
        review = client.get(f"/sessions/{sid}/workspace-review", params={"user_id": "dev"}).json()
        assert review["status"] == (
            "not_tracked" if mode in ("disabled", "no_workspace") else "unavailable"
        )
        assert review["reason"]
        assert review["ignored_entries"]["status"] == "not_checked"


def test_ignored_content_and_environment_hashes_are_excluded_explicitly(tmp_path: Path):
    provider = ScriptedProvider()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text("fixture-secret-before")
    (repo / ".git").mkdir()
    (repo / ".git" / "index").write_text("original index")
    (repo / "a.py").write_text("user edit\n")
    provider.script(make_response(content="Explained"))
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        client.post("/chat", json={"user_id": "dev", "session_id": sid, "message": "Explain"})
        (repo / ".env").write_text("fixture-secret-after")
        (repo / ".git" / "index").write_text("user index change")
        review = client.get(f"/sessions/{sid}/workspace-review", params={"user_id": "dev"}).json()
        assert review["status"] == "unchanged"
        assert ".env" in review["excluded_patterns"] and ".git" in review["excluded_patterns"]
        assert "not permission" in review["scope_note"]
        assert (repo / "a.py").read_text() == "user edit\n"
        assert (repo / ".git" / "index").read_text() == "user index change"
    store = SessionStore(tmp_path / "sessions.db")
    baseline = store.get_workspace_review(sid).baseline
    assert ".env" not in baseline.file_modes
    assert ".git/index" not in baseline.file_modes
    assert ".env" not in baseline.fingerprints
    assert ".git/index" not in baseline.fingerprints
    assert "fixture-secret" not in baseline.model_dump_json()
    store.close()


def test_workspace_review_and_active_turn_cannot_overlap(tmp_path: Path, monkeypatch):
    import api.server as server

    provider = ScriptedProvider()
    repo = completed_session(tmp_path, provider)
    entered, release = threading.Event(), threading.Event()
    original = server.compare_workspace

    def hold(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        client.post("/chat", json={"user_id": "dev", "session_id": sid, "message": "Update a.py"})
        monkeypatch.setattr(server, "compare_workspace", hold)
        with ThreadPoolExecutor() as pool:
            pending = pool.submit(
                client.get, f"/sessions/{sid}/workspace-review", params={"user_id": "dev"}
            )
            assert entered.wait(2)
            try:
                busy = client.get(
                    f"/sessions/{sid}/workspace-review", params={"user_id": "dev"}
                ).json()
                assert busy["status"] == "busy"
                blocked = client.post(
                    "/chat", json={"user_id": "dev", "session_id": sid, "message": "New turn"}
                ).json()
                assert blocked["completion_status"] == "blocked"
                assert len(provider.calls) == 2
            finally:
                release.set()
            assert pending.result(timeout=5).json()["status"] == "unchanged"


def test_old_archives_have_no_invented_baseline(tmp_path: Path):
    from harness.state import Session

    store = SessionStore(tmp_path / "sessions.db")
    session = Session(user_id="dev")
    store.save(session)
    store.close()
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:
        review = client.get(
            f"/sessions/{session.session_id}/workspace-review", params={"user_id": "dev"}
        ).json()
        assert review["status"] == "not_recorded"
        assert review["baseline_captured_at"] is None
        assert provider.calls == []


def test_comparison_read_failure_and_new_limits_return_unavailable(tmp_path: Path, monkeypatch):
    import workspace.changes as changes

    provider = ScriptedProvider()
    repo = completed_session(tmp_path, provider)
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        client.post("/chat", json={"user_id": "dev", "session_id": sid, "message": "Update a.py"})
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted, max_tracked_files=1)) as client:
        review = client.get(f"/sessions/{sid}/workspace-review", params={"user_id": "dev"}).json()
        assert review["status"] == "unavailable"
        assert "SnapshotLimitError" in review["reason"]
    with TestClient(make_app(tmp_path, restarted)) as client:

        def fail(*args, **kwargs):
            raise PermissionError("fixture failure; never expose contents")

        monkeypatch.setattr(changes, "_hash_file", fail)
        review = client.get(f"/sessions/{sid}/workspace-review", params={"user_id": "dev"}).json()
        assert review["status"] == "unavailable"
        assert "PermissionError" in review["reason"]
        assert "fixture failure" not in review["reason"]
        assert restarted.calls == []


def test_permission_only_drift_is_durable_capped_and_read_only(tmp_path: Path):
    provider = ScriptedProvider()
    repo = completed_session(tmp_path, provider)
    (repo / "a.py").chmod(0o640)
    (repo / "gone.txt").chmod(0o600)
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        client.post("/chat", json={"user_id": "dev", "session_id": sid, "message": "Update"})
        saved = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
    store = SessionStore(tmp_path / "sessions.db")
    baseline = store.get_workspace_review(sid).baseline
    assert baseline.file_modes == {"a.py": 0o640, "gone.txt": 0o600}
    store.close()
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted)) as client:
        endpoint = f"/sessions/{sid}/workspace-review"
        review = client.get(endpoint, params={"user_id": "dev"}).json()
        assert review["status"] == "unchanged"
        assert review["mode_coverage"] == "compared"
        (repo / "a.py").chmod(0o750)
        (repo / "gone.txt").chmod(0o644)
        review = client.get(endpoint, params={"user_id": "dev", "max_paths": 1}).json()
        assert review["status"] == "changed"
        assert review["modified"] == ["a.py"]
        assert review["modified_count"] == 2
        assert review["paths_truncated"] is True
        assert (repo / "a.py").read_text() == "user = 1\nagent = 2\n"
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
        assert (
            client.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Continue"}
            ).status_code
            == 409
        )
        assert restarted.calls == []


@pytest.mark.parametrize("limited", [False, True])
def test_ignored_metadata_survives_restart_without_reading_or_replaying_work(tmp_path, limited):
    provider = ScriptedProvider()
    repo = completed_session(tmp_path, provider)
    (repo / ".git").mkdir()
    (repo / ".git" / "index").write_text("preserved user index")
    (repo / ".env").write_text("fixture private environment")
    with TestClient(
        make_app(tmp_path, provider, max_tool_iterations=1 if limited else 8)
    ) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        result = client.post(
            "/chat", json={"user_id": "dev", "session_id": sid, "message": "Update a.py"}
        ).json()
        assert result["completion_status"] == ("budget_exhausted" if limited else "completed")
        assert result["workspace_changes"]["ignored_entries"]["status"] == "unchanged"
        archive = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        runs = client.get("/runs", params={"user_id": "dev"}).json()
    store = SessionStore(tmp_path / "sessions.db")
    baseline = store.get_workspace_review(sid).baseline
    assert set(baseline.ignored_entry_fingerprints) == {".git"}
    assert ".git/index" not in baseline.model_dump_json()
    assert "preserved user index" not in baseline.model_dump_json()
    assert "fixture private environment" not in baseline.model_dump_json()
    store.close()
    (repo / ".git" / "new-entry").write_text("external change")
    (repo / ".venv").mkdir()
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted)) as client:
        review = client.get(
            f"/sessions/{sid}/workspace-review", params={"user_id": "dev", "max_paths": 1}
        ).json()
        assert review["status"] == "changed"
        assert review["added_count"] == review["modified_count"] == review["deleted_count"] == 0
        ignored = review["ignored_entries"]
        assert ignored["status"] == "changed"
        assert ignored["added"] == [".venv"]
        assert ignored["modified"] == []
        assert ignored["added_count"] == ignored["modified_count"] == 1
        assert ignored["paths_truncated"] is True
        assert (repo / ".git" / "index").read_text() == "preserved user index"
        assert (repo / "a.py").read_text() == "user = 1\nagent = 2\n"
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archive
        assert client.get("/runs", params={"user_id": "dev"}).json() == runs
        assert (
            client.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Resume"}
            ).status_code
            == 409
        )
        assert restarted.calls == []
