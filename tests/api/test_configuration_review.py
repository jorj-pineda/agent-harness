from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness.admission import turn_admission
from memory import FactStore
from providers.base import ToolCall

from .conftest import ScriptedProvider, make_response
from .test_session_persistence import make_app


def create_saved(root: Path, *, known: bool = True):
    repo = root / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("user = 1\n")
    provider = ScriptedProvider()
    if known:
        provider.name = "ollama"  # Configured identity exercised with a scripted adapter.
    provider.script(
        make_response(
            prompt_tokens=10,
            completion_tokens=2,
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "a.py", "content": "user = 1\nagent = 2\n"},
                )
            ],
        ),
        make_response(content="Done", prompt_tokens=10, completion_tokens=2),
    )
    with TestClient(make_app(root, provider, max_context_tokens=16000)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        assert (
            client.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Update a.py"}
            ).status_code
            == 200
        )
    return sid, repo


@pytest.mark.parametrize("mode", ["matching", "drift", "blocked", "unknown", "missing_workspace"])
def test_restart_review_preserves_history_files_and_run_identity(tmp_path: Path, mode):
    sid, repo = create_saved(tmp_path, known=mode != "unknown")
    provider = ScriptedProvider()
    if mode != "unknown":
        provider.name = "ollama"
    settings = {"max_context_tokens": 20 if mode == "blocked" else 16000}
    if mode == "drift":
        settings["ollama_model"] = "changed-model"
    if mode == "missing_workspace":
        (repo / "a.py").unlink()
        repo.rmdir()
    with FactStore(tmp_path / "memory.db") as facts:
        facts.add("dev", "fixture private owner fact")
        facts.add("other", "fixture private other fact")
    with TestClient(make_app(tmp_path, provider, **settings)) as client:
        endpoint = f"/sessions/{sid}/configuration-review"
        archive = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        runs = client.get("/runs", params={"user_id": "dev"}).json()
        review = client.post(
            endpoint, json={"user_id": "dev", "message": "fixture proposed message"}
        ).json()
        assert review["read_only"] is True
        assert review["resume_supported"] is False
        assert "fixture private" not in str(review)
        assert "fixture proposed" not in str(review)
        if mode == "missing_workspace":
            assert review["status"] == "unavailable"
            assert "WorkspaceError" in review["reason"]
        else:
            assert review["status"] == "reviewed"
            expected = "changed" if mode in {"drift", "blocked"} else mode
            assert review["configuration"]["status"] == expected
            assert review["transcript"]["status"] == "valid"
            assert review["context"]["status"] == (
                "blocked" if mode == "blocked" else "estimated_fit"
            )
            assert review["context"]["includes_proposed_message"] is True
            assert (repo / "a.py").read_text() == "user = 1\nagent = 2\n"
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archive
        assert client.get("/runs", params={"user_id": "dev"}).json() == runs
        assert (
            client.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Continue"}
            ).status_code
            == 409
        )
        assert client.post(endpoint, json={"user_id": "other"}).status_code == 403
        assert (
            client.post(
                "/sessions/missing/configuration-review", json={"user_id": "dev"}
            ).status_code
            == 404
        )
        assert (
            client.post(endpoint, json={"user_id": "dev", "provider": "missing"}).status_code == 400
        )
        assert client.post(endpoint, json={"user_id": ""}).status_code == 422
        assert provider.calls == []


def test_review_is_busy_for_overlapping_workspace_without_submitting_work(tmp_path: Path):
    sid, repo = create_saved(tmp_path)
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:
        with turn_admission.claim("another-session", str(repo)) as busy:
            assert busy is None
            result = client.post(
                f"/sessions/{sid}/configuration-review", json={"user_id": "dev"}
            ).json()
            assert result["status"] == "busy"
            assert result["configuration"] is None
        assert len(client.get("/runs", params={"user_id": "dev"}).json()) == 1
        assert provider.calls == []


def test_creation_only_archive_is_not_recorded_and_history_only_estimate_is_labeled(tmp_path: Path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        review = client.post(
            f"/sessions/{sid}/configuration-review", json={"user_id": "dev"}
        ).json()
        assert review["configuration"]["status"] == "not_recorded"
        assert review["context"]["status"] == "unbounded_context"
        assert review["context"]["includes_proposed_message"] is False
        assert client.get("/runs", params={"user_id": "dev"}).json() == []
        assert provider.calls == []
