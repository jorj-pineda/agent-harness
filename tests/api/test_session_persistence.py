from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.server import Components, create_app
from harness.config import Settings
from harness.grounding import Grounder
from harness.router import ProviderRouter
from memory import FactStore
from providers.base import ToolCall

from .conftest import ScriptedProvider, make_response
from .test_stream import _parse_sse


def make_app(root: Path, provider: ScriptedProvider, **overrides):
    settings = Settings(
        _env_file=None,
        default_provider="scripted",
        memory_db_path=root / "memory.db",
        session_db_path=root / "sessions.db",
        **overrides,
    )

    def factory(_):
        providers = {"scripted": provider}
        return Components(
            providers=providers,
            router=ProviderRouter(providers, default="scripted"),
            embedder=None,
            collection=None,
            fact_store=FactStore(settings.memory_db_path),
            grounder=Grounder(escalation_threshold=0.55),
        )

    return create_app(settings=settings, components_factory=factory)


@pytest.mark.parametrize("transport", ["http", "stream"])
@pytest.mark.parametrize("limited", [False, True])
def test_restart_preserves_final_review_and_blocks_continuation(
    tmp_path: Path,
    transport: str,
    limited: bool,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("user = 1\n")
    provider = ScriptedProvider()
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
        make_response(content="Review ready"),
    )
    app = make_app(tmp_path, provider, max_tool_iterations=1 if limited else 8)
    with TestClient(app) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        payload = {"user_id": "dev", "session_id": sid, "message": "Update a.py"}
        if transport == "http":
            result = client.post("/chat", json=payload)
            assert result.status_code == 200, result.text
            response = result.json()
        else:
            result = client.get("/chat/stream", params=payload)
            assert result.status_code == 200, result.text
            response = json.loads(_parse_sse(result.text)[-1]["data"])["response"]
        assert response["completion_status"] == ("budget_exhausted" if limited else "completed")
        assert "+agent = 2\n" in response["workspace_changes"]["diffs"][0]["diff"]
        saved = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        assert saved["responses"][0]["response"] == response
        if not limited:
            provider.script(make_response(content="Second answer"))
            assert (
                client.post("/chat", json={**payload, "message": "Explain it"}).status_code == 200
            )
            saved = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
            assert len(saved["responses"]) == 2
    # Review must not resolve the workspace or require it to exist at restart.
    (repo / "a.py").unlink()
    repo.rmdir()
    restarted_provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted_provider)) as client:
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
        assert saved["read_only"] is True
        assert saved["session"]["messages"]
        assert saved["responses"][0]["turn_id"] == saved["session"]["turns"][0]["turn_id"]
        listing = client.get("/sessions", params={"user_id": "dev"}).json()
        assert listing[0]["session_id"] == sid
        assert listing[0]["turn_count"] == len(saved["responses"])
        assert client.get("/sessions", params={"user_id": "other"}).json() == []
        for endpoint in (f"/sessions/{sid}", "/chat/stream"):
            assert client.get(endpoint, params={**payload, "user_id": "other"}).status_code == 403
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).status_code == 200
        assert client.post("/chat", json=payload).status_code == 409
        assert client.get("/chat/stream", params=payload).status_code == 409
        assert restarted_provider.calls == []
        assert client.get("/sessions/missing", params={"user_id": "dev"}).status_code == 404
        assert client.get("/sessions", params={"user_id": "dev", "limit": 101}).status_code == 422
        assert client.get("/sessions", params={"user_id": "dev", "offset": 1}).json() == []


def test_failed_turn_leaves_last_committed_snapshot(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    provider.script(make_response(content="Saved answer"))
    with TestClient(make_app(tmp_path, provider), raise_server_exceptions=False) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        payload = {"user_id": "dev", "session_id": sid, "message": "Explain"}
        assert client.post("/chat", json=payload).status_code == 200
        before = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        # Provider queue exhaustion raises after the new Turn is appended.
        assert client.post("/chat", json=payload).status_code == 500
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == before
        assert client.post("/chat", json=payload).status_code == 409
    with TestClient(make_app(tmp_path, ScriptedProvider())) as client:
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == before


def test_archive_failure_surfaces_and_prevents_retry_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    from harness.session_store import SessionStore

    provider = ScriptedProvider()
    provider.script(make_response(content="Unsaved answer"))
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        before = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()

        def fail(*args, **kwargs):
            raise sqlite3.OperationalError("disk full")

        monkeypatch.setattr(SessionStore, "finish_run", fail)
        payload = {"user_id": "dev", "session_id": sid, "message": "Explain"}
        result = client.post("/chat", json=payload)
        assert result.status_code == 503
        assert "workspace edits may exist" in result.json()["detail"]
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == before
        assert client.post("/chat", json=payload).status_code == 409
        assert len(provider.calls) == 1
