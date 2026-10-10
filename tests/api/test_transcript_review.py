from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from harness.session_store import SessionStore
from harness.state import Session, Turn
from providers.base import ChatMessage, ToolCall

from .conftest import ScriptedProvider, make_response
from .test_session_persistence import make_app


def test_saved_dangling_history_review_survives_restart_without_repair(tmp_path: Path):
    session = Session(
        user_id="dev",
        messages=[
            ChatMessage(role="user", content="fixture private task"),
            ChatMessage(role="assistant", tool_calls=[ToolCall(id="a", name="write_file")]),
            ChatMessage(role="user", content="continue"),
            ChatMessage(role="tool", tool_call_id="a", content="fixture private result"),
        ],
        turns=[Turn(user_input="task", finished_at=datetime.now(UTC))],
    )
    store = SessionStore(tmp_path / "sessions.db")
    store.save(session)
    store.close()
    for _ in range(2):
        provider = ScriptedProvider()
        with TestClient(make_app(tmp_path, provider)) as client:
            endpoint = f"/sessions/{session.session_id}/transcript-review"
            before = client.get(f"/sessions/{session.session_id}", params={"user_id": "dev"}).json()
            review = client.get(endpoint, params={"user_id": "dev", "max_issues": 1}).json()
            assert review["status"] == "invalid"
            assert review["turn_id"] == session.turns[-1].turn_id
            assert review["issue_count"] == 2
            assert review["issues"] == [{"message_index": 1, "code": "missing_tool_result"}]
            assert review["issues_truncated"] is True
            assert "fixture private" not in str(review)
            assert (
                client.get(f"/sessions/{session.session_id}", params={"user_id": "dev"}).json()
                == before
            )
            assert client.get(endpoint, params={"user_id": "other"}).status_code == 403
            assert (
                client.get(
                    "/sessions/missing/transcript-review", params={"user_id": "dev"}
                ).status_code
                == 404
            )
            assert (
                client.get(endpoint, params={"user_id": "dev", "max_issues": 0}).status_code == 422
            )
            assert (
                client.post(
                    "/chat",
                    json={
                        "session_id": session.session_id,
                        "user_id": "dev",
                        "message": "continue",
                    },
                ).status_code
                == 409
            )
            assert provider.calls == []


def test_malformed_model_batch_cannot_edit_and_saved_review_is_structurally_valid(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "a.py"
    target.write_text("user edit\n")
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="same", name="write_file", arguments={"path": "a.py", "content": "overwrite"}
                ),
                ToolCall(id="same", name="read_file", arguments={"path": "a.py"}),
            ]
        )
    )
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        payload = {
            "user_id": "dev",
            "session_id": sid,
            "message": "task",
            "request_id": "bad-batch",
        }
        response = client.post("/chat", json=payload).json()
        assert response["completion_status"] == "incomplete"
        assert "duplicate_tool_call_id" in response["completion_reason"]
        assert target.read_text() == "user edit\n"
        assert response["tool_calls"] == []
        assert client.post("/chat", json=payload).json() == response
        assert len(provider.calls) == 1
    with TestClient(make_app(tmp_path, ScriptedProvider())) as client:
        review = client.get(f"/sessions/{sid}/transcript-review", params={"user_id": "dev"}).json()
        assert review["status"] == "valid"
        assert review["issue_count"] == 0
        assert review["read_only"] is True


def test_live_invalid_history_is_blocked_without_changing_archive_or_transcript(tmp_path: Path):
    provider = ScriptedProvider()
    app = make_app(tmp_path, provider)
    with TestClient(app) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        session = app.state.components.sessions[sid]
        session.messages.append(
            ChatMessage(role="assistant", tool_calls=[ToolCall(id="pending", name="write_file")])
        )
        before = session.model_dump()
        archive = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        response = client.post(
            "/chat",
            json={
                "user_id": "dev",
                "session_id": sid,
                "message": "continue",
                "request_id": "guard",
            },
        ).json()
        assert response["completion_status"] == "blocked"
        assert "missing_tool_result" in response["completion_reason"]
        assert session.model_dump() == before
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archive
        assert client.get("/runs", params={"user_id": "dev"}).json()[0]["response"] == response
        # Review reads the committed archive rather than the mutable live transcript.
        review = client.get(f"/sessions/{sid}/transcript-review", params={"user_id": "dev"}).json()
        assert review["status"] == "valid"
        assert review["message_count"] == 0
        assert provider.calls == []
