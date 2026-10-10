from datetime import UTC, datetime
from pathlib import Path

from harness.session_store import SessionStore
from harness.state import (
    CheckRecord,
    Session,
    Turn,
    TurnResponse,
    WorkspaceChangeReport,
    WorkspaceFileDiff,
)
from providers.base import ChatMessage, TokenUsage


def test_archive_roundtrip_retains_observed_evidence_and_identity(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    session = Session(
        user_id="dev",
        messages=[ChatMessage(role="assistant", content="Partial")],
        turns=[Turn(user_input="Fix", final_answer="Partial", finished_at=datetime.now(UTC))],
    )
    response = TurnResponse(
        answer="Partial",
        provider="ollama",
        latency_ms=123,
        confidence=1.0,
        completion_status="incomplete",
        completion_reason="Check unavailable",
        verification_status="not_run",
        check_attempts=[
            CheckRecord(
                argv=["pytest"],
                status="unavailable",
                relevant=True,
                error="Missing executable",
            )
        ],
        tool_errors=["run_command: missing"],
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=20),
        workspace_changes=WorkspaceChangeReport(
            status="tracked",
            modified=["a.py"],
            diffs=[
                WorkspaceFileDiff(path="a.py", reason="Text capture byte limit exceeded"),
            ],
        ),
    )
    store = SessionStore(path)
    store.save(session, response, configured_model="gemma4:12b")
    store.close()
    store = SessionStore(path)
    try:
        saved = store.get(session.session_id)
        assert saved is not None
        assert saved.session == session
        assert saved.responses[0].response == response
        assert saved.responses[0].configured_model == "gemma4:12b"
        assert saved.read_only
    finally:
        store.close()


def test_run_identity_and_final_archive_commit_are_atomic(tmp_path: Path, monkeypatch) -> None:
    import sqlite3

    import pytest

    from harness.runs import RequestConflictError, RunInput

    store = SessionStore(tmp_path / "sessions.db")
    try:
        session = Session(user_id="dev")
        store.save(session)
        request = RunInput(
            user_id="dev", session_id=session.session_id, message="Fix", request_id="one"
        )
        run, created = store.create_run(request, provider="ollama", configured_model="gemma4:12b")
        assert created
        assert store.create_run(request, provider="changed", configured_model=None) == (run, False)
        with pytest.raises(RequestConflictError):
            store.create_run(
                request.model_copy(update={"message": "Other"}),
                provider="ollama",
                configured_model=None,
            )
        session.turns.append(
            Turn(user_input="Fix", final_answer="Done", finished_at=datetime.now(UTC))
        )
        response = TurnResponse(answer="Done", provider="ollama", latency_ms=1, run_id=run.run_id)
        original_update = store._update_run

        def fail(record):
            raise sqlite3.OperationalError("disk full")

        monkeypatch.setattr(store, "_update_run", fail)
        with pytest.raises(sqlite3.OperationalError):
            store.finish_run(run.run_id, response, session=session)
        assert store.get(session.session_id).responses == []
        assert store.get_run(run.run_id).status == "running"
        monkeypatch.setattr(store, "_update_run", original_update)
        store.finish_run(run.run_id, response, session=session)
        store.recover_runs()
        assert store.get_run(run.run_id).response == response
        assert store.get(session.session_id).responses[0].response == response
    finally:
        store.close()
