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
