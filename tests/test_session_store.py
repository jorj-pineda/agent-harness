from contextlib import closing
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

    from harness.config import Settings
    from harness.execution_profile import capture_execution_profile
    from harness.runs import RequestConflictError, RunInput
    from harness.workspace_review import WorkspaceReviewBaseline
    from tools import ToolRegistry

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
        baseline = WorkspaceReviewBaseline(
            status="recorded",
            root="/fixture",
            root_identity=(1, 2),
            fingerprints={"a.py": "fixture"},
            ignored_entry_fingerprints={".git": "fixture-metadata-hash"},
        )
        profile = capture_execution_profile(
            Settings(_env_file=None, ollama_model="gemma4:12b"), "ollama", ToolRegistry()
        )
        store.record_execution_profile(run.run_id, profile)
        original_update = store._update_run

        def fail(record):
            raise sqlite3.OperationalError("disk full")

        monkeypatch.setattr(store, "_update_run", fail)
        with pytest.raises(sqlite3.OperationalError):
            store.finish_run(run.run_id, response, session=session, workspace_baseline=baseline)
        assert store.get(session.session_id).responses == []
        assert store.get_run(run.run_id).status == "running"
        assert store.get_workspace_review(session.session_id) is None
        assert store.get_run(run.run_id).execution_profile == profile
        monkeypatch.setattr(store, "_update_run", original_update)
        store.finish_run(run.run_id, response, session=session, workspace_baseline=baseline)
        assert store.get_workspace_review(session.session_id).baseline == baseline
        store.recover_runs()
        assert store.get_run(run.run_id).response == response
        assert store.get(session.session_id).responses[0].response == response
        assert store.get(session.session_id).responses[0].execution_profile == profile
    finally:
        store.close()


def test_profile_binding_is_once_only_and_survives_abandonment(tmp_path: Path):
    import pytest

    from harness.config import Settings
    from harness.execution_profile import capture_execution_profile
    from harness.runs import RunInput
    from tools import ToolRegistry

    profile = capture_execution_profile(Settings(_env_file=None), "ollama", ToolRegistry())
    with closing(SessionStore(tmp_path / "sessions.db")) as store:
        record, _ = store.create_run(
            RunInput(user_id="dev", session_id="s1", message="task", request_id="one"),
            provider="ollama",
            configured_model=profile.configured_model,
        )
        with pytest.raises(ValueError, match="provider/model"):
            store.record_execution_profile(
                record.run_id, profile.model_copy(update={"provider": "other"})
            )
        assert store.get_run(record.run_id).execution_profile is None
        store.record_execution_profile(record.run_id, profile)
        with pytest.raises(ValueError, match="unbound"):
            store.record_execution_profile(record.run_id, profile)
    with closing(SessionStore(tmp_path / "sessions.db")) as store:
        store.recover_runs()
        saved = store.get_run(record.run_id)
        assert saved.status == "interrupted"
        assert saved.execution_profile == profile
        with pytest.raises(ValueError, match="unbound"):
            store.record_execution_profile(record.run_id, profile)


def test_profile_storage_rollback_does_not_bind_metadata(tmp_path: Path, monkeypatch):
    import sqlite3

    import pytest

    from harness.config import Settings
    from harness.execution_profile import capture_execution_profile
    from harness.runs import RunInput
    from tools import ToolRegistry

    profile = capture_execution_profile(Settings(_env_file=None), "ollama", ToolRegistry())
    with closing(SessionStore(tmp_path / "sessions.db")) as store:
        record, _ = store.create_run(
            RunInput(user_id="dev", session_id="s1", message="task", request_id="one"),
            provider="ollama",
            configured_model=profile.configured_model,
        )
        original = store._update_run

        def fail_after_update(record):
            original(record)
            raise sqlite3.OperationalError("fixture disk full")

        monkeypatch.setattr(store, "_update_run", fail_after_update)
        with pytest.raises(sqlite3.OperationalError):
            store.record_execution_profile(record.run_id, profile)
        assert store.get_run(record.run_id).execution_profile is None


def test_legacy_payloads_have_unknown_profiles(tmp_path: Path):
    import json

    from harness.runs import RunInput, RunRecord
    from harness.session_store import SessionArchive

    run = RunRecord(
        request=RunInput(user_id="dev", session_id="s1", message="task", request_id="one"),
        provider="ollama",
    )
    assert (
        RunRecord.model_validate_json(
            run.model_dump_json(exclude={"execution_profile"})
        ).execution_profile
        is None
    )
    archive = SessionArchive(session=Session(user_id="dev"))
    payload = json.loads(archive.model_dump_json())
    payload["responses"] = [
        {"turn_id": "t1", "response": {"answer": "done", "provider": "ollama", "latency_ms": 1}}
    ]
    with closing(SessionStore(tmp_path / "sessions.db")) as store:
        store._write_archive(archive.session, json.dumps(payload))
        store._conn.commit()
        assert store.get(archive.session.session_id).responses[0].execution_profile is None
