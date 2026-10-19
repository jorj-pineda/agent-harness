"""SQLite review archives and durable submission identity.

Archives are not executable checkpoints. Interrupted turns and live events are
not persisted in archives; a separate journal retains bounded tool metadata.
Readers see the last committed snapshot rather than mutable execution state.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from providers.base import ProviderFailure

from .checkpoints import CapturedCheckpoint, CheckpointRecord, CheckpointSummary
from .execution_profile import ExecutionProfile
from .reverts import FileRevertInput, FileRevertOutcome, FileRevertRecord
from .runs import RequestConflictError, RunInput, RunRecord
from .state import Session, TurnResponse
from .tool_progress import MAX_SAVED_TOOL_EVENTS, SavedToolEvent, ToolProgress
from .workspace_review import WorkspaceReviewBaseline


class SavedTurnResponse(BaseModel):
    turn_id: str
    configured_model: str | None = None
    execution_profile: ExecutionProfile | None = Field(
        default=None,
        description=(
            "Allowlisted configured execution evidence; null when not captured, "
            "including legacy/unadmitted work. "
            "Does not attest backend behavior or grant resume permission."
        ),
    )
    response: TurnResponse


class SessionArchive(BaseModel):
    schema_version: Literal[1] = 1
    session: Session
    responses: list[SavedTurnResponse] = Field(default_factory=list)
    read_only: bool = True


class SessionSummary(BaseModel):
    session_id: str
    workspace_root: str | None
    created_at: datetime
    turn_count: int


class SavedWorkspaceReview(BaseModel):
    turn_id: str
    baseline: WorkspaceReviewBaseline


class SessionStore:
    """Single-thread, application-lifecycle-owned archive connection."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            "session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, "
            "created_at TEXT NOT NULL, archive TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS runs ("
            "run_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, request_id TEXT NOT NULL, "
            "session_id TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, "
            "record TEXT NOT NULL, UNIQUE(user_id, request_id))"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS workspace_reviews ("
            "session_id TEXT PRIMARY KEY, record TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS tool_progress ("
            "run_id TEXT PRIMARY KEY, record TEXT NOT NULL)"
        )
        self._conn.commit()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS file_reverts ("
            "revert_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, request_id TEXT NOT NULL, "
            "record TEXT NOT NULL, UNIQUE(user_id, request_id))"
        )
        self._conn.commit()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints ("
            "checkpoint_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, record TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS checkpoint_files ("
            "checkpoint_id TEXT NOT NULL, path TEXT NOT NULL, content BLOB NOT NULL, "
            "PRIMARY KEY(checkpoint_id, path))"
        )
        self._conn.commit()

    def save_checkpoint(self, captured: CapturedCheckpoint) -> None:
        if captured.record.kind != "manual" or captured.record.run_id is not None:
            raise ValueError("Run checkpoints require atomic run binding.")
        with self._conn:
            self._write_checkpoint(captured)

    def _write_checkpoint(self, captured: CapturedCheckpoint) -> None:
        record = captured.record
        if self.get(record.session_id) is None:
            raise ValueError("Checkpoint requires an archived session.")
        if set(captured.contents) != set(record.files):
            raise ValueError("Checkpoint file manifest mismatch.")
        if record.file_count != len(record.files) or record.byte_count != sum(
            len(raw) for raw in captured.contents.values()
        ):
            raise ValueError("Checkpoint counts mismatch.")
        for path, raw in captured.contents.items():
            saved = record.files[path]
            if saved.size != len(raw) or saved.sha256 != hashlib.sha256(raw).hexdigest():
                raise ValueError("Checkpoint content mismatch.")
        self._conn.execute(
            "INSERT INTO checkpoints VALUES (?, ?, ?)",
            (
                record.checkpoint_id,
                record.session_id,
                record.model_dump_json(),
            ),
        )
        self._conn.executemany(
            "INSERT INTO checkpoint_files VALUES (?, ?, ?)",
            [(record.checkpoint_id, path, raw) for path, raw in captured.contents.items()],
        )

    def bind_run_checkpoint(
        self, run_id: str, captured: CapturedCheckpoint, phase: Literal["before", "after"]
    ) -> None:
        run = self.get_run(run_id)
        if (
            run is None
            or run.status != "running"
            or run.request.session_id != captured.record.session_id
        ):
            raise ValueError("Checkpoint requires a matching running run.")
        field = "before_checkpoint_id" if phase == "before" else "after_checkpoint_id"
        if getattr(run, field) is not None or (
            phase == "after" and run.before_checkpoint_id is None
        ):
            raise ValueError("Checkpoint phase already bound or out of order.")
        record = CheckpointRecord.model_validate(
            {
                **captured.record.model_dump(),
                "kind": f"turn_{phase}",
                "run_id": run_id,
                "paired_before_id": run.before_checkpoint_id if phase == "after" else None,
            }
        )
        if phase == "after":
            before = self.get_checkpoint(run.before_checkpoint_id or "")
            if before is None or (before.root, before.root_identity) != (
                record.root,
                record.root_identity,
            ):
                raise ValueError("Workspace root changed between checkpoints.")
        setattr(run, field, record.checkpoint_id)
        with self._conn:
            self._write_checkpoint(CapturedCheckpoint(record, captured.contents))
            self._update_run(run)

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        row = self._conn.execute(
            "SELECT record FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,)
        ).fetchone()
        return CheckpointRecord.model_validate_json(row[0]) if row else None

    def checkpoint_content(self, record: CheckpointRecord, path: str) -> bytes | None:
        file = record.files.get(path)
        if file is None:
            return None
        row = self._conn.execute(
            "SELECT substr(content, 1, ?) FROM checkpoint_files WHERE checkpoint_id=? AND path=?",
            (file.size + 1, record.checkpoint_id, path),
        ).fetchone()
        raw = row[0] if row else None
        if (
            not isinstance(raw, bytes)
            or len(raw) != file.size
            or hashlib.sha256(raw).hexdigest() != file.sha256
        ):
            raise ValueError("Checkpoint content unavailable or inconsistent.")
        return raw

    def find_file_revert(self, user_id: str, request_id: str) -> FileRevertRecord | None:
        row = self._conn.execute(
            "SELECT record FROM file_reverts WHERE user_id=? AND request_id=?",
            (user_id, request_id),
        ).fetchone()
        return FileRevertRecord.model_validate_json(row[0]) if row else None

    def get_file_revert(self, revert_id: str) -> FileRevertRecord | None:
        row = self._conn.execute(
            "SELECT record FROM file_reverts WHERE revert_id=?", (revert_id,)
        ).fetchone()
        return FileRevertRecord.model_validate_json(row[0]) if row else None

    def create_file_revert(
        self, request: FileRevertInput, before: str, after: str
    ) -> FileRevertRecord:
        record = FileRevertRecord(
            request=request, before_checkpoint_id=before, after_checkpoint_id=after
        )
        with self._conn:
            self._conn.execute(
                "INSERT INTO file_reverts VALUES (?,?,?,?)",
                (record.revert_id, request.user_id, request.request_id, record.model_dump_json()),
            )
        return record

    def finish_file_revert(self, record: FileRevertRecord, outcome: FileRevertOutcome) -> None:
        saved = self.get_file_revert(record.revert_id)
        if saved is None or saved.status != "pending" or saved.request != record.request:
            raise ValueError("Revert must be pending before recording its outcome.")
        record.status = outcome.status
        record.reason = outcome.reason
        record.finished_at = datetime.now(UTC)
        with self._conn:
            self._conn.execute(
                "UPDATE file_reverts SET record=? WHERE revert_id=?",
                (record.model_dump_json(), record.revert_id),
            )

    def recover_file_reverts(self) -> None:
        rows = self._conn.execute("SELECT record FROM file_reverts").fetchall()
        for row in rows:
            record = FileRevertRecord.model_validate_json(row[0])
            if record.status == "pending":
                record.status = "interrupted"
                record.reason = "Outcome unknown after restart; inspect file. No automatic retry."
                record.finished_at = datetime.now(UTC)
                with self._conn:
                    self._conn.execute(
                        "UPDATE file_reverts SET record=? WHERE revert_id=?",
                        (record.model_dump_json(), record.revert_id),
                    )

    def list_checkpoints(
        self, session_id: str, *, limit: int, offset: int
    ) -> list[CheckpointSummary]:
        rows = self._conn.execute(
            "SELECT record FROM checkpoints WHERE session_id = ? "
            "ORDER BY rowid DESC LIMIT ? OFFSET ?",
            (session_id, limit, offset),
        ).fetchall()
        return [CheckpointSummary.model_validate_json(row[0]) for row in rows]

    def save(
        self,
        session: Session,
        response: TurnResponse | None = None,
        *,
        configured_model: str | None = None,
    ) -> None:
        payload = self._archive_payload(session, response, configured_model=configured_model)
        with self._conn:
            self._write_archive(session, payload)

    def _fork_payload(self, source: SessionArchive, session: Session) -> str:
        if (
            session.user_id != source.session.user_id
            or session.session_id == source.session.session_id
            or session.continuation_source is None
            or not session.checkpoint_turns
            or session.continuation_source.session_id != source.session.session_id
            or not source.session.turns
            or session.continuation_source.turn_id != source.session.turns[-1].turn_id
        ):
            raise ValueError("Invalid continuation fork identity.")
        if [turn.turn_id for turn in session.turns] != [
            item.turn_id for item in source.responses
        ] or any(turn.finished_at is None for turn in session.turns):
            raise ValueError("Source archive turn/response alignment is inconsistent.")
        return SessionArchive(session=session, responses=source.responses).model_dump_json()

    def _archive_payload(
        self,
        session: Session,
        response: TurnResponse | None,
        *,
        configured_model: str | None,
        execution_profile: ExecutionProfile | None = None,
    ) -> str:
        if not session.user_id:
            raise ValueError("Archived sessions require a user_id")
        if any(turn.finished_at is None for turn in session.turns):
            raise ValueError("Unfinished turns cannot be archived")
        previous = self.get(session.session_id)
        responses = previous.responses if previous else []
        if response is not None:
            if not session.turns:
                raise ValueError("Only finalized turns can be archived")
            turn_id = session.turns[-1].turn_id
            if any(item.turn_id == turn_id for item in responses):
                raise ValueError("Turn response already archived")
            responses.append(
                SavedTurnResponse(
                    turn_id=turn_id,
                    configured_model=configured_model,
                    execution_profile=execution_profile,
                    response=response,
                )
            )
        return SessionArchive(session=session, responses=responses).model_dump_json()

    def _write_archive(self, session: Session, payload: str) -> None:
        self._conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET archive=excluded.archive",
            (session.session_id, session.user_id, session.created_at.isoformat(), payload),
        )

    def find_run(self, user_id: str, request_id: str) -> RunRecord | None:
        row = self._conn.execute(
            "SELECT record FROM runs WHERE user_id = ? AND request_id = ?",
            (user_id, request_id),
        ).fetchone()
        return RunRecord.model_validate_json(row[0]) if row else None

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self._conn.execute(
            "SELECT record FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return RunRecord.model_validate_json(row[0]) if row else None

    def has_newer_unfinished_work(
        self,
        user_id: str,
        session_id: str,
        after: datetime,
        *,
        excluding_run_id: str,
        workspace_root: str | None = None,
    ) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM runs r WHERE created_at>? AND run_id!=? "
                "AND status IN ('running','failed','interrupted') "
                "AND ((user_id=? AND session_id=?) OR EXISTS ("
                "SELECT 1 FROM sessions s WHERE s.session_id=r.session_id "
                "AND json_extract(s.archive, '$.session.workspace_root')=?)) LIMIT 1",
                (after.isoformat(), excluding_run_id, user_id, session_id, workspace_root),
            ).fetchone()
            is not None
        )

    def create_run(
        self,
        request: RunInput,
        *,
        provider: str,
        configured_model: str | None,
        fork: tuple[SessionArchive, Session] | None = None,
    ) -> tuple[RunRecord, bool]:
        record = RunRecord(request=request, provider=provider, configured_model=configured_model)
        fork_payload = self._fork_payload(*fork) if fork else None
        if fork and (
            fork[1].session_id != request.session_id
            or fork[1].user_id != request.user_id
            or fork[1].continuation_source != request.continuation_source
        ):
            raise ValueError("Fork/run identity mismatch.")
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.run_id,
                        request.user_id,
                        request.request_id,
                        request.session_id,
                        record.status,
                        record.created_at.isoformat(),
                        record.model_dump_json(),
                    ),
                )
                if fork is not None:
                    session = fork[1]
                    self._conn.execute(
                        "INSERT INTO sessions VALUES (?,?,?,?)",
                        (
                            session.session_id,
                            session.user_id,
                            session.created_at.isoformat(),
                            fork_payload,
                        ),
                    )
        except sqlite3.IntegrityError:
            existing = self.find_run(request.user_id, request.request_id)
            if existing is None:
                raise
            if existing.request != request:
                raise RequestConflictError(
                    "request_id already belongs to different input"
                ) from None
            return existing, False
        return record, True

    def _update_run(self, record: RunRecord) -> None:
        self._conn.execute(
            "UPDATE runs SET status = ?, record = ? WHERE run_id = ?",
            (record.status, record.model_dump_json(), record.run_id),
        )

    def record_execution_profile(self, run_id: str, profile: ExecutionProfile) -> None:
        """Bind once before execution; failures prevent inference and tools."""
        record = self.get_run(run_id)
        if record is None or record.status != "running" or record.execution_profile is not None:
            raise ValueError("Execution profile requires an unbound running run")
        if (profile.provider, profile.configured_model) != (
            record.provider,
            record.configured_model,
        ):
            raise ValueError("Execution profile does not match run provider/model identity")
        record.execution_profile = profile
        with self._conn:
            self._update_run(record)
            progress = ToolProgress(
                run_id=run_id, coverage="recorded", updated_at=datetime.now(UTC)
            )
            self._conn.execute(
                "INSERT INTO tool_progress VALUES (?, ?)", (run_id, progress.model_dump_json())
            )

    def get_tool_progress(self, run_id: str) -> ToolProgress:
        row = self._conn.execute(
            "SELECT record FROM tool_progress WHERE run_id = ?", (run_id,)
        ).fetchone()
        return ToolProgress.model_validate_json(row[0]) if row else ToolProgress(run_id=run_id)

    def record_tool_event(self, run_id: str, event: SavedToolEvent) -> None:
        """Commit metadata before onward work; terminal/legacy runs cannot acquire history."""
        record = self.get_run(run_id)
        progress = self.get_tool_progress(run_id)
        if record is None or record.status != "running" or progress.coverage != "recorded":
            raise ValueError("Tool progress requires an admitted running run")
        if event.sequence != progress.observed_events + 1:
            raise ValueError("Tool progress sequence is not next")
        progress.observed_events += 1
        progress.updated_at = event.observed_at
        if len(progress.events) < MAX_SAVED_TOOL_EVENTS:
            progress.events.append(event)
        else:
            progress.events_truncated = True
        with self._conn:
            self._conn.execute(
                "UPDATE tool_progress SET record = ? WHERE run_id = ?",
                (progress.model_dump_json(), run_id),
            )

    def finish_run(
        self,
        run_id: str,
        response: TurnResponse,
        *,
        session: Session | None = None,
        workspace_baseline: WorkspaceReviewBaseline | None = None,
    ) -> None:
        record = self.get_run(run_id)
        if record is None or record.status != "running":
            raise ValueError("Run must exist and be running before finalization")
        if response.run_id != run_id:
            raise ValueError("Response run identity does not match")
        payload = None
        if session is not None:
            if session.session_id != record.request.session_id:
                raise ValueError("Session run identity does not match")
            payload = self._archive_payload(
                session,
                response,
                configured_model=record.configured_model,
                execution_profile=record.execution_profile,
            )
        record.status = "cancelled" if response.completion_status == "cancelled" else "finished"
        record.response = response
        record.finished_at = datetime.now(UTC)
        with self._conn:
            if session is not None and payload is not None:
                self._write_archive(session, payload)
                if workspace_baseline is None:
                    self._conn.execute(
                        "DELETE FROM workspace_reviews WHERE session_id = ?", (session.session_id,)
                    )
                else:
                    saved = SavedWorkspaceReview(
                        turn_id=session.turns[-1].turn_id, baseline=workspace_baseline
                    )
                    self._conn.execute(
                        "INSERT INTO workspace_reviews VALUES (?, ?) "
                        "ON CONFLICT(session_id) DO UPDATE SET record=excluded.record",
                        (session.session_id, saved.model_dump_json()),
                    )
            self._update_run(record)

    def fail_run(self, run_id: str, *, provider_failure: ProviderFailure | None = None) -> None:
        record = self.get_run(run_id)
        if record is None or record.status != "running":
            return
        record.status = "failed"
        record.error = (
            "Execution or final storage failed; workspace edits may exist. Review before new work."
        )
        record.provider_failure = provider_failure
        if provider_failure is not None:
            status = f" HTTP {provider_failure.http_status}" if provider_failure.http_status else ""
            record.error += f" Provider failure: {provider_failure.kind}{status}."
        record.finished_at = datetime.now(UTC)
        with self._conn:
            self._update_run(record)

    def recover_runs(self) -> None:
        """One API process only: mark abandoned records, never dispatch work."""
        rows = self._conn.execute("SELECT record FROM runs WHERE status = 'running'").fetchall()
        with self._conn:
            for row in rows:
                record = RunRecord.model_validate_json(row[0])
                record.status = "interrupted"
                record.error = (
                    "Server restarted before a durable final result; workspace edits may exist."
                )
                record.finished_at = datetime.now(UTC)
                self._update_run(record)

    def list_runs(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RunRecord]:
        rows = self._conn.execute(
            "SELECT record FROM runs WHERE user_id = ? AND (? IS NULL OR session_id = ?) "
            "AND (? IS NULL OR request_id = ?) "
            "ORDER BY created_at DESC, run_id DESC LIMIT ? OFFSET ?",
            (user_id, session_id, session_id, request_id, request_id, limit, offset),
        ).fetchall()
        return [RunRecord.model_validate_json(row[0]) for row in rows]

    def get_workspace_review(self, session_id: str) -> SavedWorkspaceReview | None:
        row = self._conn.execute(
            "SELECT record FROM workspace_reviews WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return SavedWorkspaceReview.model_validate_json(row[0]) if row else None

    def get(self, session_id: str) -> SessionArchive | None:
        row = self._conn.execute(
            "SELECT archive FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return SessionArchive.model_validate_json(row[0]) if row else None

    def list(self, user_id: str, *, limit: int = 50, offset: int = 0) -> list[SessionSummary]:
        rows = self._conn.execute(
            "SELECT archive FROM sessions WHERE user_id = ? "
            "ORDER BY created_at DESC, session_id DESC LIMIT ? OFFSET ?",
            (user_id, limit, offset),
        ).fetchall()
        sessions = [SessionArchive.model_validate_json(row[0]).session for row in rows]
        return [
            SessionSummary(
                session_id=s.session_id,
                workspace_root=s.workspace_root,
                created_at=s.created_at,
                turn_count=len(s.turns),
            )
            for s in sessions
        ]

    def close(self) -> None:
        self._conn.close()
