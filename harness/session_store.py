"""SQLite review archives and durable submission identity.

Archives are not executable checkpoints. Interrupted turns and live events are
not persisted in archives; run records retain abandonment status, not partial tools.
Readers see the last committed snapshot rather than mutable execution state.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .runs import RequestConflictError, RunInput, RunRecord
from .state import Session, TurnResponse


class SavedTurnResponse(BaseModel):
    turn_id: str
    configured_model: str | None = None
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


class SessionStore:
    """Single-thread, application-lifecycle-owned archive connection."""

    def __init__(self, path: Path) -> None:
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
        self._conn.commit()

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

    def _archive_payload(
        self,
        session: Session,
        response: TurnResponse | None,
        *,
        configured_model: str | None,
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

    def create_run(
        self,
        request: RunInput,
        *,
        provider: str,
        configured_model: str | None,
    ) -> tuple[RunRecord, bool]:
        record = RunRecord(request=request, provider=provider, configured_model=configured_model)
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

    def finish_run(
        self,
        run_id: str,
        response: TurnResponse,
        *,
        session: Session | None = None,
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
            )
        record.status = "cancelled" if response.completion_status == "cancelled" else "finished"
        record.response = response
        record.finished_at = datetime.now(UTC)
        with self._conn:
            if session is not None and payload is not None:
                self._write_archive(session, payload)
            self._update_run(record)

    def fail_run(self, run_id: str) -> None:
        record = self.get_run(run_id)
        if record is None or record.status != "running":
            return
        record.status = "failed"
        record.error = (
            "Execution or final storage failed; workspace edits may exist. Review before new work."
        )
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
        limit: int = 50,
        offset: int = 0,
    ) -> list[RunRecord]:
        rows = self._conn.execute(
            "SELECT record FROM runs WHERE user_id = ? AND (? IS NULL OR session_id = ?) "
            "ORDER BY created_at DESC, run_id DESC LIMIT ? OFFSET ?",
            (user_id, session_id, session_id, limit, offset),
        ).fetchall()
        return [RunRecord.model_validate_json(row[0]) for row in rows]

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
