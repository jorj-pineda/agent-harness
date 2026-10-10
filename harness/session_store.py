"""SQLite review archives, saved only at session creation and finalized turns.

Archives are not executable checkpoints. Interrupted turns and live events are
not persisted; readers see the last committed snapshot rather than mutable state.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

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
        self._conn.commit()

    def save(
        self,
        session: Session,
        response: TurnResponse | None = None,
        *,
        configured_model: str | None = None,
    ) -> None:
        if not session.user_id:
            raise ValueError("Archived sessions require a user_id")
        if any(turn.finished_at is None for turn in session.turns):
            raise ValueError("Unfinished turns cannot be archived")
        previous = self.get(session.session_id)
        responses = previous.responses if previous else []
        if response is not None:
            if not session.turns or session.turns[-1].finished_at is None:
                raise ValueError("Only finalized turns can be archived")
            turn_id = session.turns[-1].turn_id
            if any(item.turn_id == turn_id for item in responses):
                raise ValueError("Turn response already archived")
            responses.append(
                SavedTurnResponse(
                    turn_id=turn_id, configured_model=configured_model, response=response
                )
            )
        archive = SessionArchive(session=session, responses=responses)
        payload = archive.model_dump_json()
        with self._conn:
            self._conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET archive=excluded.archive",
                (session.session_id, session.user_id, session.created_at.isoformat(), payload),
            )

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
