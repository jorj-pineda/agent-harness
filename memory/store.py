"""Long-term personalization fact store.

SQLite-backed, per-user keyed. The agent writes via the `remember_fact`
tool and reads via `recall_facts` (both wired in `tools/memory.py`); the
API layer will also read for automatic system-prompt injection.

Design choices worth calling out:

* **Separate DB file from the support schema.** The SQL tool runs on a
  strictly read-only connection; memory is inherently read-write. Keeping
  them in different files means one accidental write path can't taint
  the other.
* **Exact-string dedupe per user.** `UNIQUE(user_id, fact)` at the schema
  level plus `INSERT OR IGNORE` at the call site. The agent is free to
  "remember" the same thing twice; the store just no-ops.
* **Generated timestamps in Python, not SQL.** ISO-8601 UTC strings match
  the format already in `data/schema.sql` and stay inspectable in tests.

No embeddings yet — chronological recall is enough for the current eval
scenarios. When semantic recall is needed, add an embedding column and a
nearest-neighbor helper without changing the public surface.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from pydantic import BaseModel, Field

from data.sqlite import init_schema, open_rw

log = logging.getLogger(__name__)

MEMORY_SCHEMA_PATH = Path(__file__).parent.parent / "data" / "memory_schema.sql"
DEFAULT_LIST_LIMIT = 20
FACTS_HEADING = "Known facts about the user:"


class Fact(BaseModel):
    """One persisted personalization fact about a user."""

    id: int
    user_id: str
    fact: str
    source_turn_id: str | None = None
    created_at: str = Field(..., description="ISO-8601 UTC timestamp")


class FactStore:
    """SQLite-backed long-term memory for user-scoped facts.

    Holds a single read-write connection for the lifetime of the store;
    callers should `close()` (or use as a context manager) on shutdown.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._conn: sqlite3.Connection = open_rw(self._path)
        init_schema(self._conn, MEMORY_SCHEMA_PATH)

    def add(
        self,
        user_id: str,
        fact: str,
        source_turn_id: str | None = None,
    ) -> bool:
        """Insert a fact. Returns True if new, False if already recorded."""
        if not user_id:
            raise ValueError("user_id must not be empty")
        normalized = fact.strip()
        if not normalized:
            raise ValueError("fact must not be empty")

        now = datetime.now(UTC).isoformat()
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO memory_facts "
            "(user_id, fact, source_turn_id, created_at) VALUES (?, ?, ?, ?)",
            (user_id, normalized, source_turn_id, now),
        )
        self._conn.commit()
        inserted = cursor.rowcount > 0
        log.info(
            "fact_store=add user_id=%s inserted=%s source_turn_id=%s",
            user_id,
            inserted,
            source_turn_id,
        )
        return inserted

    def list(self, user_id: str, limit: int = DEFAULT_LIST_LIMIT) -> list[Fact]:
        """Return a user's facts, most recently added first."""
        if not user_id:
            raise ValueError("user_id must not be empty")
        if limit < 1:
            raise ValueError("limit must be >= 1")
        rows = self._conn.execute(
            "SELECT id, user_id, fact, source_turn_id, created_at "
            "FROM memory_facts WHERE user_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [Fact(**dict(r)) for r in rows]

    def format_for_system_prompt(
        self,
        user_id: str,
        limit: int = DEFAULT_LIST_LIMIT,
    ) -> str:
        """Render a user's facts as a system-prompt injection block.

        Returns "" when the user has no facts so the API layer can
        concatenate unconditionally. Facts are listed most-recent first,
        matching `list()`.
        """
        facts = self.list(user_id, limit=limit)
        if not facts:
            return ""
        lines = [FACTS_HEADING]
        lines.extend(f"- {f.fact}" for f in facts)
        return "\n".join(lines)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> FactStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
