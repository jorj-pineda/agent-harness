"""Long-term engineering memory store.

SQLite-backed, per-user keyed. The agent writes via `remember_fact` / `remember`
and reads via `recall_facts` / `recall` (wired in `tools/memory.py`); the API
layer reads for automatic system-prompt injection.

Notes capture durable engineering context — stack choices, repo conventions,
review preferences — not ephemeral chat state.

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

import hashlib
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
MAX_FACT_BYTES = 4096
MAX_MEMORY_PROMPT_BYTES = 8192
MEMORY_OMISSION_NOTICE = (
    "[Some notes in the selected recent window were omitted to fit the memory byte limit.]"
)
FACTS_HEADING = "Engineering memory for this developer (persists across sessions):"


def normalize_fact(fact: str) -> str:
    normalized = fact.strip()
    if not normalized:
        raise ValueError("fact must not be empty")
    if len(normalized.encode("utf-8")) > MAX_FACT_BYTES:
        raise ValueError(f"fact exceeds {MAX_FACT_BYTES} UTF-8 bytes after trimming")
    return normalized


class Fact(BaseModel):
    """One persisted engineering note about a user/developer."""

    id: int
    user_id: str
    fact: str
    source_turn_id: str | None = None
    created_at: str = Field(..., description="ISO-8601 UTC timestamp")
    updated_at: str | None = None


class FactConflictError(ValueError):
    """The stored note no longer matches the reviewed content."""


class FactStore:
    """SQLite-backed long-term memory for user-scoped facts.

    Holds a single read-write connection for the lifetime of the store;
    callers should `close()` (or use as a context manager) on shutdown.
    """

    @property
    def path(self) -> Path:
        """Actual local storage path, for excluding harness-owned data from captures."""
        return self._path.resolve()

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._conn: sqlite3.Connection = open_rw(self._path)
        try:
            init_schema(self._conn, MEMORY_SCHEMA_PATH)
            # Serialize this additive migration across concurrent constructors.
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                columns = {
                    row["name"] for row in self._conn.execute("PRAGMA table_info(memory_facts)")
                }
                if "updated_at" not in columns:
                    self._conn.execute("ALTER TABLE memory_facts ADD COLUMN updated_at TEXT")
        except Exception:
            self._conn.close()
            raise

    def add(
        self,
        user_id: str,
        fact: str,
        source_turn_id: str | None = None,
    ) -> bool:
        """Insert a fact. Returns True if new, False if already recorded."""
        if not user_id:
            raise ValueError("user_id must not be empty")
        normalized = normalize_fact(fact)

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

    def list(
        self, user_id: str, limit: int = DEFAULT_LIST_LIMIT, *, before_id: int | None = None
    ) -> list[Fact]:
        """Return a user's facts, most recently added first."""
        if not user_id:
            raise ValueError("user_id must not be empty")
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if before_id is not None and before_id < 1:
            raise ValueError("before_id must be >= 1")
        rows = self._conn.execute(
            "SELECT id, user_id, fact, source_turn_id, created_at, updated_at "
            "FROM memory_facts WHERE user_id = ? "
            "AND (? IS NULL OR id < ?) ORDER BY id DESC LIMIT ?",
            (user_id, before_id, before_id, limit),
        ).fetchall()
        return [Fact(**dict(r)) for r in rows]

    def delete(self, user_id: str, fact_id: int, expected_fact_sha256: str) -> bool:
        """Delete only this owner's exact reviewed content; missing IDs return False.

        The final SQL content predicate detects a concurrent edit after the read.
        No transcript/archive erasure or secure storage deletion is implied.
        """
        if not user_id or fact_id < 1:
            raise ValueError("user_id must not be empty and fact_id must be positive")
        row = self._conn.execute(
            "SELECT fact FROM memory_facts WHERE user_id = ? AND id = ?", (user_id, fact_id)
        ).fetchone()
        if row is None:
            return False
        fact: str = row["fact"]
        if hashlib.sha256(fact.encode("utf-8")).hexdigest() != expected_fact_sha256:
            raise FactConflictError("Stored note differs from the reviewed content; inspect again.")
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM memory_facts WHERE user_id = ? AND id = ? AND fact = ?",
                (user_id, fact_id, fact),
            )
            if cursor.rowcount != 1:
                raise FactConflictError("Stored note changed during deletion; inspect again.")
        return True

    def correct(
        self, user_id: str, fact_id: int, expected_fact_sha256: str, replacement: str
    ) -> Fact | None:
        """Replace reviewed content atomically, preserving creation/ID ordering.

        Owner correction records its time and clears original source attribution.
        Missing/other-owner IDs return None; duplicate-note conflicts roll back.
        """
        if not user_id or fact_id < 1:
            raise ValueError("user_id must not be empty and fact_id must be positive")
        normalized = normalize_fact(replacement)
        row = self._conn.execute(
            "SELECT fact FROM memory_facts WHERE user_id = ? AND id = ?", (user_id, fact_id)
        ).fetchone()
        if row is None:
            return None
        fact: str = row["fact"]
        if hashlib.sha256(fact.encode("utf-8")).hexdigest() != expected_fact_sha256:
            raise FactConflictError("Stored note differs from the reviewed content; inspect again.")
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE memory_facts SET fact = ?, source_turn_id = NULL, updated_at = ? "
                "WHERE user_id = ? AND id = ? AND fact = ?",
                (normalized, datetime.now(UTC).isoformat(), user_id, fact_id, fact),
            )
            if cursor.rowcount != 1:
                raise FactConflictError("Stored note changed during correction; inspect again.")
            result = self._conn.execute(
                "SELECT id, user_id, fact, source_turn_id, created_at, updated_at "
                "FROM memory_facts WHERE user_id = ? AND id = ?",
                (user_id, fact_id),
            ).fetchone()
            return Fact(**dict(result))

    def format_for_system_prompt(
        self,
        user_id: str,
        limit: int = DEFAULT_LIST_LIMIT,
    ) -> str:
        """Render a user's engineering notes as a system-prompt injection block.

        Returns "" when the user has no notes so the API layer can concatenate
        unconditionally. Notes are listed most-recent first, matching `list()`. The UTF-8 block
        is bounded; omitted whole notes are announced without changing storage.
        """
        facts = self.list(user_id, limit=limit)
        if not facts:
            return ""
        candidates = [f"- {f.fact}" for f in facts]
        sizes = [len(line.encode("utf-8")) + 1 for line in candidates]
        heading_size = len(FACTS_HEADING.encode("utf-8"))
        if heading_size + sum(sizes) <= MAX_MEMORY_PROMPT_BYTES:
            return "\n".join([FACTS_HEADING, *candidates])

        # Reserve the notice before selecting whole notes. Skipping a large recent
        # note may let smaller older notes within the same selected window fit.
        remaining = (
            MAX_MEMORY_PROMPT_BYTES - heading_size - 1 - len(MEMORY_OMISSION_NOTICE.encode("utf-8"))
        )
        lines = [FACTS_HEADING]
        for line, size in zip(candidates, sizes, strict=True):
            if size <= remaining:
                lines.append(line)
                remaining -= size
        lines.append(MEMORY_OMISSION_NOTICE)
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
