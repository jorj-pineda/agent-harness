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
* **Exact-string dedupe per user and namespace.** Legacy global rows stay
  in their original table; workspace rows dedupe by user/root/text. One
  transactional counter assigns IDs across both without reusing deleted IDs.
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
from typing import Literal

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


_TABLE_SQL = {
    "memory_facts": {
        "find": "SELECT id FROM memory_facts WHERE user_id = ? AND fact = ?",
        "insert": "INSERT INTO memory_facts (id, user_id, fact, source_turn_id, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        "delete": "DELETE FROM memory_facts WHERE user_id = ? AND id = ? AND fact = ?",
        "correct": "UPDATE memory_facts SET fact = ?, source_turn_id = NULL, updated_at = ? "
        "WHERE user_id = ? AND id = ? AND fact = ?",
        "get": "SELECT id, user_id, fact, source_turn_id, created_at, updated_at, "
        "NULL AS workspace_root FROM memory_facts WHERE user_id = ? AND id = ?",
    },
    "workspace_memory_facts": {
        "find": "SELECT id FROM workspace_memory_facts WHERE user_id = ? AND fact = ? "
        "AND workspace_root = ?",
        "insert": "INSERT INTO workspace_memory_facts "
        "(id, user_id, fact, source_turn_id, created_at, workspace_root) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        "delete": "DELETE FROM workspace_memory_facts WHERE user_id = ? AND id = ? AND fact = ?",
        "correct": "UPDATE workspace_memory_facts SET fact = ?, source_turn_id = NULL, "
        "updated_at = ? WHERE user_id = ? AND id = ? AND fact = ?",
        "get": "SELECT id, user_id, fact, source_turn_id, created_at, updated_at, workspace_root "
        "FROM workspace_memory_facts WHERE user_id = ? AND id = ?",
    },
}


def normalize_fact(fact: str) -> str:
    normalized = fact.strip()
    if not normalized:
        raise ValueError("fact must not be empty")
    if len(normalized.encode("utf-8")) > MAX_FACT_BYTES:
        raise ValueError(f"fact exceeds {MAX_FACT_BYTES} UTF-8 bytes after trimming")
    return normalized


def _workspace_key(root: str | None) -> str | None:
    if root is None:
        return None
    path = Path(root)
    if not path.is_absolute() or ".." in path.parts or str(path) != root:
        raise ValueError("workspace_root must be a canonical absolute path")
    return root


class Fact(BaseModel):
    """One persisted engineering note about a user/developer."""

    id: int
    user_id: str
    fact: str
    source_turn_id: str | None = None
    created_at: str = Field(..., description="ISO-8601 UTC timestamp")
    updated_at: str | None = None
    workspace_root: str | None = None


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
                self._conn.execute(
                    "UPDATE memory_id_counter SET last_id = MAX(last_id, "
                    "COALESCE((SELECT MAX(id) FROM memory_facts), 0), "
                    "COALESCE((SELECT MAX(id) FROM workspace_memory_facts), 0))"
                )
        except Exception:
            self._conn.close()
            raise

    def add(
        self,
        user_id: str,
        fact: str,
        source_turn_id: str | None = None,
        *,
        workspace_root: str | None = None,
    ) -> bool:
        """Insert a fact. Returns True if new, False if already recorded."""
        if not user_id:
            raise ValueError("user_id must not be empty")
        normalized = normalize_fact(fact)

        workspace_root = _workspace_key(workspace_root)
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            sql = _TABLE_SQL["memory_facts" if workspace_root is None else "workspace_memory_facts"]
            existing = self._conn.execute(
                sql["find"],
                (user_id, normalized, workspace_root)
                if workspace_root is not None
                else (user_id, normalized),
            ).fetchone()
            if existing is not None:
                return False
            last_id = self._conn.execute(
                "SELECT MAX(value) FROM (SELECT last_id AS value FROM memory_id_counter "
                "UNION ALL SELECT MAX(id) FROM memory_facts "
                "UNION ALL SELECT MAX(id) FROM workspace_memory_facts)"
            ).fetchone()[0]
            fact_id = int(last_id) + 1
            self._conn.execute("UPDATE memory_id_counter SET last_id = ?", (fact_id,))
            values = (fact_id, user_id, normalized, source_turn_id, datetime.now(UTC).isoformat())
            self._conn.execute(
                sql["insert"],
                (*values, workspace_root) if workspace_root is not None else values,
            )
        inserted = True
        log.info(
            "fact_store=add user_id=%s inserted=%s source_turn_id=%s",
            user_id,
            inserted,
            source_turn_id,
        )
        return inserted

    def list(
        self,
        user_id: str,
        limit: int = DEFAULT_LIST_LIMIT,
        *,
        before_id: int | None = None,
        workspace_root: str | None = None,
        scoped: bool = False,
        namespace: Literal["all", "global", "workspace"] = "all",
    ) -> list[Fact]:
        """Return a user's facts, most recently added first."""
        if not user_id:
            raise ValueError("user_id must not be empty")
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if before_id is not None and before_id < 1:
            raise ValueError("before_id must be >= 1")
        workspace_root = _workspace_key(workspace_root)
        if namespace not in {"all", "global", "workspace"}:
            raise ValueError("invalid memory namespace")
        if namespace == "workspace" and workspace_root is None:
            raise ValueError("workspace namespace requires workspace_root")
        rows = self._conn.execute(
            "SELECT * FROM ("
            "SELECT id, user_id, fact, source_turn_id, created_at, updated_at, "
            "NULL AS workspace_root FROM memory_facts UNION ALL "
            "SELECT id, user_id, fact, source_turn_id, created_at, updated_at, workspace_root "
            "FROM workspace_memory_facts) WHERE user_id = ? "
            "AND (? IS NULL OR id < ?) "
            "AND (? = 0 OR workspace_root IS NULL OR workspace_root = ?) "
            "AND (? = 'all' OR (? = 'global' AND workspace_root IS NULL) "
            "OR (? = 'workspace' AND workspace_root = ?)) "
            "ORDER BY id DESC LIMIT ?",
            (
                user_id,
                before_id,
                before_id,
                scoped,
                workspace_root,
                namespace,
                namespace,
                namespace,
                workspace_root,
                limit,
            ),
        ).fetchall()
        return [Fact(**dict(r)) for r in rows]

    def _lookup(self, user_id: str, fact_id: int) -> tuple[str, str] | None:
        rows = self._conn.execute(
            "SELECT 'memory_facts' AS storage, fact FROM memory_facts WHERE user_id=? AND id=? "
            "UNION ALL SELECT 'workspace_memory_facts', fact FROM workspace_memory_facts "
            "WHERE user_id=? AND id=?",
            (user_id, fact_id, user_id, fact_id),
        ).fetchall()
        if len(rows) > 1:
            raise FactConflictError("Ambiguous memory ID from unsupported writes; inspect storage.")
        return (rows[0]["storage"], rows[0]["fact"]) if rows else None

    def delete(self, user_id: str, fact_id: int, expected_fact_sha256: str) -> bool:
        """Delete only this owner's exact reviewed content; missing IDs return False.

        The final SQL content predicate detects a concurrent edit after the read.
        No transcript/archive erasure or secure storage deletion is implied.
        """
        if not user_id or fact_id < 1:
            raise ValueError("user_id must not be empty and fact_id must be positive")
        found = self._lookup(user_id, fact_id)
        if found is None:
            return False
        table, fact = found
        if hashlib.sha256(fact.encode("utf-8")).hexdigest() != expected_fact_sha256:
            raise FactConflictError("Stored note differs from the reviewed content; inspect again.")
        with self._conn:
            cursor = self._conn.execute(
                _TABLE_SQL[table]["delete"],
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
        found = self._lookup(user_id, fact_id)
        if found is None:
            return None
        table, fact = found
        if hashlib.sha256(fact.encode("utf-8")).hexdigest() != expected_fact_sha256:
            raise FactConflictError("Stored note differs from the reviewed content; inspect again.")
        with self._conn:
            cursor = self._conn.execute(
                _TABLE_SQL[table]["correct"],
                (normalized, datetime.now(UTC).isoformat(), user_id, fact_id, fact),
            )
            if cursor.rowcount != 1:
                raise FactConflictError("Stored note changed during correction; inspect again.")
            result = self._conn.execute(
                _TABLE_SQL[table]["get"],
                (user_id, fact_id),
            ).fetchone()
            return Fact(**dict(result))

    def format_for_system_prompt(
        self,
        user_id: str,
        limit: int = DEFAULT_LIST_LIMIT,
        *,
        workspace_root: str | None = None,
    ) -> str:
        """Render a user's engineering notes as a system-prompt injection block.

        Returns "" when the user has no notes so the API layer can concatenate
        unconditionally. Notes are listed most-recent first, matching `list()`. The UTF-8 block
        is bounded; omitted whole notes are announced without changing storage.
        """
        facts = self.list(user_id, limit=limit, workspace_root=workspace_root, scoped=True)
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
