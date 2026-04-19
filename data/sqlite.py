"""SQLite helpers: read-write for seeding, read-only for the SQL tool.

Mutation safety for the SQL tool rests on SQLite's own URI `mode=ro`
open flag — writes are blocked at the OS file layer, not just by a
library-side check that could be bypassed with clever SQL.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def open_rw(path: str | Path) -> sqlite3.Connection:
    """Open a read-write connection; creates the parent dir and file if missing."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def open_ro(path: str | Path) -> sqlite3.Connection:
    """Open a read-only connection — mutations raise sqlite3.OperationalError."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SQLite database not found: {p}")
    # as_posix() keeps the URI portable on Windows (C:/... instead of C:\...).
    uri = f"file:{p.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection, schema_path: str | Path = SCHEMA_PATH) -> None:
    """Execute the schema DDL against a read-write connection."""
    sql = Path(schema_path).read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()
