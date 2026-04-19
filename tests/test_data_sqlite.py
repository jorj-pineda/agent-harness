from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from data.sqlite import init_schema, open_ro, open_rw


def _seed_minimal(conn: sqlite3.Connection) -> None:
    init_schema(conn)
    conn.execute(
        "INSERT INTO customers (id, email, name, tier, created_at) VALUES (?, ?, ?, ?, ?)",
        (1, "a@example.com", "Alice", "standard", "2026-04-01T00:00:00Z"),
    )
    conn.commit()


def test_open_rw_creates_parent_dir_and_file(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "missing" / "db.sqlite"
    conn = open_rw(db_path)
    try:
        assert db_path.exists()
    finally:
        conn.close()


def test_open_rw_enables_foreign_keys(tmp_path: Path) -> None:
    conn = open_rw(tmp_path / "fk.db")
    try:
        (fk,) = conn.execute("PRAGMA foreign_keys").fetchone()
        assert fk == 1
    finally:
        conn.close()


def test_init_schema_creates_expected_tables(tmp_path: Path) -> None:
    conn = open_rw(tmp_path / "schema.db")
    try:
        init_schema(conn)
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        names = {r["name"] for r in rows}
        assert {"customers", "products", "orders", "order_items", "tickets"} <= names
    finally:
        conn.close()


def test_open_ro_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_ro(tmp_path / "does_not_exist.db")


def test_open_ro_allows_select(tmp_path: Path) -> None:
    db_path = tmp_path / "ro.db"
    rw = open_rw(db_path)
    try:
        _seed_minimal(rw)
    finally:
        rw.close()

    ro = open_ro(db_path)
    try:
        row = ro.execute("SELECT name FROM customers WHERE id = 1").fetchone()
        assert row["name"] == "Alice"
    finally:
        ro.close()


def test_open_ro_blocks_insert(tmp_path: Path) -> None:
    db_path = tmp_path / "ro_insert.db"
    rw = open_rw(db_path)
    try:
        _seed_minimal(rw)
    finally:
        rw.close()

    ro = open_ro(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            ro.execute(
                "INSERT INTO customers (id, email, name, tier, created_at) "
                "VALUES (2, 'b@example.com', 'Bob', 'standard', '2026-04-01T00:00:00Z')"
            )
    finally:
        ro.close()


def test_open_ro_blocks_update(tmp_path: Path) -> None:
    db_path = tmp_path / "ro_update.db"
    rw = open_rw(db_path)
    try:
        _seed_minimal(rw)
    finally:
        rw.close()

    ro = open_ro(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            ro.execute("UPDATE customers SET name = 'Mallory' WHERE id = 1")
    finally:
        ro.close()


def test_open_ro_blocks_ddl(tmp_path: Path) -> None:
    db_path = tmp_path / "ro_ddl.db"
    rw = open_rw(db_path)
    try:
        _seed_minimal(rw)
    finally:
        rw.close()

    ro = open_ro(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            ro.execute("CREATE TABLE evil (id INTEGER)")
    finally:
        ro.close()


def test_schema_enforces_tier_check(tmp_path: Path) -> None:
    conn = open_rw(tmp_path / "check.db")
    try:
        init_schema(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO customers (id, email, name, tier, created_at) "
                "VALUES (1, 'x@y.z', 'X', 'bogus', '2026-04-01T00:00:00Z')"
            )
    finally:
        conn.close()
