import os
import sqlite3
from pathlib import Path

import pytest

from harness.checkpoints import capture_checkpoint
from harness.session_store import SessionStore
from harness.state import Session


def test_checkpoint_saves_exact_binary_text_modes_and_excludes_private_entries(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_bytes(b"user edit\n")
    (root / "a.py").chmod(0o755)
    (root / "binary").write_bytes(b"\x00\xff")
    (root / "empty").write_bytes(b"")
    (root / ".env").write_bytes(b"fixture-private-marker")
    (root / ".git").mkdir()
    (root / ".git/index").write_bytes(b"index-marker")
    session = Session(user_id="dev", workspace_root=str(root))
    captured = capture_checkpoint(root, session.session_id)
    assert captured.contents == {"a.py": b"user edit\n", "binary": b"\x00\xff", "empty": b""}
    assert captured.record.files["a.py"].mode == 0o755
    assert captured.record.byte_count == 12
    assert captured.record.restore_supported is False
    store = SessionStore(tmp_path / "sessions.db")
    store.save(session)
    store.save_checkpoint(captured)
    with pytest.raises(sqlite3.IntegrityError):
        store.save_checkpoint(captured)
    store.close()
    restarted = SessionStore(tmp_path / "sessions.db")
    assert restarted.get_checkpoint(captured.record.checkpoint_id) == captured.record
    assert restarted.list_checkpoints(session.session_id, limit=1, offset=0)[0].file_count == 3
    assert restarted.list_checkpoints(session.session_id, limit=1, offset=1) == []
    restarted.close()
    with sqlite3.connect(tmp_path / "sessions.db") as conn:
        rows = dict(conn.execute("SELECT path, content FROM checkpoint_files").fetchall())
    assert rows == captured.contents
    assert (root / ".git/index").read_bytes() == b"index-marker"
    assert (root / ".env").read_bytes() == b"fixture-private-marker"


@pytest.mark.parametrize("kind", ["file_link", "dir_link", "fifo"])
def test_checkpoint_rejects_unsupported_entries(tmp_path: Path, kind: str):
    import os

    root = tmp_path / "repo"
    root.mkdir()
    if kind == "fifo":
        os.mkfifo(root / "unsupported")
    else:
        (root / "unsupported").symlink_to(tmp_path, target_is_directory=kind == "dir_link")
    with pytest.raises((ValueError, OSError)):
        capture_checkpoint(root, "session")


@pytest.mark.parametrize("limit", ["bytes", "entries"])
def test_checkpoint_limits_do_not_produce_partial_records(tmp_path: Path, monkeypatch, limit: str):
    (tmp_path / "a").write_bytes(b"1234")
    (tmp_path / "b").write_bytes(b"5678")
    if limit == "bytes":
        monkeypatch.setattr("harness.checkpoints.MAX_CHECKPOINT_BYTES", 7)
    else:
        monkeypatch.setattr("harness.checkpoints.MAX_CHECKPOINT_ENTRIES", 1)
    with pytest.raises(ValueError, match="limit exceeded"):
        capture_checkpoint(tmp_path, "session")


def test_checkpoint_rejects_harness_storage_inside_capture(tmp_path: Path):
    with pytest.raises(ValueError, match="storage outside"):
        capture_checkpoint(tmp_path, "session", protected_paths=(tmp_path / "sessions.db",))


def test_changed_file_during_read_rejects_capture(tmp_path: Path, monkeypatch):
    target = tmp_path / "a"
    target.write_bytes(b"original")
    real_fstat = os.fstat
    calls = 0

    def changing_fstat(descriptor):
        nonlocal calls
        calls += 1
        observed = real_fstat(descriptor)
        if calls == 1:
            target.write_bytes(b"external change with different size")
        return observed

    monkeypatch.setattr("harness.checkpoints.os.fstat", changing_fstat)
    with pytest.raises(ValueError, match="changed during"):
        capture_checkpoint(tmp_path, "session")
    assert target.read_bytes() == b"external change with different size"


def test_failed_blob_insert_rolls_back_metadata_and_contents(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a").write_bytes(b"original")
    session = Session(user_id="dev", workspace_root=str(root))
    store = SessionStore(tmp_path / "sessions.db")
    store.save(session)
    captured = capture_checkpoint(root, session.session_id)
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "CREATE TRIGGER deny_blob BEFORE INSERT ON checkpoint_files "
            "BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        store.save_checkpoint(captured)
    assert store.get_checkpoint(captured.record.checkpoint_id) is None
    assert store.list_checkpoints(session.session_id, limit=10, offset=0) == []
    assert (root / "a").read_bytes() == b"original"
    store.close()
