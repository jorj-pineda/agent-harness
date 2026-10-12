import sqlite3

import pytest

from harness.checkpoints import capture_checkpoint
from harness.runs import RunInput
from harness.session_store import SessionStore
from harness.state import Session


@pytest.mark.parametrize("table", ["runs", "checkpoint_files"])
def test_checkpoint_binding_and_blobs_roll_back_together(tmp_path, table):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a").write_bytes(b"user edit")
    session = Session(user_id="dev", workspace_root=str(root))
    store = SessionStore(tmp_path / "sessions.db")
    store.save(session)
    run, _ = store.create_run(
        RunInput(
            user_id="dev", session_id=session.session_id, message="Inspect", request_id="once"
        ),
        provider="scripted",
        configured_model=None,
    )
    captured = capture_checkpoint(root, session.session_id)
    with sqlite3.connect(store.path) as db:
        operation = "UPDATE" if table == "runs" else "INSERT"
        db.execute(
            f"CREATE TRIGGER deny_binding BEFORE {operation} ON {table} "
            "BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        store.bind_run_checkpoint(run.run_id, captured, "before")
    assert store.get_run(run.run_id).before_checkpoint_id is None
    assert store.get_checkpoint(captured.record.checkpoint_id) is None
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM checkpoint_files").fetchone()[0] == 0
    store.close()


def test_checkpoint_phases_are_ordered_immutable_and_terminal_gated(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    session = Session(user_id="dev", workspace_root=str(root))
    store = SessionStore(tmp_path / "sessions.db")
    store.save(session)
    run, _ = store.create_run(
        RunInput(
            user_id="dev", session_id=session.session_id, message="Inspect", request_id="once"
        ),
        provider="scripted",
        configured_model=None,
    )
    captured = capture_checkpoint(root, session.session_id)
    with pytest.raises(ValueError, match="out of order"):
        store.bind_run_checkpoint(run.run_id, captured, "after")
    store.bind_run_checkpoint(run.run_id, captured, "before")
    with pytest.raises(ValueError, match="already bound"):
        store.bind_run_checkpoint(run.run_id, captured, "before")
    after = capture_checkpoint(root, session.session_id)
    store.bind_run_checkpoint(run.run_id, after, "after")
    assert (
        store.get_checkpoint(after.record.checkpoint_id).paired_before_id
        == captured.record.checkpoint_id
    )
    store.fail_run(run.run_id)
    with pytest.raises(ValueError, match="running run"):
        store.bind_run_checkpoint(run.run_id, capture_checkpoint(root, session.session_id), "after")
    store.close()
