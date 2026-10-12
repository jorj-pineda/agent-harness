"""Metadata journal bounds, transactionality, and legacy coverage."""

import sqlite3

import pytest

from harness.config import Settings
from harness.execution_profile import capture_execution_profile
from harness.runs import RunInput
from harness.session_store import SessionStore
from harness.tool_progress import MAX_SAVED_TOOL_EVENTS, SavedToolEvent
from tools import ToolRegistry


def admitted(store):
    record, _ = store.create_run(
        RunInput(user_id="dev", session_id="s", message="Fix", request_id="one"),
        provider="ollama",
        configured_model="gemma4:12b",
    )
    profile = capture_execution_profile(
        Settings(_env_file=None, ollama_model="gemma4:12b"), "ollama", ToolRegistry()
    )
    store.record_execution_profile(record.run_id, profile)
    return record


def test_bounded_prefix_survives_restart_and_cannot_append_to_abandoned_run(tmp_path):
    path = tmp_path / "sessions.db"
    store = SessionStore(path)
    record = admitted(store)
    for sequence in range(1, MAX_SAVED_TOOL_EVENTS + 4):
        store.record_tool_event(
            record.run_id,
            SavedToolEvent(
                sequence=sequence,
                type="tool_start" if sequence % 2 else "tool_end",
                tool="write_file",
                error_observed=False if sequence % 2 == 0 else None,
            ),
        )
    progress = store.get_tool_progress(record.run_id)
    assert len(progress.events) == MAX_SAVED_TOOL_EVENTS
    assert [event.sequence for event in progress.events] == list(
        range(1, MAX_SAVED_TOOL_EVENTS + 1)
    )
    assert progress.observed_events == MAX_SAVED_TOOL_EVENTS + 3
    assert progress.events_truncated
    store.close()
    store = SessionStore(path)
    store.recover_runs()
    assert store.get_run(record.run_id).status == "interrupted"
    assert store.get_tool_progress(record.run_id) == progress
    with pytest.raises(ValueError, match="running run"):
        store.record_tool_event(
            record.run_id, SavedToolEvent(sequence=204, type="tool_start", tool="write_file")
        )
    assert store.get_tool_progress(record.run_id) == progress
    store.close()


def test_journal_initialization_and_profile_binding_are_atomic(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    store._conn.execute(
        "CREATE TRIGGER deny_progress BEFORE INSERT ON tool_progress "
        "BEGIN SELECT RAISE(ABORT, 'fixture storage failure'); END"
    )
    with pytest.raises(sqlite3.IntegrityError):
        admitted(store)
    record = store.find_run("dev", "one")
    assert record.execution_profile is None
    assert store.get_tool_progress(record.run_id).coverage == "not_recorded"
    store.close()


def test_legacy_and_out_of_order_events_do_not_invent_or_replace_evidence(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    record, _ = store.create_run(
        RunInput(user_id="dev", session_id="s", message="Fix", request_id="one"),
        provider="ollama",
        configured_model="gemma4:12b",
    )
    assert store.get_tool_progress(record.run_id).coverage == "not_recorded"
    with pytest.raises(ValueError):
        store.record_tool_event(
            record.run_id, SavedToolEvent(sequence=1, type="tool_start", tool="write_file")
        )
    store.record_execution_profile(
        record.run_id,
        capture_execution_profile(
            Settings(_env_file=None, ollama_model="gemma4:12b"), "ollama", ToolRegistry()
        ),
    )
    store.record_tool_event(
        record.run_id, SavedToolEvent(sequence=1, type="tool_start", tool="write_file")
    )
    before = store.get_tool_progress(record.run_id)
    for sequence in (1, 3):
        with pytest.raises(ValueError, match="sequence"):
            store.record_tool_event(
                record.run_id, SavedToolEvent(sequence=sequence, type="tool_end", tool="write_file")
            )
    assert store.get_tool_progress(record.run_id) == before
    store.close()
