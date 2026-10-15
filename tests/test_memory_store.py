from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from memory import FACTS_HEADING, Fact, FactStore


@pytest.fixture
def store(tmp_path: Path) -> FactStore:
    return FactStore(tmp_path / "memory.db")


def test_add_returns_true_for_new_fact_and_list_reflects_it(store: FactStore) -> None:
    inserted = store.add("alice", "prefers vegan options")

    assert inserted is True
    facts = store.list("alice")
    assert len(facts) == 1
    assert facts[0].fact == "prefers vegan options"
    assert facts[0].user_id == "alice"


def test_adding_same_user_fact_twice_dedupes_and_returns_false(store: FactStore) -> None:
    assert store.add("alice", "size 9 shoe") is True
    assert store.add("alice", "size 9 shoe") is False

    assert len(store.list("alice")) == 1


def test_identical_fact_under_different_users_is_not_a_duplicate(store: FactStore) -> None:
    assert store.add("alice", "lives in Austin") is True
    assert store.add("bob", "lives in Austin") is True

    assert [f.user_id for f in store.list("alice")] == ["alice"]
    assert [f.user_id for f in store.list("bob")] == ["bob"]


def test_list_returns_most_recent_first(store: FactStore) -> None:
    store.add("alice", "first")
    store.add("alice", "second")
    store.add("alice", "third")

    assert [f.fact for f in store.list("alice")] == ["third", "second", "first"]


def test_list_limit_caps_rows(store: FactStore) -> None:
    for i in range(5):
        store.add("alice", f"fact-{i}")

    facts = store.list("alice", limit=2)

    assert len(facts) == 2
    assert [f.fact for f in facts] == ["fact-4", "fact-3"]


def test_list_for_unknown_user_returns_empty_list(store: FactStore) -> None:
    store.add("alice", "something")

    assert store.list("ghost") == []


def test_source_turn_id_is_persisted(store: FactStore) -> None:
    store.add("alice", "left-handed", source_turn_id="turn-42")

    (fact,) = store.list("alice")
    assert fact.source_turn_id == "turn-42"


def test_source_turn_id_defaults_to_none(store: FactStore) -> None:
    store.add("alice", "no context")

    (fact,) = store.list("alice")
    assert fact.source_turn_id is None


def test_fact_is_whitespace_stripped_before_storage(store: FactStore) -> None:
    store.add("alice", "  has a cat  ")

    (fact,) = store.list("alice")
    assert fact.fact == "has a cat"


def test_whitespace_normalization_drives_dedupe(store: FactStore) -> None:
    assert store.add("alice", "has a cat") is True
    assert store.add("alice", "  has a cat  ") is False


def test_empty_user_id_is_rejected_on_add(store: FactStore) -> None:
    with pytest.raises(ValueError, match="user_id"):
        store.add("", "something")


def test_empty_fact_is_rejected_on_add(store: FactStore) -> None:
    with pytest.raises(ValueError, match="fact"):
        store.add("alice", "")


def test_whitespace_only_fact_is_rejected(store: FactStore) -> None:
    with pytest.raises(ValueError, match="fact"):
        store.add("alice", "   \t\n  ")


def test_empty_user_id_is_rejected_on_list(store: FactStore) -> None:
    with pytest.raises(ValueError, match="user_id"):
        store.list("")


def test_non_positive_limit_is_rejected(store: FactStore) -> None:
    with pytest.raises(ValueError, match="limit"):
        store.list("alice", limit=0)
    with pytest.raises(ValueError, match="limit"):
        store.list("alice", limit=-1)


def test_created_at_is_iso_utc_parseable(store: FactStore) -> None:
    store.add("alice", "timestamp check")

    (fact,) = store.list("alice")
    parsed = datetime.fromisoformat(fact.created_at)
    assert parsed.utcoffset() is not None  # tz-aware


def test_facts_persist_across_store_reopens(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    with FactStore(db_path) as store:
        store.add("alice", "persistent")

    reopened = FactStore(db_path)
    try:
        (fact,) = reopened.list("alice")
        assert fact.fact == "persistent"
    finally:
        reopened.close()


def test_context_manager_closes_connection(tmp_path: Path) -> None:
    import sqlite3

    with FactStore(tmp_path / "memory.db") as store:
        store.add("alice", "x")

    with pytest.raises(sqlite3.ProgrammingError):
        store.list("alice")


def test_fact_model_is_json_serializable(store: FactStore) -> None:
    store.add("alice", "serializable", source_turn_id="t1")

    (fact,) = store.list("alice")
    parsed = Fact.model_validate_json(fact.model_dump_json())
    assert parsed == fact


# ─── format_for_system_prompt ──────────────────────────────────────────────


def test_format_for_system_prompt_returns_empty_when_user_has_no_facts(
    store: FactStore,
) -> None:
    assert store.format_for_system_prompt("alice") == ""


def test_format_for_system_prompt_renders_heading_and_bulleted_facts(
    store: FactStore,
) -> None:
    store.add("alice", "fact one")
    store.add("alice", "fact two")

    block = store.format_for_system_prompt("alice")

    assert block == f"{FACTS_HEADING}\n- fact two\n- fact one"


def test_format_for_system_prompt_lists_most_recent_first(store: FactStore) -> None:
    for fact in ("oldest", "middle", "newest"):
        store.add("alice", fact)

    block = store.format_for_system_prompt("alice")

    lines = block.splitlines()
    assert lines[0] == FACTS_HEADING
    assert lines[1:] == ["- newest", "- middle", "- oldest"]


def test_format_for_system_prompt_respects_limit(store: FactStore) -> None:
    for i in range(5):
        store.add("alice", f"fact-{i}")

    block = store.format_for_system_prompt("alice", limit=2)

    assert block.splitlines()[1:] == ["- fact-4", "- fact-3"]


def test_format_for_system_prompt_isolates_by_user_id(store: FactStore) -> None:
    store.add("alice", "alice-fact")
    store.add("bob", "bob-fact")

    assert store.format_for_system_prompt("alice") == f"{FACTS_HEADING}\n- alice-fact"
    assert store.format_for_system_prompt("bob") == f"{FACTS_HEADING}\n- bob-fact"


def test_engineering_note_persists_for_cross_session_prompt_injection(store: FactStore) -> None:
    store.add("dev-1", "prefer async handlers")

    block = store.format_for_system_prompt("dev-1")

    assert FACTS_HEADING in block
    assert "prefer async handlers" in block


def test_format_for_system_prompt_rejects_empty_user_id(store: FactStore) -> None:
    with pytest.raises(ValueError, match="user_id"):
        store.format_for_system_prompt("")


def test_format_for_system_prompt_rejects_non_positive_limit(store: FactStore) -> None:
    with pytest.raises(ValueError, match="limit"):
        store.format_for_system_prompt("alice", limit=0)


@pytest.mark.parametrize("note", ["x" * 4097, "é" * 2049])
def test_oversized_note_rejected_without_mutation(store: FactStore, note: str) -> None:
    store.add("alice", "keep this")
    with pytest.raises(ValueError, match="4096 UTF-8 bytes"):
        store.add("alice", note)
    assert [f.fact for f in store.list("alice")] == ["keep this"]


def test_note_byte_boundary_applies_after_trimming(store: FactStore) -> None:
    note = "é" * 2048
    assert store.add("alice", f"  {note}\n")
    assert store.list("alice")[0].fact == note


def test_legacy_oversized_note_remains_readable(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "memory.db"
    with FactStore(path):
        pass
    note = "x" * 4097
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO memory_facts (user_id, fact, created_at) VALUES (?, ?, ?)",
            ("alice", note, "2026-10-14"),
        )
    with FactStore(path) as reopened:
        assert reopened.list("alice")[0].fact == note
        with pytest.raises(ValueError, match="4096 UTF-8 bytes"):
            reopened.add("alice", note)
        assert reopened.list("alice")[0].fact == note


def test_memory_prompt_bounded_whole_utf8_notes_with_notice(store: FactStore) -> None:
    from memory.store import MAX_MEMORY_PROMPT_BYTES, MEMORY_OMISSION_NOTICE

    notes = [f"{i}:" + "é" * 2000 for i in range(20)]
    for note in notes:
        store.add("alice", note)
    block = store.format_for_system_prompt("alice")
    assert len(block.encode("utf-8")) <= MAX_MEMORY_PROMPT_BYTES
    assert block.splitlines()[1:-1] == [f"- {notes[-1]}", f"- {notes[-2]}"]
    assert block.endswith(MEMORY_OMISSION_NOTICE)
    assert [f.fact for f in store.list("alice")] == list(reversed(notes))


def test_memory_prompt_exact_byte_boundary_needs_no_notice(store: FactStore) -> None:
    from memory.store import MAX_MEMORY_PROMPT_BYTES, MEMORY_OMISSION_NOTICE

    first = "a" * 4096
    second = "b" * (MAX_MEMORY_PROMPT_BYTES - len(FACTS_HEADING.encode()) - 6 - len(first))
    store.add("alice", first)
    store.add("alice", second)
    block = store.format_for_system_prompt("alice")
    assert len(block.encode()) == MAX_MEMORY_PROMPT_BYTES
    assert block == f"{FACTS_HEADING}\n- {second}\n- {first}"
    assert MEMORY_OMISSION_NOTICE not in block


def test_memory_prompt_skips_legacy_large_note_and_retains_smaller_older_note(
    tmp_path: Path,
) -> None:
    import sqlite3

    from memory.store import MAX_MEMORY_PROMPT_BYTES, MEMORY_OMISSION_NOTICE

    path = tmp_path / "memory.db"
    with FactStore(path) as store:
        store.add("alice", "small older note")
    huge = "old large note " * 2000
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO memory_facts (user_id, fact, created_at) VALUES (?, ?, ?)",
            ("alice", huge, "2026-10-14"),
        )
    with FactStore(path) as reopened:
        block = reopened.format_for_system_prompt("alice")
        assert block == f"{FACTS_HEADING}\n- small older note\n{MEMORY_OMISSION_NOTICE}"
        assert len(block.encode()) <= MAX_MEMORY_PROMPT_BYTES
        assert reopened.list("alice")[0].fact == huge
        assert reopened.format_for_system_prompt("alice", limit=1) == (
            f"{FACTS_HEADING}\n{MEMORY_OMISSION_NOTICE}"
        )


def test_delete_requires_owner_and_reviewed_hash_and_survives_reopen(tmp_path: Path) -> None:
    import hashlib

    from memory.store import FactConflictError

    path = tmp_path / "memory.db"
    with FactStore(path) as store:
        store.add("alice", "delete me")
        store.add("alice", "keep")
        note = store.list("alice")[-1]
        digest = hashlib.sha256(note.fact.encode()).hexdigest()
        assert not store.delete("bob", note.id, digest)
        with pytest.raises(FactConflictError):
            store.delete("alice", note.id, "0" * 64)
        assert store.delete("alice", note.id, digest)
        assert not store.delete("alice", note.id, digest)
    with FactStore(path) as store:
        assert [f.fact for f in store.list("alice")] == ["keep"]


def test_delete_final_content_predicate_preserves_concurrent_change(
    store: FactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import sqlite3

    from memory.store import FactConflictError

    store.add("alice", "original")
    note = store.list("alice")[0]
    expected = hashlib.sha256(note.fact.encode()).hexdigest()
    digest = hashlib.sha256

    def change_after_read(raw: bytes):
        with sqlite3.connect(store.path) as other:
            other.execute("UPDATE memory_facts SET fact=? WHERE id=?", ("later edit", note.id))
        return digest(raw)

    monkeypatch.setattr("memory.store.hashlib.sha256", change_after_read)
    with pytest.raises(FactConflictError, match="changed during deletion"):
        store.delete("alice", note.id, expected)
    assert store.list("alice")[0].fact == "later edit"


def test_delete_reused_id_cannot_delete_different_note(store: FactStore) -> None:
    import hashlib

    from memory.store import FactConflictError

    store.add("alice", "original")
    note = store.list("alice")[0]
    expected = hashlib.sha256(note.fact.encode()).hexdigest()
    assert store.delete("alice", note.id, expected)
    store.add("alice", "replacement")
    assert store.list("alice")[0].id == note.id
    with pytest.raises(FactConflictError):
        store.delete("alice", note.id, expected)
    assert store.list("alice")[0].fact == "replacement"
