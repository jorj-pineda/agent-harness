from __future__ import annotations

from pathlib import Path

import pytest

from memory import FactStore
from tools import ToolError, ToolRegistry
from tools.memory import build_memory_tools, register_memory_tools


@pytest.fixture
def store(tmp_path: Path) -> FactStore:
    return FactStore(tmp_path / "memory.db")


@pytest.fixture
def registry(store: FactStore) -> ToolRegistry:
    reg = ToolRegistry()
    register_memory_tools(reg, store=store, user_id="alice")
    return reg


async def test_remember_fact_stores_new_fact_and_reports_inserted(
    registry: ToolRegistry, store: FactStore
) -> None:
    result = await registry.invoke("remember_fact", {"fact": "prefers vegan options"})

    assert result == {"stored": True, "fact": "prefers vegan options"}
    assert [f.fact for f in store.list("alice")] == ["prefers vegan options"]


async def test_remember_fact_duplicate_reports_not_stored(registry: ToolRegistry) -> None:
    first = await registry.invoke("remember_fact", {"fact": "size 9 shoe"})
    second = await registry.invoke("remember_fact", {"fact": "size 9 shoe"})

    assert first["stored"] is True
    assert second["stored"] is False


async def test_remember_fact_strips_whitespace_in_return_value(registry: ToolRegistry) -> None:
    result = await registry.invoke("remember_fact", {"fact": "  has a cat  "})

    assert result == {"stored": True, "fact": "has a cat"}


async def test_recall_facts_returns_most_recent_first(registry: ToolRegistry) -> None:
    for fact in ("one", "two", "three"):
        await registry.invoke("remember_fact", {"fact": fact})

    recalled = await registry.invoke("recall_facts", {})

    assert recalled == ["three", "two", "one"]


async def test_recall_facts_empty_for_fresh_user(registry: ToolRegistry) -> None:
    assert await registry.invoke("recall_facts", {}) == []


async def test_recall_facts_respects_max_results(registry: ToolRegistry) -> None:
    for i in range(5):
        await registry.invoke("remember_fact", {"fact": f"fact-{i}"})

    recalled = await registry.invoke("recall_facts", {"max_results": 2})

    assert recalled == ["fact-4", "fact-3"]


async def test_remember_fact_rejects_empty_fact(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError):
        await registry.invoke("remember_fact", {"fact": ""})


async def test_remember_fact_rejects_whitespace_only_fact(registry: ToolRegistry) -> None:
    # Passes pydantic min_length=1 but the store rejects after strip().
    with pytest.raises(ToolError):
        await registry.invoke("remember_fact", {"fact": "   \t\n  "})


async def test_recall_facts_rejects_non_positive_max_results(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError):
        await registry.invoke("recall_facts", {"max_results": 0})


async def test_factory_rejects_empty_user_id(store: FactStore) -> None:
    with pytest.raises(ValueError, match="user_id"):
        build_memory_tools(store, "")


async def test_build_memory_tools_isolates_by_user_id(store: FactStore) -> None:
    alice_reg = ToolRegistry()
    for t in build_memory_tools(store, "alice"):
        alice_reg.register(t)

    bob_reg = ToolRegistry()
    for t in build_memory_tools(store, "bob"):
        bob_reg.register(t)

    await alice_reg.invoke("remember_fact", {"fact": "alice-only"})
    await bob_reg.invoke("remember_fact", {"fact": "bob-only"})

    assert await alice_reg.invoke("recall_facts", {}) == ["alice-only"]
    assert await bob_reg.invoke("recall_facts", {}) == ["bob-only"]


async def test_tools_expose_specs_for_provider_layer(registry: ToolRegistry) -> None:
    names = {s.name for s in registry.as_tool_specs()}

    assert {"remember_fact", "remember", "recall_facts", "recall"} <= names


async def test_remember_alias_stores_engineering_note(registry: ToolRegistry) -> None:
    result = await registry.invoke("remember", {"note": "prefer async handlers"})

    assert result == {"stored": True, "fact": "prefer async handlers"}
    recalled = await registry.invoke("recall", {})
    assert recalled == ["prefer async handlers"]


async def test_cross_session_engineering_note_surfaces_via_recall(
    store: FactStore,
) -> None:
    """Session A writes via remember; session B (same user_id) reads via recall."""
    session_a = ToolRegistry()
    for tool in build_memory_tools(store, "dev-1"):
        session_a.register(tool)

    await session_a.invoke("remember", {"note": "prefer async handlers"})

    session_b = ToolRegistry()
    for tool in build_memory_tools(store, "dev-1"):
        session_b.register(tool)

    recalled = await session_b.invoke("recall_facts", {})
    assert recalled == ["prefer async handlers"]
    assert "prefer async" in store.format_for_system_prompt("dev-1")


async def test_register_memory_tools_rejects_double_registration(store: FactStore) -> None:
    reg = ToolRegistry()
    register_memory_tools(reg, store=store, user_id="alice")

    with pytest.raises(ToolError, match="already registered"):
        register_memory_tools(reg, store=store, user_id="alice")


@pytest.mark.parametrize("name,key", [("remember", "note"), ("remember_fact", "fact")])
async def test_remember_aliases_reject_oversized_utf8_notes(
    registry: ToolRegistry, store: FactStore, name: str, key: str
) -> None:
    with pytest.raises(ToolError, match="4096 UTF-8 bytes"):
        await registry.invoke(name, {key: "é" * 2049})
    assert store.list("alice") == []
    assert await registry.invoke(name, {key: "short note"}) == {
        "stored": True,
        "fact": "short note",
    }


@pytest.mark.parametrize("name", ["recall", "recall_facts"])
async def test_recall_exact_json_boundary_and_overflow_recovery(
    registry: ToolRegistry, store: FactStore, name: str
) -> None:
    from harness.loop import _encode_tool_result
    from tools.memory import MAX_RECALL_BYTES

    notes = [chr(97 + i) * 4096 for i in range(7)]
    notes.append("z" * (MAX_RECALL_BYTES - 7 * 4096 - 4 * 8))
    for note in notes:
        store.add("alice", note)
    result = await registry.invoke(name, {})
    assert result == list(reversed(notes))
    assert len(_encode_tool_result(result).encode()) == MAX_RECALL_BYTES
    store.add("alice", "latest")
    with pytest.raises(ToolError, match=r"32000 serialized JSON bytes.*Reduce max_results"):
        await registry.invoke(name, {})
    assert await registry.invoke(name, {"max_results": 1}) == ["latest"]
    assert [f.fact for f in store.list("alice")] == ["latest", *reversed(notes)]


@pytest.mark.parametrize("name", ["recall", "recall_facts"])
async def test_recall_counts_json_unicode_and_control_escaping(
    registry: ToolRegistry, store: FactStore, name: str
) -> None:
    from harness.loop import _encode_tool_result
    from tools.memory import MAX_RECALL_BYTES

    notes = [f"{i}" + "é" * 2047 for i in range(3)]
    for note in notes:
        store.add("alice", note)
    assert sum(len(note.encode()) for note in notes) < MAX_RECALL_BYTES
    assert len(_encode_tool_result(notes).encode()) > MAX_RECALL_BYTES
    with pytest.raises(ToolError, match="Reduce max_results"):
        await registry.invoke(name, {})
    assert await registry.invoke(name, {"max_results": 2}) == notes[:0:-1]
    # JSON control escapes also consume more bytes than their raw note text.
    store.add("bob", "first" + "\x01" * 4000)
    store.add("bob", "second" + "\x01" * 4000)
    bob = ToolRegistry()
    register_memory_tools(bob, store=store, user_id="bob")
    with pytest.raises(ToolError, match="Reduce max_results"):
        await bob.invoke(name, {})
    assert len(_encode_tool_result(await bob.invoke(name, {"max_results": 1})).encode()) <= (
        MAX_RECALL_BYTES
    )


@pytest.mark.parametrize("name", ["recall", "recall_facts"])
async def test_single_legacy_oversized_note_recall_fails_explicitly(
    store: FactStore, registry: ToolRegistry, name: str
) -> None:
    # Simulate a record written by an older harness without the new write limit.
    note = "legacy" * 6000
    with store._conn:
        store._conn.execute(
            "INSERT INTO memory_facts (user_id, fact, created_at) VALUES (?, ?, ?)",
            ("alice", note, "2026-10-14"),
        )
    with pytest.raises(ToolError, match="inspect stored memory outside the model"):
        await registry.invoke(name, {"max_results": 1})
    assert store.list("alice")[0].fact == note
