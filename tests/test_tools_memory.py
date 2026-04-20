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

    assert {"remember_fact", "recall_facts"} <= names


async def test_register_memory_tools_rejects_double_registration(store: FactStore) -> None:
    reg = ToolRegistry()
    register_memory_tools(reg, store=store, user_id="alice")

    with pytest.raises(ToolError, match="already registered"):
        register_memory_tools(reg, store=store, user_id="alice")
