from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from harness.loop import run_turn
from harness.memory import harvest_memory_writes
from harness.state import Session, ToolCallRecord
from memory import FactStore
from providers.base import (
    ChatMessage,
    FinishReason,
    ProviderResponse,
    ToolCall,
    ToolSpec,
)
from tools import ToolRegistry
from tools.base import Tool
from tools.memory import register_memory_tools


def _remember_call(
    *,
    stored: bool,
    fact: str,
    error: str | None = None,
    result_override: object | None = None,
) -> ToolCallRecord:
    result: object = (
        result_override if result_override is not None else {"stored": stored, "fact": fact}
    )
    return ToolCallRecord(
        name="remember_fact",
        arguments={"fact": fact},
        result=result if error is None else None,
        error=error,
    )


def test_harvest_returns_empty_for_no_tool_calls() -> None:
    assert harvest_memory_writes([]) == []


def test_harvest_ignores_non_memory_tools() -> None:
    calls = [
        ToolCallRecord(name="lookup_order", arguments={}, result={"id": 1}),
        ToolCallRecord(name="search_docs", arguments={}, result=[]),
    ]
    assert harvest_memory_writes(calls) == []


def test_harvest_collects_stored_facts_in_call_order() -> None:
    calls = [
        _remember_call(stored=True, fact="prefers vegan options"),
        _remember_call(stored=True, fact="size 9 shoe"),
    ]
    assert harvest_memory_writes(calls) == ["prefers vegan options", "size 9 shoe"]


def test_harvest_skips_duplicates_reported_by_store() -> None:
    calls = [
        _remember_call(stored=True, fact="has a cat"),
        _remember_call(stored=False, fact="has a cat"),
    ]
    assert harvest_memory_writes(calls) == ["has a cat"]


def test_harvest_skips_errored_calls() -> None:
    calls = [
        _remember_call(stored=True, fact="good"),
        _remember_call(stored=True, fact="bad", error="boom"),
    ]
    assert harvest_memory_writes(calls) == ["good"]


def test_harvest_skips_malformed_results_without_crashing() -> None:
    calls = [
        _remember_call(stored=True, fact="ok"),
        _remember_call(stored=True, fact="x", result_override="not-a-dict"),
        _remember_call(stored=True, fact="y", result_override={"stored": True}),  # no fact
        _remember_call(stored=True, fact="z", result_override={"fact": "z"}),  # no stored
    ]
    assert harvest_memory_writes(calls) == ["ok"]


def test_harvest_skips_entries_with_empty_fact_string() -> None:
    calls = [_remember_call(stored=True, fact="", result_override={"stored": True, "fact": ""})]
    assert harvest_memory_writes(calls) == []


# ─── run_turn integration ──────────────────────────────────────────────────


class _FakeProvider:
    name = "fake"

    def __init__(self, responses: Iterable[ProviderResponse]) -> None:
        self._queue = list(responses)

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        return self._queue.pop(0)


def _response(
    *,
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
) -> ProviderResponse:
    return ProviderResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=cast(FinishReason, finish_reason),
        model="fake",
        latency_ms=1.0,
    )


class _EchoInput(BaseModel):
    msg: str


def _echo_tool() -> Tool:
    async def echo(args: _EchoInput) -> str:
        return args.msg

    return Tool(name="echo", description="echo", input_model=_EchoInput, fn=echo)


@pytest.fixture
def store(tmp_path: Path) -> FactStore:
    return FactStore(tmp_path / "memory.db")


async def test_run_turn_populates_memory_writes_from_remember_fact(store: FactStore) -> None:
    provider = _FakeProvider(
        [
            _response(
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="remember_fact",
                        arguments={"fact": "prefers vegan options"},
                    )
                ],
                finish_reason="tool_use",
            ),
            _response(content="got it"),
        ]
    )
    registry = ToolRegistry()
    register_memory_tools(registry, store=store, user_id="alice")

    session = Session()
    resp = await run_turn(
        session=session,
        user_input="remember I'm vegan",
        provider=provider,
        registry=registry,
    )

    assert resp.memory_writes == ["prefers vegan options"]
    assert session.turns[-1].memory_writes == ["prefers vegan options"]
    assert [f.fact for f in store.list("alice")] == ["prefers vegan options"]


async def test_run_turn_excludes_duplicate_remember_calls_from_memory_writes(
    store: FactStore,
) -> None:
    store.add("alice", "already known")
    provider = _FakeProvider(
        [
            _response(
                tool_calls=[
                    ToolCall(id="t1", name="remember_fact", arguments={"fact": "already known"})
                ],
                finish_reason="tool_use",
            ),
            _response(content="noted"),
        ]
    )
    registry = ToolRegistry()
    register_memory_tools(registry, store=store, user_id="alice")

    resp = await run_turn(
        session=Session(),
        user_input="x",
        provider=provider,
        registry=registry,
    )

    assert resp.memory_writes == []


async def test_run_turn_leaves_memory_writes_empty_when_no_memory_tools_are_called() -> None:
    provider = _FakeProvider([_response(content="hi")])
    registry = ToolRegistry()
    registry.register(_echo_tool())

    resp = await run_turn(
        session=Session(),
        user_input="hello",
        provider=provider,
        registry=registry,
    )

    assert resp.memory_writes == []


async def test_run_turn_collects_multiple_memory_writes_in_call_order(store: FactStore) -> None:
    provider = _FakeProvider(
        [
            _response(
                tool_calls=[
                    ToolCall(id="t1", name="remember_fact", arguments={"fact": "fact one"}),
                    ToolCall(id="t2", name="remember_fact", arguments={"fact": "fact two"}),
                ],
                finish_reason="tool_use",
            ),
            _response(content="done"),
        ]
    )
    registry = ToolRegistry()
    register_memory_tools(registry, store=store, user_id="alice")

    resp = await run_turn(
        session=Session(),
        user_input="x",
        provider=provider,
        registry=registry,
    )

    assert resp.memory_writes == ["fact one", "fact two"]
