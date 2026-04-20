from __future__ import annotations

import math
from collections.abc import Iterable
from typing import cast

import pytest
from pydantic import BaseModel

from harness.grounding import (
    DEFAULT_MAX_CITATIONS,
    ERROR_HEALTH_PENALTY,
    MAX_ITERATION_HEALTH_PENALTY,
    Grounder,
    GroundingResult,
)
from harness.loop import MAX_ITERATIONS_STUB, run_turn
from harness.state import Session, ToolCallRecord
from providers.base import (
    ChatMessage,
    FinishReason,
    ProviderResponse,
    ToolCall,
    ToolSpec,
)
from tools import ToolRegistry
from tools.base import Tool


def _hit(chunk_id: str, score: float, **extra: object) -> dict[str, object]:
    return {"chunk_id": chunk_id, "score": score, **extra}


def _search_call(result: object, *, error: str | None = None) -> ToolCallRecord:
    return ToolCallRecord(
        name="search_docs",
        arguments={"query": "q"},
        result=result,
        error=error,
    )


def test_no_tool_calls_yields_none_confidence_and_no_escalation() -> None:
    g = Grounder(escalation_threshold=0.5)

    result = g.ground(answer="hi", tool_calls=[])

    assert result.confidence is None
    assert result.citations == []
    assert result.escalated is False


def test_non_retrieval_tool_calls_still_leave_confidence_none() -> None:
    g = Grounder(escalation_threshold=0.5)
    unrelated = ToolCallRecord(name="lookup_order", arguments={}, result={"ok": True})

    result = g.ground(answer="hi", tool_calls=[unrelated])

    assert result.confidence is None
    assert result.escalated is False


def test_high_scoring_retrieval_yields_high_confidence_and_citations() -> None:
    g = Grounder(escalation_threshold=0.5, min_citation_score=0.3)
    hits = [_hit("returns#window", 0.9), _hit("returns#refunds", 0.8)]

    result = g.ground(answer="within 30 days", tool_calls=[_search_call(hits)])

    assert result.confidence == pytest.approx(0.9)  # top * coverage(1.0) * health(1.0)
    assert result.citations == ["returns#window", "returns#refunds"]
    assert result.escalated is False


def test_citations_deduped_across_multiple_search_calls_preserving_first_seen_order() -> None:
    g = Grounder(escalation_threshold=0.5)
    call_a = _search_call([_hit("a", 0.9), _hit("b", 0.7)])
    call_b = _search_call([_hit("b", 0.8), _hit("c", 0.6)])

    result = g.ground(answer="x", tool_calls=[call_a, call_b])

    assert result.citations == ["a", "b", "c"]


def test_low_scoring_hits_are_excluded_from_citations_and_drag_coverage() -> None:
    g = Grounder(escalation_threshold=0.5, min_citation_score=0.5)
    hits = [_hit("good", 0.8), _hit("meh", 0.2), _hit("bad", 0.1)]

    result = g.ground(answer="x", tool_calls=[_search_call(hits)])

    assert result.citations == ["good"]
    # top=0.8, coverage=1/3, health=1.0
    assert result.confidence == pytest.approx(0.8 * (1 / 3))


def test_empty_retrieval_result_collapses_confidence_to_zero_and_escalates() -> None:
    g = Grounder(escalation_threshold=0.5)

    result = g.ground(answer="x", tool_calls=[_search_call([])])

    assert result.confidence == 0.0
    assert result.citations == []
    assert result.escalated is True


def test_search_errors_are_treated_as_no_hits_and_trigger_escalation() -> None:
    g = Grounder(escalation_threshold=0.5)
    errored = _search_call(result=None, error="Chroma unavailable")

    result = g.ground(answer="x", tool_calls=[errored])

    assert result.confidence == 0.0
    assert result.citations == []
    assert result.escalated is True


def test_tool_error_on_any_call_applies_health_penalty() -> None:
    g = Grounder(escalation_threshold=0.5, min_citation_score=0.3)
    good = _search_call([_hit("a", 0.9)])
    other_error = ToolCallRecord(name="lookup_order", arguments={}, error="boom")

    result = g.ground(answer="x", tool_calls=[good, other_error])

    assert result.confidence == pytest.approx(0.9 * ERROR_HEALTH_PENALTY)


def test_max_iterations_reached_applies_health_penalty() -> None:
    g = Grounder(escalation_threshold=0.5)
    good = _search_call([_hit("a", 0.9)])

    result = g.ground(answer="x", tool_calls=[good], max_iterations_reached=True)

    assert result.confidence == pytest.approx(0.9 * MAX_ITERATION_HEALTH_PENALTY)


def test_both_penalties_stack_multiplicatively() -> None:
    g = Grounder(escalation_threshold=0.5)
    errored = ToolCallRecord(name="lookup_order", arguments={}, error="boom")
    good = _search_call([_hit("a", 1.0)])

    result = g.ground(
        answer="x",
        tool_calls=[good, errored],
        max_iterations_reached=True,
    )

    expected = 1.0 * ERROR_HEALTH_PENALTY * MAX_ITERATION_HEALTH_PENALTY
    assert result.confidence == pytest.approx(expected)


def test_confidence_crosses_threshold_from_above_to_below() -> None:
    hits = [_hit("a", 0.6)]

    above = Grounder(escalation_threshold=0.5).ground(answer="x", tool_calls=[_search_call(hits)])
    below = Grounder(escalation_threshold=0.7).ground(answer="x", tool_calls=[_search_call(hits)])

    assert above.escalated is False
    assert below.escalated is True


def test_citations_are_capped_at_max_citations() -> None:
    g = Grounder(escalation_threshold=0.0, max_citations=3)
    hits = [_hit(f"c{i}", 0.9) for i in range(10)]

    result = g.ground(answer="x", tool_calls=[_search_call(hits)])

    assert result.citations == ["c0", "c1", "c2"]
    assert len(result.citations) <= DEFAULT_MAX_CITATIONS


def test_malformed_hits_are_skipped_without_crashing() -> None:
    g = Grounder(escalation_threshold=0.0)
    hits: list[object] = [
        "not-a-dict",
        {"score": 0.9},  # missing chunk_id
        {"chunk_id": "bare"},  # missing score
        _hit("ok", 0.8),
    ]

    result = g.ground(answer="x", tool_calls=[_search_call(hits)])

    assert result.citations == ["ok"]
    assert result.confidence == pytest.approx(0.8)


def test_confidence_is_clamped_to_unit_interval() -> None:
    g = Grounder(escalation_threshold=0.5)
    # Chroma distances can exceed 1.0 → 1 - distance can go negative; inverse also possible.
    hits = [_hit("weird", 1.5)]

    result = g.ground(answer="x", tool_calls=[_search_call(hits)])

    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0


def test_construction_rejects_invalid_escalation_threshold() -> None:
    with pytest.raises(ValueError, match="escalation_threshold"):
        Grounder(escalation_threshold=1.5)
    with pytest.raises(ValueError, match="escalation_threshold"):
        Grounder(escalation_threshold=-0.1)


def test_construction_rejects_invalid_min_citation_score() -> None:
    with pytest.raises(ValueError, match="min_citation_score"):
        Grounder(escalation_threshold=0.5, min_citation_score=2.0)


def test_construction_rejects_non_positive_max_citations() -> None:
    with pytest.raises(ValueError, match="max_citations"):
        Grounder(escalation_threshold=0.5, max_citations=0)


def test_grounding_result_is_json_serializable() -> None:
    r = GroundingResult(confidence=0.82, citations=["a", "b"], escalated=False)
    parsed = GroundingResult.model_validate_json(r.model_dump_json())

    assert parsed == r


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


class _SearchInput(BaseModel):
    query: str


def _stub_search_tool(hits: list[dict[str, object]]) -> Tool:
    async def search_docs(args: _SearchInput) -> list[dict[str, object]]:
        return hits

    return Tool(
        name="search_docs",
        description="stub",
        input_model=_SearchInput,
        fn=search_docs,
    )


async def test_run_turn_without_grounder_leaves_grounding_fields_at_defaults() -> None:
    provider = _FakeProvider([_response(content="hi")])
    resp = await run_turn(
        session=Session(),
        user_input="x",
        provider=provider,
        registry=ToolRegistry(),
    )

    assert resp.confidence is None
    assert resp.citations == []
    assert resp.escalated is False


async def test_run_turn_with_grounder_fills_confidence_citations_escalated() -> None:
    hits = [{"chunk_id": "returns#window", "score": 0.9}]
    provider = _FakeProvider(
        [
            _response(
                tool_calls=[ToolCall(id="t1", name="search_docs", arguments={"query": "returns"})],
                finish_reason="tool_use",
            ),
            _response(content="within 30 days"),
        ]
    )
    registry = ToolRegistry()
    registry.register(_stub_search_tool(hits))

    resp = await run_turn(
        session=Session(),
        user_input="how do returns work?",
        provider=provider,
        registry=registry,
        grounder=Grounder(escalation_threshold=0.5),
    )

    assert resp.answer == "within 30 days"
    assert resp.citations == ["returns#window"]
    assert resp.confidence is not None
    assert math.isclose(resp.confidence, 0.9)
    assert resp.escalated is False


async def test_run_turn_escalates_when_retrieval_is_weak() -> None:
    hits = [{"chunk_id": "maybe", "score": 0.35}]
    provider = _FakeProvider(
        [
            _response(
                tool_calls=[ToolCall(id="t1", name="search_docs", arguments={"query": "x"})],
                finish_reason="tool_use",
            ),
            _response(content="uncertain answer"),
        ]
    )
    registry = ToolRegistry()
    registry.register(_stub_search_tool(hits))

    resp = await run_turn(
        session=Session(),
        user_input="x",
        provider=provider,
        registry=registry,
        grounder=Grounder(escalation_threshold=0.5),
    )

    assert resp.escalated is True
    assert resp.citations == ["maybe"]


async def test_run_turn_passes_max_iterations_signal_to_grounder() -> None:
    hits = [{"chunk_id": "a", "score": 0.9}]
    # Two tool-use responses then stop — but max_iterations=2 means we never reach the stop.
    provider = _FakeProvider(
        [
            _response(
                tool_calls=[ToolCall(id=f"t{i}", name="search_docs", arguments={"query": "x"})],
                finish_reason="tool_use",
            )
            for i in range(2)
        ]
    )
    registry = ToolRegistry()
    registry.register(_stub_search_tool(hits))

    resp = await run_turn(
        session=Session(),
        user_input="spin",
        provider=provider,
        registry=registry,
        max_iterations=2,
        grounder=Grounder(escalation_threshold=0.5),
    )

    assert resp.answer == MAX_ITERATIONS_STUB
    # 0.9 * 1.0 coverage * 0.5 max-iter penalty = 0.45 → below threshold
    assert resp.confidence == pytest.approx(0.9 * MAX_ITERATION_HEALTH_PENALTY)
    assert resp.escalated is True
