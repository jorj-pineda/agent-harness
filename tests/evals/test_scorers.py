from __future__ import annotations

import pytest

from evals.scorers import correctness, escalation, faithfulness, memory_recall
from harness.state import TurnResponse


def _response(
    *,
    answer: str = "",
    citations: list[str] | None = None,
    memory_writes: list[str] | None = None,
    escalated: bool = False,
) -> TurnResponse:
    return TurnResponse(
        answer=answer,
        citations=citations or [],
        memory_writes=memory_writes or [],
        escalated=escalated,
        provider="fake",
        latency_ms=1.0,
    )


# ─── faithfulness ───────────────────────────────────────────────────────────


def test_faithfulness_full_coverage_scores_one() -> None:
    resp = _response(citations=["chunk_a", "chunk_b"])
    assert faithfulness(resp, ["chunk_a", "chunk_b"]) == pytest.approx(1.0)


def test_faithfulness_partial_coverage_is_fractional() -> None:
    resp = _response(citations=["chunk_a"])
    assert faithfulness(resp, ["chunk_a", "chunk_b"]) == pytest.approx(0.5)


def test_faithfulness_no_citations_scores_zero_when_gold_nonempty() -> None:
    resp = _response(citations=[])
    assert faithfulness(resp, ["chunk_a"]) == pytest.approx(0.0)


def test_faithfulness_empty_gold_is_one_regardless_of_citations() -> None:
    resp = _response(citations=["irrelevant"])
    assert faithfulness(resp, []) == pytest.approx(1.0)


def test_faithfulness_extra_citations_do_not_penalize() -> None:
    resp = _response(citations=["chunk_a", "chunk_b", "bonus"])
    assert faithfulness(resp, ["chunk_a"]) == pytest.approx(1.0)


def test_faithfulness_is_order_independent() -> None:
    resp = _response(citations=["b", "a"])
    assert faithfulness(resp, ["a", "b"]) == pytest.approx(1.0)


# ─── correctness ────────────────────────────────────────────────────────────


def test_correctness_exact_match_scores_one() -> None:
    resp = _response(answer="Items can be returned within 30 days.")
    assert correctness(resp, "Items can be returned within 30 days.") == pytest.approx(1.0)


def test_correctness_case_and_punctuation_insensitive() -> None:
    resp = _response(answer="ITEMS, returned within 30 days!")
    score = correctness(resp, "items returned within 30 days")
    assert score == pytest.approx(1.0)


def test_correctness_partial_overlap_is_fractional() -> None:
    resp = _response(answer="returned within 30 days")
    gold = "items can be returned within 30 days of purchase"
    score = correctness(resp, gold)
    assert 0.0 < score < 1.0


def test_correctness_no_overlap_scores_zero() -> None:
    resp = _response(answer="completely unrelated text")
    assert correctness(resp, "quantum entanglement phenomena") == pytest.approx(0.0)


def test_correctness_both_empty_is_one() -> None:
    resp = _response(answer="")
    assert correctness(resp, "") == pytest.approx(1.0)


def test_correctness_empty_prediction_nonempty_gold_is_zero() -> None:
    resp = _response(answer="")
    assert correctness(resp, "anything goes here") == pytest.approx(0.0)


def test_correctness_empty_gold_nonempty_prediction_is_zero() -> None:
    resp = _response(answer="surplus text")
    assert correctness(resp, "") == pytest.approx(0.0)


# ─── memory_recall ──────────────────────────────────────────────────────────


def test_memory_recall_matches_fact_in_answer() -> None:
    resp = _response(answer="Your known preference: prefers vegan options.")
    assert memory_recall(resp, ["prefers vegan options"]) == pytest.approx(1.0)


def test_memory_recall_matches_fact_in_memory_writes() -> None:
    resp = _response(memory_writes=["size 9 shoe"])
    assert memory_recall(resp, ["size 9 shoe"]) == pytest.approx(1.0)


def test_memory_recall_partial_hit_is_fractional() -> None:
    resp = _response(answer="Remembered: prefers vegan options.", memory_writes=[])
    score = memory_recall(resp, ["prefers vegan options", "size 9 shoe"])
    assert score == pytest.approx(0.5)


def test_memory_recall_case_insensitive() -> None:
    resp = _response(answer="You PREFER VEGAN options.")
    assert memory_recall(resp, ["prefer vegan"]) == pytest.approx(1.0)


def test_memory_recall_empty_expected_is_one() -> None:
    resp = _response(answer="")
    assert memory_recall(resp, []) == pytest.approx(1.0)


def test_memory_recall_no_hits_scores_zero() -> None:
    resp = _response(answer="totally unrelated")
    assert memory_recall(resp, ["vegan"]) == pytest.approx(0.0)


def test_memory_recall_blank_expected_fact_does_not_count_as_hit() -> None:
    resp = _response(answer="hello world")
    # blank needles should score 0 hits out of 1 → 0.0, not 1/1.
    assert memory_recall(resp, ["   "]) == pytest.approx(0.0)


# ─── escalation ─────────────────────────────────────────────────────────────


def test_escalation_matches_when_both_true() -> None:
    resp = _response(escalated=True)
    assert escalation(resp, True) is True


def test_escalation_matches_when_both_false() -> None:
    resp = _response(escalated=False)
    assert escalation(resp, False) is True


def test_escalation_mismatch_returns_false() -> None:
    resp = _response(escalated=True)
    assert escalation(resp, False) is False
    resp2 = _response(escalated=False)
    assert escalation(resp2, True) is False
