"""Scorers for eval scenarios.

Deliberately cheap, deterministic scorers that run without a model call so
the eval matrix stays fast and reproducible in CI. Each scorer takes the
rule-#5 `TurnResponse` (`harness.state.TurnResponse`) plus a scenario-gold
payload and returns a scalar (or bool for escalation).

Why no LLM-judge here:
    An LLM self-assessment would add provider coupling, cost, and
    non-determinism to the one layer that's supposed to be the objective
    judge. We keep scorers first-principles and measurable; an LLM-judge
    correctness scorer is a `Deferred` item in CLAUDE.md to swap in once
    the heuristics visibly underperform on a scenario set.

Contracts:
    faithfulness(response, gold_chunks)    -> float in [0, 1]
    correctness(response, gold_answer)     -> float in [0, 1]
    memory_recall(response, expected_facts) -> float in [0, 1]
    escalation(response, should_escalate)  -> bool
"""

from __future__ import annotations

import re
from collections import Counter

from harness.state import TurnResponse

_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in _WORD_RE.findall(text)]


def faithfulness(response: TurnResponse, gold_chunks: list[str]) -> float:
    """Fraction of gold chunk ids present in `response.citations`.

    Recall-flavored on purpose: the faithfulness question is "did every
    claim that *should* be cited actually get cited?". Precision (over-
    citing) is not penalized here — surplus citations are tracked in the
    run report but don't pull the score down.

    If `gold_chunks` is empty the scenario didn't make a factual claim
    requiring support, so faithfulness is 1.0 regardless of what the agent
    cited.
    """
    if not gold_chunks:
        return 1.0
    cited = set(response.citations)
    hits = sum(1 for chunk_id in gold_chunks if chunk_id in cited)
    return hits / len(gold_chunks)


def correctness(response: TurnResponse, gold_answer: str) -> float:
    """Token-overlap F1 between `response.answer` and `gold_answer`.

    SQuAD-style: tokenize on word characters, case-fold, compare as
    multisets. This catches paraphrase better than exact match and stays
    deterministic.

    Edge cases:
    * both empty         -> 1.0   (agreed the answer is the empty string)
    * one side empty     -> 0.0   (no overlap possible)
    * zero-token overlap -> 0.0
    """
    pred = _tokenize(response.answer)
    gold = _tokenize(gold_answer)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    common = Counter(pred) & Counter(gold)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def memory_recall(response: TurnResponse, expected_facts: list[str]) -> float:
    """Fraction of expected facts surfaced in the answer or memory_writes.

    A fact counts as recalled iff its normalized text (trimmed, lowercased)
    appears as a substring in either `response.answer` or any entry in
    `response.memory_writes`. This accepts both shapes of personalization
    success:
    * the agent *mentions* a previously-stored fact in its reply,
    * the agent *writes* the fact to long-term memory this turn.

    Substring match is a deliberately strict first pass — semantic
    paraphrase scoring is an upgrade path once scenarios start failing it.
    """
    if not expected_facts:
        return 1.0
    haystack_parts = [response.answer, *response.memory_writes]
    haystack = "\n".join(haystack_parts).lower()
    hits = 0
    for fact in expected_facts:
        needle = fact.strip().lower()
        if needle and needle in haystack:
            hits += 1
    return hits / len(expected_facts)


def escalation(response: TurnResponse, should_escalate: bool) -> bool:
    """True iff the agent's escalation decision matches the scenario's gold.

    Returned as a bool rather than a float so the run report can compute
    precision/recall across scenarios directly.
    """
    return response.escalated is should_escalate
