"""Scorers for eval scenarios.

Deterministic scorers over the rule-#5 `TurnResponse` envelope. Support-era
scorers (`faithfulness`, `correctness`, `memory_recall`) remain for archived
`scenarios_support.yaml`. Coding scenarios use `code_faithfulness`,
`patch_correctness`, and `verification_score` alongside `escalation`.
"""

from __future__ import annotations

import re
from collections import Counter

from harness.state import TurnResponse

_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in _WORD_RE.findall(text)]


def faithfulness(response: TurnResponse, gold_chunks: list[str]) -> float:
    """Fraction of gold citation keys present in `response.citations` (RAG chunk ids)."""
    if not gold_chunks:
        return 1.0
    cited = set(response.citations)
    hits = sum(1 for chunk_id in gold_chunks if chunk_id in cited)
    return hits / len(gold_chunks)


def code_faithfulness(response: TurnResponse, gold_citations: list[str]) -> float:
    """Fraction of required file:line citation keys present in `response.citations`."""
    return faithfulness(response, gold_citations)


def patch_correctness(response: TurnResponse, gold_files: list[str]) -> float:
    """Fraction of expected repo-relative paths present in `response.files_touched`."""
    if not gold_files:
        return 1.0
    touched = set(response.files_touched)
    hits = sum(1 for path in gold_files if path in touched)
    return hits / len(gold_files)


def verification_score(response: TurnResponse, should_verify: bool | None) -> float:
    """1.0 when verification expectation matches `response.verification_ran`."""
    if should_verify is None or should_verify is False:
        return 1.0
    return 1.0 if response.verification_ran else 0.0


def correctness(response: TurnResponse, gold_answer: str) -> float:
    """Token-overlap F1 between `response.answer` and `gold_answer`."""
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
    """Fraction of expected engineering notes surfaced in answer or memory_writes."""
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
    """True iff the agent's escalation decision matches the scenario gold."""
    return response.escalated is should_escalate
