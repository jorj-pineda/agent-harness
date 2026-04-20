"""Grounding layer: convert answer + tool-call history into rule-#5 metadata.

`Grounder.ground()` harvests citations from `search_docs` results, scores
confidence with a deterministic heuristic, and decides escalation against
a threshold. Deliberately no LLM calls — the confidence signal is
inspectable, cheap, and measurable by the eval harness directly.

Heuristic shape:

    confidence = top_score * coverage_factor * health_factor

- `top_score`        — best retrieval score across all `search_docs` hits.
- `coverage_factor`  — fraction of hits at or above `min_citation_score`.
- `health_factor`    — 1.0 by default, penalized for tool errors and for
                       loops that hit their max-iteration cap.

If no `search_docs` call happened at all, confidence is `None` — the
answer is ungrounded and escalation is not triggered (pure chitchat
shouldn't page a human). If retrieval *was* attempted but returned
nothing usable, confidence is 0.0 and escalation fires.

Planned upgrades (see CLAUDE.md "Deferred"): LLM-judge confidence,
per-sentence attribution, answer rewriting on escalation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from .state import ToolCallRecord

log = logging.getLogger(__name__)

SEARCH_TOOL_NAME = "search_docs"
DEFAULT_MIN_CITATION_SCORE = 0.3
DEFAULT_MAX_CITATIONS = 5
ERROR_HEALTH_PENALTY = 0.7
MAX_ITERATION_HEALTH_PENALTY = 0.5


class GroundingResult(BaseModel):
    """Output of `Grounder.ground()` — the rule-#5 metadata the loop fills in."""

    confidence: float | None = None
    citations: list[str] = Field(default_factory=list)
    escalated: bool = False


class Grounder:
    """Score confidence and harvest citations from a completed turn."""

    def __init__(
        self,
        *,
        escalation_threshold: float,
        min_citation_score: float = DEFAULT_MIN_CITATION_SCORE,
        max_citations: int = DEFAULT_MAX_CITATIONS,
    ) -> None:
        if not 0.0 <= escalation_threshold <= 1.0:
            raise ValueError("escalation_threshold must be in [0, 1]")
        if not 0.0 <= min_citation_score <= 1.0:
            raise ValueError("min_citation_score must be in [0, 1]")
        if max_citations < 1:
            raise ValueError("max_citations must be >= 1")
        self._escalation_threshold = escalation_threshold
        self._min_citation_score = min_citation_score
        self._max_citations = max_citations

    @property
    def escalation_threshold(self) -> float:
        return self._escalation_threshold

    def ground(
        self,
        *,
        answer: str,
        tool_calls: list[ToolCallRecord],
        max_iterations_reached: bool = False,
    ) -> GroundingResult:
        retrieval_calls = [tc for tc in tool_calls if tc.name == SEARCH_TOOL_NAME]

        if not retrieval_calls:
            return GroundingResult(confidence=None, citations=[], escalated=False)

        hits = list(self._collect_hits(retrieval_calls))
        citations = self._pick_citations(hits)
        confidence = self._score(
            hits,
            tool_errors=any(tc.error is not None for tc in tool_calls),
            max_iterations_reached=max_iterations_reached,
        )
        escalated = confidence < self._escalation_threshold

        log.info(
            "grounding confidence=%.3f citations=%d escalated=%s",
            confidence,
            len(citations),
            escalated,
        )
        return GroundingResult(
            confidence=confidence,
            citations=citations,
            escalated=escalated,
        )

    def _collect_hits(
        self,
        retrieval_calls: list[ToolCallRecord],
    ) -> Iterable[dict[str, Any]]:
        for call in retrieval_calls:
            if call.error is not None or not isinstance(call.result, list):
                continue
            for hit in call.result:
                if isinstance(hit, dict) and "chunk_id" in hit and "score" in hit:
                    yield hit

    def _pick_citations(self, hits: list[dict[str, Any]]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for hit in hits:
            if float(hit["score"]) < self._min_citation_score:
                continue
            chunk_id = str(hit["chunk_id"])
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            out.append(chunk_id)
            if len(out) >= self._max_citations:
                break
        return out

    def _score(
        self,
        hits: list[dict[str, Any]],
        *,
        tool_errors: bool,
        max_iterations_reached: bool,
    ) -> float:
        if not hits:
            return 0.0
        scores = [float(h["score"]) for h in hits]
        top_score = max(scores)
        above = sum(1 for s in scores if s >= self._min_citation_score)
        coverage_factor = above / len(scores)
        health_factor = 1.0
        if tool_errors:
            health_factor *= ERROR_HEALTH_PENALTY
        if max_iterations_reached:
            health_factor *= MAX_ITERATION_HEALTH_PENALTY
        return max(0.0, min(1.0, top_score * coverage_factor * health_factor))
