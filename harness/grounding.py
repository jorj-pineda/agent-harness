"""Grounding layer: convert answer + tool-call history into rule-#5 metadata.

`Grounder.ground()` harvests citations from retrieval tool results, scores
confidence with a deterministic heuristic, and decides escalation against
a threshold. Deliberately no LLM calls — the confidence signal is
inspectable, cheap, and measurable by the eval harness directly.

Evidence sources (pivot Phase 3):

* **Support RAG** — `search_docs` hits with Chroma similarity scores.
* **Code tools** — `read_file` / `grep_repo` results mapped to file:line
  citation keys (`path:start-end` on the wire in `TurnResponse.citations`).

Heuristic shape (unchanged):

    confidence = top_score * coverage_factor * health_factor

If no retrieval or code-evidence tool ran, confidence is `None` — ungrounded
chitchat. If evidence tools ran but returned nothing usable, confidence is
0.0 and escalation fires.

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
READ_FILE_TOOL_NAME = "read_file"
GREP_REPO_TOOL_NAME = "grep_repo"
CODE_EVIDENCE_TOOL_NAMES = frozenset({READ_FILE_TOOL_NAME, GREP_REPO_TOOL_NAME})

DEFAULT_MIN_CITATION_SCORE = 0.3
DEFAULT_MAX_CITATIONS = 5
ERROR_HEALTH_PENALTY = 0.7
MAX_ITERATION_HEALTH_PENALTY = 0.5

READ_FILE_EVIDENCE_SCORE = 1.0
GREP_HIT_EVIDENCE_SCORE = 0.85


class CodeCitation(BaseModel):
    """Structured file:line citation harvested from code tools."""

    path: str
    start_line: int
    end_line: int
    snippet: str | None = None

    def wire_key(self) -> str:
        if self.start_line == self.end_line:
            return f"{self.path}:{self.start_line}"
        return f"{self.path}:{self.start_line}-{self.end_line}"


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
        rag_calls = [tc for tc in tool_calls if tc.name == SEARCH_TOOL_NAME]
        code_calls = [tc for tc in tool_calls if tc.name in CODE_EVIDENCE_TOOL_NAMES]

        if not rag_calls and not code_calls:
            return GroundingResult(confidence=None, citations=[], escalated=False)

        hits = list(self._collect_rag_hits(rag_calls)) + list(self._collect_code_hits(code_calls))
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

    def _collect_rag_hits(
        self,
        retrieval_calls: list[ToolCallRecord],
    ) -> Iterable[dict[str, Any]]:
        for call in retrieval_calls:
            if call.error is not None or not isinstance(call.result, list):
                continue
            for hit in call.result:
                if isinstance(hit, dict) and "chunk_id" in hit and "score" in hit:
                    yield hit

    def _collect_code_hits(
        self,
        code_calls: list[ToolCallRecord],
    ) -> Iterable[dict[str, Any]]:
        for call in code_calls:
            if call.error is not None:
                continue
            if call.name == READ_FILE_TOOL_NAME:
                yield from self._hits_from_read_file(call.result)
            elif call.name == GREP_REPO_TOOL_NAME:
                yield from self._hits_from_grep(call.result)

    def _hits_from_read_file(self, result: Any) -> Iterable[dict[str, Any]]:
        if not isinstance(result, dict) or "path" not in result:
            return
        path = str(result["path"])
        start = int(result.get("start_line", 1))
        end = int(result.get("end_line", start))
        snippet = result.get("content")
        snippet_str = snippet if isinstance(snippet, str) else None
        citation = CodeCitation(
            path=path,
            start_line=start,
            end_line=end,
            snippet=snippet_str,
        )
        yield {"chunk_id": citation.wire_key(), "score": READ_FILE_EVIDENCE_SCORE}

    def _hits_from_grep(self, result: Any) -> Iterable[dict[str, Any]]:
        if not isinstance(result, list):
            return
        for hit in result:
            if not isinstance(hit, dict) or "path" not in hit or "line" not in hit:
                continue
            line = int(hit["line"])
            path = str(hit["path"])
            text = hit.get("text")
            snippet = text if isinstance(text, str) else None
            citation = CodeCitation(path=path, start_line=line, end_line=line, snippet=snippet)
            yield {"chunk_id": citation.wire_key(), "score": GREP_HIT_EVIDENCE_SCORE}

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
