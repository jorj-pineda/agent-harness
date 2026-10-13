"""Per-turn model token budgets checked before each provider request.

Prompt size is estimated, not tokenized: providers use different tokenizers, and
the harness does not load one per model. The estimate takes the larger of:

- a byte heuristic over the whole prompt (UTF-8 bytes / 3, plus per-message
  overhead), which is conservative for English and code; and
- the previous request's reported prompt tokens plus the heuristic for messages
  appended since (including the reply), which tracks the provider's own count.
  Reported output tokens are not reused: they can include hidden reasoning that
  is never sent back.

The heuristic alone matters because some backends omit cached prompt tokens
from reported usage. Reported totals remain the authority after each request;
a provider can still exceed a budget the estimate expected to fit.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

from providers.base import ChatMessage, TokenUsage, ToolSpec

from .outcome_types import CompletionStatus

BYTES_PER_TOKEN = 3
MESSAGE_OVERHEAD_TOKENS = 4


def _tokens_for_bytes(size: int) -> int:
    return math.ceil(size / BYTES_PER_TOKEN)


def estimate_message_tokens(messages: Sequence[ChatMessage]) -> int:
    size = 0
    for message in messages:
        size += len(message.content.encode("utf-8"))
        size += len((message.tool_name or "").encode("utf-8"))
        for call in message.tool_calls:
            size += len(call.name.encode("utf-8"))
            size += len(json.dumps(call.arguments, default=str).encode("utf-8"))
    return _tokens_for_bytes(size) + MESSAGE_OVERHEAD_TOKENS * len(messages)


def estimate_tool_tokens(tools: Sequence[ToolSpec]) -> int:
    if not tools:
        return 0
    return _tokens_for_bytes(len(json.dumps([tool.model_dump() for tool in tools]).encode()))


@dataclass(frozen=True)
class BudgetStop:
    status: CompletionStatus
    reason: str


@dataclass
class TokenBudget:
    """Output, total, and context-window budgets for one turn; 0 disables each."""

    max_output: int = 0
    max_total: int = 0
    max_context: int = 0
    min_request_output: int = 1
    prompt_used: int | None = 0
    completion_used: int | None = 0
    _anchor_count: int = 0
    _anchor_tokens: int | None = None

    def estimate_prompt(self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> int:
        heuristic = estimate_message_tokens(messages) + estimate_tool_tokens(tools)
        if self._anchor_tokens is None:
            return heuristic
        anchored = self._anchor_tokens + estimate_message_tokens(messages[self._anchor_count :])
        return max(heuristic, anchored)

    def request_allowance(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> int | None | BudgetStop:
        """Return `max_tokens` for the next request, None for no cap, or why to stop first."""
        allowances: list[int] = []
        if self.max_output > 0 and self.completion_used is not None:
            if self.completion_used >= self.max_output:
                return BudgetStop("budget_exhausted", "Model output-token budget reached.")
            allowances.append(self.max_output - self.completion_used)
        if self.max_total <= 0 and self.max_context <= 0:
            return min(allowances) if allowances else None
        estimate = self.estimate_prompt(messages, tools)
        if self.max_context > 0:
            room = self.max_context - estimate
            if room < self.min_request_output:
                return BudgetStop(
                    "budget_exhausted",
                    f"Estimated prompt of ~{estimate} tokens leaves fewer than "
                    f"{self.min_request_output} output tokens in the {self.max_context}-token "
                    "context limit.",
                )
            allowances.append(room)
        if self.max_total > 0 and self.prompt_used is not None and self.completion_used is not None:
            room = self.max_total - self.prompt_used - self.completion_used - estimate
            if room < self.min_request_output:
                return BudgetStop(
                    "budget_exhausted",
                    f"Estimated next request of ~{estimate} prompt tokens does not fit the "
                    f"remaining total token budget ({self.max_total} per turn).",
                )
            allowances.append(room)
        return min(allowances) if allowances else None

    def record(self, usage: TokenUsage, prompt_message_count: int) -> None:
        """Add reported usage for a request that sent `prompt_message_count` messages."""
        prompt, completion = usage.prompt_tokens, usage.completion_tokens
        self.prompt_used = (
            self.prompt_used + prompt
            if self.prompt_used is not None and prompt is not None
            else None
        )
        self.completion_used = (
            self.completion_used + completion
            if self.completion_used is not None and completion is not None
            else None
        )
        if prompt is None:
            self._anchor_tokens = None
            return
        self._anchor_count = prompt_message_count
        self._anchor_tokens = prompt

    def mark_unknown(self) -> None:
        self.prompt_used = None
        self.completion_used = None
        self._anchor_tokens = None

    def after_response(self, *, wants_tools: bool) -> BudgetStop | None:
        """Decide whether reported usage stops the turn before tools or a final answer."""
        if self.max_output > 0:
            if self.completion_used is None:
                return BudgetStop(
                    "incomplete",
                    "Provider did not report output-token usage; budget cannot be enforced.",
                )
            if self.completion_used > self.max_output:
                return BudgetStop(
                    "budget_exhausted", "Provider exceeded the requested model output-token budget."
                )
            if self.completion_used == self.max_output and wants_tools:
                return BudgetStop("budget_exhausted", "Model output-token budget reached.")
        if self.max_total > 0:
            if self.prompt_used is None or self.completion_used is None:
                return BudgetStop(
                    "incomplete",
                    "Provider did not report token usage; total budget cannot be enforced.",
                )
            used = self.prompt_used + self.completion_used
            if used > self.max_total:
                return BudgetStop(
                    "budget_exhausted", "Reported model tokens exceeded the total token budget."
                )
            if wants_tools and self._next_request_cannot_fit(self.max_total - used):
                return BudgetStop(
                    "budget_exhausted",
                    "Total token budget cannot fit another model request after these tools.",
                )
        if self.max_context > 0 and wants_tools and self._next_request_cannot_fit(self.max_context):
            return BudgetStop(
                "budget_exhausted",
                "Context limit cannot fit another model request after these tools.",
            )
        return None

    def _next_request_cannot_fit(self, room: int) -> bool:
        # The next prompt resends the last one, so its reported size is a lower bound.
        return self._anchor_tokens is not None and (
            self._anchor_tokens + self.min_request_output > room
        )
