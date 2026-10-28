from __future__ import annotations

from harness.context import (
    MESSAGE_OVERHEAD_TOKENS,
    BudgetStop,
    TokenBudget,
    estimate_message_tokens,
    estimate_tool_tokens,
)
from providers.base import ChatMessage, TokenUsage, ToolCall, ToolSpec

SPEC = ToolSpec(name="echo", description="Echo text", parameters_schema={"type": "object"})


def test_estimate_counts_content_bytes_tool_calls_and_specs() -> None:
    ascii_message = ChatMessage(role="user", content="a" * 30)
    assert estimate_message_tokens([ascii_message]) == 10 + MESSAGE_OVERHEAD_TOKENS
    wide = ChatMessage(role="user", content="é" * 30)
    assert estimate_message_tokens([wide]) == 20 + MESSAGE_OVERHEAD_TOKENS
    call = ChatMessage(
        role="assistant",
        tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "x" * 300})],
    )
    assert estimate_message_tokens([call]) > 100
    assert estimate_tool_tokens([SPEC]) > 0
    assert estimate_tool_tokens([]) == 0


def test_estimate_takes_larger_of_heuristic_and_reported_anchor() -> None:
    messages = [ChatMessage(role="user", content="x" * 300)]
    heuristic = estimate_message_tokens(messages) + estimate_tool_tokens([SPEC])

    reported_large = TokenBudget()
    reported_large.record(TokenUsage(prompt_tokens=5_000, completion_tokens=10), 1)
    reply = [*messages, ChatMessage(role="assistant", content="ok")]
    assert reported_large.estimate_prompt(reply, [SPEC]) == 5_000 + estimate_message_tokens(
        reply[1:]
    )

    cache_undercount = TokenBudget()
    cache_undercount.record(TokenUsage(prompt_tokens=3, completion_tokens=10), 1)
    assert cache_undercount.estimate_prompt(messages, [SPEC]) == heuristic


def test_allowance_is_smallest_remaining_budget_or_a_stop() -> None:
    messages = [ChatMessage(role="user", content="x" * 300)]
    estimate = estimate_message_tokens(messages)
    budget = TokenBudget(max_output=50, max_total=estimate + 80, max_context=estimate + 200)
    assert budget.request_allowance(messages, []) == 50
    budget.completion_used = 40
    assert budget.request_allowance(messages, []) == 10
    budget.completion_used = 0
    budget.prompt_used = 60
    assert budget.request_allowance(messages, []) == 20

    tight = TokenBudget(max_context=estimate + 99, min_request_output=100)
    stop = tight.request_allowance(messages, [])
    assert isinstance(stop, BudgetStop)
    assert stop.status == "budget_exhausted"
    assert "context limit" in stop.reason
    assert TokenBudget().request_allowance(messages, []) is None


def test_after_response_fails_closed_and_blocks_tools_that_cannot_be_followed_up() -> None:
    missing = TokenBudget(max_total=1_000)
    missing.record(TokenUsage(prompt_tokens=None, completion_tokens=5), 1)
    stop = missing.after_response(wants_tools=True)
    assert stop is not None and stop.status == "incomplete"

    over = TokenBudget(max_total=100)
    over.record(TokenUsage(prompt_tokens=90, completion_tokens=20), 1)
    stop = over.after_response(wants_tools=False)
    assert stop is not None and "exceeded" in stop.reason

    full = TokenBudget(max_total=200, min_request_output=50)
    full.record(TokenUsage(prompt_tokens=100, completion_tokens=10), 1)
    assert full.after_response(wants_tools=False) is None
    stop = full.after_response(wants_tools=True)
    assert stop is not None and "cannot fit another model request" in stop.reason

    context = TokenBudget(max_context=120, min_request_output=50)
    context.record(TokenUsage(prompt_tokens=100, completion_tokens=10), 1)
    stop = context.after_response(wants_tools=True)
    assert stop is not None and "Context limit" in stop.reason


def test_request_refusal_records_estimate_and_reported_lower_bound_separately() -> None:
    # The two observed recovery boundaries: C fails its reported lower bound;
    # P fits that bound, executes a read, then fails the whole-prompt estimate.
    c = TokenBudget(max_total=32000, min_request_output=256)
    c.record(TokenUsage(prompt_tokens=23880 - 9719, completion_tokens=38), 1)
    c.record(TokenUsage(prompt_tokens=9719, completion_tokens=77), 3)
    stop = c.after_response(wants_tools=True)
    assert stop is not None
    assert stop.request_block.model_dump() == {
        "format": "request-budget-block-v1",
        "budget": "total",
        "basis": "reported_prompt_lower_bound",
        "limit_tokens": 32000,
        "available_tokens": 8005,
        "prompt_tokens": 9719,
        "minimum_output_tokens": 256,
    }
    p = TokenBudget(max_total=32000, min_request_output=256)
    p.record(TokenUsage(prompt_tokens=11252, completion_tokens=36), 1)
    p.record(TokenUsage(prompt_tokens=8799, completion_tokens=31), 1)
    assert p.after_response(wants_tools=True) is None
    messages = [ChatMessage(role="user", content="a" * (11854 - 4) * 3)]
    stop = p.request_allowance(messages, [])
    assert isinstance(stop, BudgetStop)
    assert stop.request_block.model_dump() == {
        "format": "request-budget-block-v1",
        "budget": "total",
        "basis": "estimate",
        "limit_tokens": 32000,
        "available_tokens": 11882,
        "prompt_tokens": 11854,
        "minimum_output_tokens": 256,
    }


def test_context_refusal_details_and_unrelated_stops_remain_distinct() -> None:
    budget = TokenBudget(max_context=100, min_request_output=10)
    messages = [ChatMessage(role="user", content="x" * 300)]
    stop = budget.request_allowance(messages, [])
    assert isinstance(stop, BudgetStop)
    assert stop.request_block.budget == "context"
    assert stop.request_block.basis == "estimate"
    assert stop.request_block.available_tokens == 100
    budget.record(TokenUsage(prompt_tokens=95, completion_tokens=1), 1)
    stop = budget.after_response(wants_tools=True)
    assert stop.request_block.basis == "reported_prompt_lower_bound"
    assert stop.request_block.prompt_tokens == 95
    missing = TokenBudget(max_total=100)
    missing.record(TokenUsage(prompt_tokens=None, completion_tokens=1), 1)
    assert missing.after_response(wants_tools=True).request_block is None
    overrun = TokenBudget(max_total=100)
    overrun.record(TokenUsage(prompt_tokens=101, completion_tokens=1), 1)
    assert overrun.after_response(wants_tools=True).request_block is None
