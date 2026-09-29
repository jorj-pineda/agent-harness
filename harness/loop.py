"""ReAct loop: pump user input through provider + tools until a final answer.

The loop owns three concerns:

1. **Transcript maintenance** — every user/assistant/tool message is appended
   to `session.messages` in the exact order the provider needs to see it on
   the next call.
2. **Tool dispatch** — when the provider emits `tool_calls`, invoke each via
   the registry, wrap the result back into a `role="tool"` message, and
   record a `ToolCallRecord` on the active `Turn`.
3. **Rule-#5 assembly** — return a `TurnResponse` carrying answer, tool-call
   history, provider name, and wall-clock latency. Grounding (step 7) fills
   in `confidence`/`citations`; memory (step 8) fills in `memory_writes`.

`max_iterations` is a safety net against runaway tool chains. Hitting the
cap returns a stub answer rather than raising — the eval harness scores the
misfire and the session remains usable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from providers.base import ChatMessage, ChatProvider, TokenUsage
from tools import ToolError, ToolRegistry

from .context import BudgetStop, TokenBudget
from .grounding import Grounder
from .memory import harvest_memory_writes
from .outcome import (
    harvest_checks,
    harvest_files_touched,
    harvest_patch_summary,
    harvest_tool_errors,
    verification_status,
)
from .outcome_types import CompletionStatus
from .policy import edit_precondition_error, repeated_unchanged_call, unresolved_edit_blocks
from .state import Session, ToolCallRecord, Turn, TurnResponse
from .stream import EventCallback, ToolEndEvent, ToolStartEvent

RESULT_SNIPPET_LIMIT = 600

log = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 8
MAX_ITERATIONS_STUB = "(max tool iterations reached without a final answer)"


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _encode_tool_result(value: Any) -> str:
    """Serialize a tool result into the `content` string of a tool message.

    `default=str` catches exotic types (datetimes, pydantic models, etc.) so
    the provider always receives valid JSON instead of a raise.
    """
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


async def run_turn(
    *,
    session: Session,
    user_input: str,
    provider: ChatProvider,
    registry: ToolRegistry,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    grounder: Grounder | None = None,
    require_verification_before_finish: bool = False,
    require_plan_before_edit: bool = False,
    max_files_touched_per_turn: int = 0,
    max_tool_calls_per_turn: int = 0,
    max_turn_wall_seconds: float = 0,
    max_completion_tokens_per_turn: int = 0,
    max_total_tokens_per_turn: int = 0,
    max_context_tokens: int = 0,
    min_request_output_tokens: int = 1,
    max_identical_tool_calls: int = 0,
    max_completion_retries: int = 0,
    required_check: list[str] | None = None,
    on_event: EventCallback | None = None,
) -> TurnResponse:
    """Drive one user turn to completion via ReAct + tool dispatch.

    Mutates `session` in place (appends to `messages` and `turns`) and
    returns the rule-#5 payload. When `grounder` is provided, fills in
    `confidence` / `citations` / `escalated` from the turn's tool-call
    history; otherwise those fields stay at their defaults.

    `on_event`, when supplied, observes each tool call (start/end) for live
    streaming — it never alters control flow.
    """
    turn = Turn(user_input=user_input)
    session.turns.append(turn)
    session.messages.append(ChatMessage(role="user", content=user_input))

    tool_specs = registry.as_tool_specs()
    start = _now_ms()
    final_answer = ""
    max_iterations_reached = False
    completion_status: CompletionStatus = "completed"
    completion_reason: str | None = None
    completion_retries = 0
    budget = TokenBudget(
        max_output=max_completion_tokens_per_turn,
        max_total=max_total_tokens_per_turn,
        max_context=max_context_tokens,
        min_request_output=min_request_output_tokens,
    )
    deadline = (
        asyncio.get_running_loop().time() + max_turn_wall_seconds
        if max_turn_wall_seconds > 0
        else None
    )
    wall_limit_reason = f"Turn wall-time budget of {max_turn_wall_seconds:g}s reached."

    for iteration in range(max_iterations):
        allowance = budget.request_allowance(session.messages, tool_specs)
        if isinstance(allowance, BudgetStop):
            final_answer = "(model token budget exhausted before task completion)"
            completion_status = allowance.status
            completion_reason = allowance.reason
            break
        if deadline is not None and asyncio.get_running_loop().time() >= deadline:
            final_answer = "(turn stopped before task completion)"
            completion_status = "budget_exhausted"
            completion_reason = wall_limit_reason
            break
        sent_count = len(session.messages)
        if deadline is None:
            if allowance is None:
                response = await provider.chat(session.messages, tools=tool_specs)
            else:
                response = await provider.chat(
                    session.messages, tools=tool_specs, max_tokens=allowance
                )
        else:
            budget_timeout = asyncio.timeout_at(deadline)
            try:
                async with budget_timeout:
                    if allowance is None:
                        response = await provider.chat(session.messages, tools=tool_specs)
                    else:
                        response = await provider.chat(
                            session.messages, tools=tool_specs, max_tokens=allowance
                        )
            except TimeoutError:
                if not budget_timeout.expired():
                    raise
                budget.mark_unknown()
                final_answer = "(turn stopped before task completion)"
                completion_status = "budget_exhausted"
                completion_reason = wall_limit_reason
                break
        budget.record(response.usage, sent_count)
        if deadline is not None and asyncio.get_running_loop().time() >= deadline:
            final_answer = "(turn stopped before task completion)"
            completion_status = "budget_exhausted"
            completion_reason = wall_limit_reason
            break

        incomplete_reasons = {
            "length": "The model response was truncated.",
            "content_filter": "The provider filtered the response.",
            "error": "The provider reported an error finish reason.",
        }
        usable_calls = [] if response.finish_reason in incomplete_reasons else response.tool_calls
        session.messages.append(
            ChatMessage(
                role="assistant",
                content=response.content,
                tool_calls=list(usable_calls),
            )
        )

        if response.finish_reason in incomplete_reasons:
            final_answer = response.content
            completion_status = "incomplete"
            completion_reason = incomplete_reasons[response.finish_reason]
            break

        budget_stop = budget.after_response(wants_tools=bool(usable_calls))
        if budget_stop is not None and not usable_calls:
            final_answer = response.content
            completion_status = budget_stop.status
            completion_reason = budget_stop.reason
            break

        if not usable_calls:
            checked = verification_status(turn.tool_calls, required_check=required_check)
            edited = bool(harvest_files_touched(turn.tool_calls))
            blocked = unresolved_edit_blocks(turn.tool_calls)
            needs_check = require_verification_before_finish and edited and checked != "passed"
            if needs_check or blocked:
                if completion_retries < max_completion_retries and iteration + 1 < max_iterations:
                    guidance = (
                        f"The edits to {', '.join(blocked)} were blocked. Resolve the tool error "
                        "or report the task as incomplete."
                        if blocked
                        else (
                            f"The latest edit is not verified. Run the configured check "
                            f"{json.dumps(required_check)} after editing, repair any failure, "
                            "then report the observed result."
                            if required_check is not None
                            else "The latest edit is not verified. Run a relevant check after "
                            "editing, repair any failure, then report the observed result."
                        )
                    )
                    session.messages.append(ChatMessage(role="user", content=guidance))
                    completion_retries += 1
                    continue
                completion_status = "incomplete"
                completion_reason = (
                    f"Edits blocked for: {', '.join(blocked)}."
                    if blocked
                    else f"Final edits are not verified ({checked})."
                )
            final_answer = response.content
            break

        stop_reason = budget_stop.reason if budget_stop is not None else None
        stop_status: CompletionStatus = (
            budget_stop.status if budget_stop is not None else "completed"
        )
        for tc in usable_calls:
            if on_event is not None:
                await on_event(ToolStartEvent(tool=tc.name, arguments=dict(tc.arguments)))
            tool_start = _now_ms()
            result: Any = None
            error: str | None = None
            try:
                if stop_reason is not None:
                    raise ToolError("Tool not executed: this turn has already stopped.")
                if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                    stop_reason = wall_limit_reason
                    stop_status = "budget_exhausted"
                    raise ToolError(f"Tool not executed: {stop_reason}")
                if max_tool_calls_per_turn > 0 and len(turn.tool_calls) >= max_tool_calls_per_turn:
                    stop_reason = f"Tool-call limit of {max_tool_calls_per_turn} reached."
                    stop_status = "budget_exhausted"
                    raise ToolError(f"Tool not executed: {stop_reason}")
                if repeated_unchanged_call(
                    tc.name,
                    tc.arguments,
                    turn.tool_calls,
                    max_identical=max_identical_tool_calls,
                ):
                    stop_reason = f"Repeated unchanged call to {tc.name!r}."
                    stop_status = "blocked"
                    raise ToolError(f"Tool not executed: {stop_reason}")
                precondition_error = edit_precondition_error(
                    tc.name,
                    tc.arguments,
                    turn.tool_calls,
                    require_plan=require_plan_before_edit,
                    max_files=max_files_touched_per_turn,
                )
                if precondition_error is not None:
                    raise ToolError(precondition_error)
                result = await registry.invoke(tc.name, tc.arguments)
            except ToolError as exc:
                error = str(exc)
            tool_latency = _now_ms() - tool_start

            turn.tool_calls.append(
                ToolCallRecord(
                    name=tc.name,
                    arguments=dict(tc.arguments),
                    result=result,
                    error=error,
                    latency_ms=tool_latency,
                )
            )
            payload = error if error is not None else result
            if on_event is not None:
                await on_event(
                    ToolEndEvent(
                        tool=tc.name,
                        latency_ms=tool_latency,
                        error=error,
                        result_snippet=_encode_tool_result(payload)[:RESULT_SNIPPET_LIMIT],
                    )
                )
            session.messages.append(
                ChatMessage(
                    role="tool",
                    content=_encode_tool_result(payload),
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                )
            )
        if stop_reason is not None:
            final_answer = "(tool execution stopped before task completion)"
            completion_status = stop_status
            completion_reason = stop_reason
            break
    else:
        final_answer = MAX_ITERATIONS_STUB
        max_iterations_reached = True
        completion_status = "budget_exhausted"
        completion_reason = "Model iteration limit reached before completion."
        log.warning(
            "harness=run_turn max_iterations=%d reached without final answer",
            max_iterations,
        )

    turn.final_answer = final_answer
    turn.finished_at = datetime.now(UTC)
    turn.memory_writes = harvest_memory_writes(turn.tool_calls)

    grounding = (
        grounder.ground(
            answer=final_answer,
            tool_calls=turn.tool_calls,
            max_iterations_reached=max_iterations_reached,
        )
        if grounder is not None
        else None
    )

    files_touched = harvest_files_touched(turn.tool_calls)
    patch_summary = harvest_patch_summary(turn.tool_calls)
    checked = verification_status(turn.tool_calls, required_check=required_check)
    verification_ran = checked == "passed"
    escalated = (grounding.escalated if grounding else False) or completion_status != "completed"

    return TurnResponse(
        answer=final_answer,
        confidence=grounding.confidence if grounding else None,
        citations=list(grounding.citations) if grounding else [],
        escalated=escalated,
        tool_calls=list(turn.tool_calls),
        memory_writes=list(turn.memory_writes),
        files_touched=files_touched,
        verification_ran=verification_ran,
        verification_status=checked,
        check_attempts=harvest_checks(turn.tool_calls, required_check=required_check),
        tool_errors=harvest_tool_errors(turn.tool_calls),
        token_usage=TokenUsage(
            prompt_tokens=budget.prompt_used, completion_tokens=budget.completion_used
        ),
        completion_status=completion_status,
        completion_reason=completion_reason,
        patch_summary=patch_summary,
        provider=provider.name,
        latency_ms=_now_ms() - start,
    )
