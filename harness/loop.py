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

import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from providers.base import ChatMessage, ChatProvider
from tools import ToolError, ToolRegistry

from .grounding import Grounder
from .memory import harvest_memory_writes
from .state import Session, ToolCallRecord, Turn, TurnResponse

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
) -> TurnResponse:
    """Drive one user turn to completion via ReAct + tool dispatch.

    Mutates `session` in place (appends to `messages` and `turns`) and
    returns the rule-#5 payload. When `grounder` is provided, fills in
    `confidence` / `citations` / `escalated` from the turn's tool-call
    history; otherwise those fields stay at their defaults.
    """
    turn = Turn(user_input=user_input)
    session.turns.append(turn)
    session.messages.append(ChatMessage(role="user", content=user_input))

    tool_specs = registry.as_tool_specs()
    start = _now_ms()
    final_answer = ""
    max_iterations_reached = False

    for _ in range(max_iterations):
        response = await provider.chat(session.messages, tools=tool_specs)

        session.messages.append(
            ChatMessage(
                role="assistant",
                content=response.content,
                tool_calls=list(response.tool_calls),
            )
        )

        if not response.tool_calls:
            final_answer = response.content
            break

        for tc in response.tool_calls:
            tool_start = _now_ms()
            result: Any = None
            error: str | None = None
            try:
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
            session.messages.append(
                ChatMessage(
                    role="tool",
                    content=_encode_tool_result(payload),
                    tool_call_id=tc.id,
                )
            )
    else:
        final_answer = MAX_ITERATIONS_STUB
        max_iterations_reached = True
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

    return TurnResponse(
        answer=final_answer,
        confidence=grounding.confidence if grounding else None,
        citations=list(grounding.citations) if grounding else [],
        escalated=grounding.escalated if grounding else False,
        tool_calls=list(turn.tool_calls),
        memory_writes=list(turn.memory_writes),
        provider=provider.name,
        latency_ms=_now_ms() - start,
    )
