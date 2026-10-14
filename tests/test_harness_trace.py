from __future__ import annotations

import asyncio

import pytest

from harness.loop import run_turn
from harness.state import Session
from harness.trace import (
    BudgetStopTrace,
    MessageTrace,
    RequestFailureTrace,
    ResponseTrace,
    TurnTraceRecord,
)
from providers.base import ChatMessage, ToolCall
from tests.test_harness_loop import FakeProvider, _echo_tool, _response
from tools import ToolRegistry


async def test_trace_reconstructs_requests_and_preserves_blank_recovery() -> None:
    records: list[TurnTraceRecord] = []
    first = _response(content=" ")
    first.raw = {"authorization": "secret-must-not-be-copied"}
    provider = FakeProvider(
        [
            first,
            _response(
                tool_calls=[
                    ToolCall(id="one", name="echo", arguments={"text": "hello"}),
                    ToolCall(id="two", name="echo", arguments={"text": "world"}),
                ]
            ),
            _response(content="Done", prompt_tokens=20, completion_tokens=2),
        ]
    )
    registry = ToolRegistry()
    registry.register(_echo_tool())
    session = Session(messages=[ChatMessage(role="system", content="instructions")])
    result = await run_turn(
        session=session,
        user_input="task",
        provider=provider,
        registry=registry,
        max_completion_retries=1,
        trace=records,
    )
    assert result.answer == "Done"
    assert [r.kind for r in records] == [
        "request",
        "response",
        "recovery",
        "request",
        "response",
        "tool_result",
        "tool_result",
        "request",
        "response",
    ]
    transcript: list[ChatMessage] = []
    requests = 0
    for record in records:
        if record.kind == "request":
            if requests == 0:
                transcript = list(record.initial_messages)
            else:
                assert record.initial_messages == []
            assert transcript == provider.calls[requests][0]
            assert record.message_count == len(transcript)
            assert record.estimated_prompt_tokens > 0
            assert record.max_tokens is None
            requests += 1
        elif record.kind == "response":
            transcript.append(
                ChatMessage(
                    role="assistant",
                    content=record.content,
                    tool_calls=record.tool_calls,
                )
            )
        elif record.kind in ("tool_result", "recovery"):
            transcript.append(record.message)
    assert transcript == session.messages
    assert requests == 3
    assert "secret-must-not-be-copied" not in str([r.model_dump() for r in records])
    assert isinstance(records[1], ResponseTrace)
    assert records[1].content == " "
    assert isinstance(records[-1], ResponseTrace)
    assert records[-1].usage.prompt_tokens == 20
    # Evidence is copied, not a view of mutable provider/session objects.
    first.content = "changed"
    session.messages[0].content = "changed"
    assert records[1].content == " "
    assert records[0].kind == "request"
    assert records[0].initial_messages[0].content == "instructions"


@pytest.mark.parametrize("finish", ["length", "content_filter", "error"])
async def test_trace_keeps_suppressed_provider_calls(finish: str) -> None:
    records: list[TurnTraceRecord] = []
    provider = FakeProvider(
        [
            _response(
                content="partial",
                finish_reason=finish,
                tool_calls=[ToolCall(id="unsafe", name="echo", arguments={"text": "x"})],
            )
        ]
    )
    session = Session()
    result = await run_turn(
        session=session,
        user_input="task",
        provider=provider,
        registry=ToolRegistry(),
        trace=records,
    )
    assert result.completion_status == "incomplete"
    assert result.tool_calls == []
    assert [r.kind for r in records] == ["request", "response"]
    assert isinstance(records[1], ResponseTrace)
    assert records[1].tool_calls[0].id == "unsafe"
    assert session.messages[-1].tool_calls == []


async def test_trace_keeps_last_batch_including_budget_rejection() -> None:
    records: list[TurnTraceRecord] = []
    registry = ToolRegistry()
    registry.register(_echo_tool())
    result = await run_turn(
        session=Session(),
        user_input="task",
        registry=registry,
        provider=FakeProvider(
            [
                _response(
                    tool_calls=[
                        ToolCall(id="one", name="echo", arguments={"text": "x"}),
                        ToolCall(id="two", name="echo", arguments={"text": "y"}),
                    ]
                )
            ]
        ),
        max_tool_calls_per_turn=1,
        trace=records,
    )
    assert result.completion_status == "budget_exhausted"
    assert [r.kind for r in records] == ["request", "response", "tool_result", "tool_result"]
    messages = [r.message for r in records if isinstance(r, MessageTrace)]
    assert [m.tool_call_id for m in messages] == ["one", "two"]
    assert "not executed" in messages[-1].content


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_trace_records_failure_type_without_exception_text(
    failure: type[BaseException],
) -> None:
    class FailingProvider(FakeProvider):
        async def chat(self, *args: object, **kwargs: object) -> None:
            raise failure("sensitive exception detail")

    records: list[TurnTraceRecord] = []
    with pytest.raises(failure):
        await run_turn(
            session=Session(),
            user_input="task",
            registry=ToolRegistry(),
            provider=FailingProvider([]),
            trace=records,
        )
    assert isinstance(records[-1], RequestFailureTrace)
    assert records[-1].exception_type == failure.__name__
    assert "sensitive exception detail" not in str(records)


async def test_trace_matches_untraced_execution_and_limits() -> None:
    records: list[TurnTraceRecord] = []
    results = []
    sessions = []
    providers = []
    for trace in (None, records):
        provider = FakeProvider(
            [
                _response(content="", prompt_tokens=10, completion_tokens=2),
                _response(content="Done", prompt_tokens=12, completion_tokens=3),
            ]
        )
        session = Session()
        results.append(
            await run_turn(
                session=session,
                user_input="task",
                provider=provider,
                registry=ToolRegistry(),
                max_completion_retries=1,
                max_completion_tokens_per_turn=10,
                trace=trace,
            )
        )
        sessions.append(session)
        providers.append(provider)
    assert sessions[0].messages == sessions[1].messages
    assert providers[0].calls == providers[1].calls
    assert providers[0].max_tokens_seen == providers[1].max_tokens_seen == [10, 8]
    assert results[0].model_dump(exclude={"latency_ms"}) == results[1].model_dump(
        exclude={"latency_ms"}
    )
    assert [r.max_tokens for r in records if r.kind == "request"] == [10, 8]


async def test_budget_before_request_emits_no_request_evidence() -> None:
    records: list[TurnTraceRecord] = []
    provider = FakeProvider([])
    result = await run_turn(
        session=Session(),
        user_input="long task",
        provider=provider,
        registry=ToolRegistry(),
        max_context_tokens=1,
        trace=records,
    )
    assert result.completion_status == "budget_exhausted"
    assert len(records) == 1 and isinstance(records[0], BudgetStopTrace)
    assert records[0].footprint.estimated_prompt_tokens > 1
    assert records[0].max_context_tokens == 1
    assert records[0].message_count == 1
    assert records[0].prompt_tokens_used == 0
    assert "long task" not in records[0].model_dump_json()
    assert provider.calls == []


async def test_wall_timeout_records_request_failure() -> None:
    class SlowProvider(FakeProvider):
        async def chat(self, *args: object, **kwargs: object) -> None:
            await asyncio.sleep(1)

    records: list[TurnTraceRecord] = []
    result = await run_turn(
        session=Session(),
        user_input="task",
        registry=ToolRegistry(),
        provider=SlowProvider([]),
        max_turn_wall_seconds=0.01,
        trace=records,
    )
    assert result.completion_status == "budget_exhausted"
    assert [r.kind for r in records] == ["request", "request_failure"]
    assert isinstance(records[-1], RequestFailureTrace)
    assert records[-1].exception_type == "TimeoutError"


def test_footprint_uses_budget_math_and_caps_payload_free_rankings() -> None:
    import json

    from harness.context import TokenBudget, estimate_message_tokens, estimate_tool_tokens
    from harness.trace import RequestTrace, prompt_footprint
    from providers.base import TokenUsage, ToolSpec

    messages = [ChatMessage(role="user", content="é" * 6) for _ in range(12)]
    messages.append(
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="private-call-id",
                    name="read_file",
                    arguments={"path": "private-path", "text": "secret-text"},
                )
            ],
        )
    )
    tools = [ToolSpec(name="echo", description="private-schema", parameters_schema={})]
    budget = TokenBudget()
    budget.record(TokenUsage(prompt_tokens=10000, completion_tokens=2), len(messages))
    footprint = prompt_footprint(messages, tools, budget)
    assert footprint.heuristic_prompt_tokens == estimate_message_tokens(
        messages
    ) + estimate_tool_tokens(tools)
    assert footprint.estimated_prompt_tokens == 10000
    assert footprint.heuristic_prompt_tokens + footprint.anchor_adjustment_tokens == 10000
    assert footprint.largest_messages_truncated
    assert len(footprint.largest_messages) == 10
    assert [row.message_index for row in footprint.largest_messages] == [12, *range(9)]
    assert footprint.largest_messages[1].payload_bytes == 12
    serialized = footprint.model_dump_json()
    for secret in ("private-call-id", "private-path", "secret-text", "private-schema"):
        assert secret not in serialized
    legacy = RequestTrace.model_validate(
        json.loads(
            '{"iteration":0,"message_count":1,"estimated_prompt_tokens":10,"max_tokens":null}'
        )
    )
    assert legacy.footprint is None


async def test_real_reads_reach_a_blocked_request_without_changing_user_work(tmp_path) -> None:
    from harness.config import Settings
    from harness.runtime import build_registry, run_configured_turn
    from memory import FactStore

    content = "preserved source " * 1000
    path = tmp_path / "source.txt"
    path.write_text(content)
    settings = Settings(_env_file=None, max_context_tokens=0)
    with FactStore(tmp_path / "facts.db") as store:
        registry = build_registry(fact_store=store, user_id="dev", workspace_root=str(tmp_path))

        def provider():
            return FakeProvider(
                [
                    _response(
                        tool_calls=[
                            ToolCall(id="read", name="read_file", arguments={"path": "source.txt"})
                        ],
                        prompt_tokens=10,
                        completion_tokens=2,
                    ),
                    _response(content="Reviewed", prompt_tokens=20, completion_tokens=2),
                ]
            )

        control_trace: list[TurnTraceRecord] = []
        await run_configured_turn(
            session=Session(workspace_root=str(tmp_path)),
            message="Inspect source",
            provider=provider(),
            registry=registry,
            fact_store=store,
            user_id="dev",
            settings=settings,
            trace=control_trace,
        )
        first = control_trace[0]
        assert first.kind == "request"
        limit = first.estimated_prompt_tokens + settings.min_request_output_tokens
        traced: list[TurnTraceRecord] = []
        model = provider()
        session = Session(workspace_root=str(tmp_path))
        result = await run_configured_turn(
            session=session,
            message="Inspect source",
            provider=model,
            registry=registry,
            fact_store=store,
            user_id="dev",
            settings=settings.model_copy(update={"max_context_tokens": limit}),
            trace=traced,
        )
    assert result.completion_status == "budget_exhausted"
    assert len(model.calls) == 1
    stop = traced[-1]
    assert isinstance(stop, BudgetStopTrace) and stop.iteration == 1
    assert stop.max_context_tokens == limit
    assert stop.footprint.estimated_prompt_tokens + stop.min_request_output_tokens > limit
    assert stop.footprint.largest_messages[0].tool_name == "read_file"
    assert stop.footprint.largest_messages[0].message_index == 3
    assert stop.prompt_tokens_used == 10 and stop.completion_tokens_used == 2
    assert result.tool_calls[0].result["content"] == content
    assert path.read_text() == content
    assert result.workspace_changes.status == "tracked"
    assert result.workspace_changes.modified == []
