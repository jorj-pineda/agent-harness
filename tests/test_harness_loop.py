from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from pydantic import BaseModel

from harness.loop import DEFAULT_MAX_ITERATIONS, MAX_ITERATIONS_STUB, run_turn
from harness.state import Session
from providers.base import (
    ChatMessage,
    FinishReason,
    ProviderResponse,
    ToolCall,
    ToolSpec,
)
from tools import ToolRegistry
from tools.base import Tool


class FakeProvider:
    """Scripted provider — pops pre-canned responses per chat() call."""

    name = "fake"

    def __init__(self, responses: Iterable[ProviderResponse]) -> None:
        self._queue = list(responses)
        self.calls: list[tuple[list[ChatMessage], list[ToolSpec] | None]] = []

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        self.calls.append(([m.model_copy(deep=True) for m in messages], tools))
        assert self._queue, "FakeProvider ran out of scripted responses"
        return self._queue.pop(0)


def _response(
    *,
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
) -> ProviderResponse:
    return ProviderResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=cast(FinishReason, finish_reason),
        model="fake-model",
        latency_ms=1.0,
    )


class EchoInput(BaseModel):
    text: str


class AddInput(BaseModel):
    a: int
    b: int


async def _echo(args: EchoInput) -> str:
    return args.text


async def _add(args: AddInput) -> int:
    return args.a + args.b


def _echo_tool() -> Tool:
    return Tool(name="echo", description="Echo text", input_model=EchoInput, fn=_echo)


def _add_tool() -> Tool:
    return Tool(name="add", description="Add two ints", input_model=AddInput, fn=_add)


def _registry(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


async def test_one_shot_answer_skips_tool_dispatch() -> None:
    provider = FakeProvider([_response(content="hello")])
    session = Session()

    resp = await run_turn(session=session, user_input="hi", provider=provider, registry=_registry())

    assert resp.answer == "hello"
    assert resp.tool_calls == []
    assert resp.provider == "fake"
    assert resp.latency_ms > 0
    assert [m.role for m in session.messages] == ["user", "assistant"]
    assert session.messages[0].content == "hi"
    assert session.messages[1].content == "hello"
    assert len(session.turns) == 1
    assert session.turns[0].final_answer == "hello"


async def test_single_tool_call_then_final_answer() -> None:
    provider = FakeProvider(
        [
            _response(
                tool_calls=[ToolCall(id="t1", name="echo", arguments={"text": "abc"})],
                finish_reason="tool_use",
            ),
            _response(content="done"),
        ]
    )
    session = Session()

    resp = await run_turn(
        session=session,
        user_input="echo abc",
        provider=provider,
        registry=_registry(_echo_tool()),
    )

    assert resp.answer == "done"
    assert len(resp.tool_calls) == 1
    rec = resp.tool_calls[0]
    assert rec.name == "echo"
    assert rec.arguments == {"text": "abc"}
    assert rec.result == "abc"
    assert rec.error is None
    assert rec.latency_ms >= 0

    assert [m.role for m in session.messages] == ["user", "assistant", "tool", "assistant"]
    tool_msg = session.messages[2]
    assert tool_msg.tool_call_id == "t1"
    assert tool_msg.content == '"abc"'  # json-encoded result


async def test_multiple_tool_calls_in_single_response() -> None:
    provider = FakeProvider(
        [
            _response(
                tool_calls=[
                    ToolCall(id="t1", name="add", arguments={"a": 2, "b": 3}),
                    ToolCall(id="t2", name="echo", arguments={"text": "ok"}),
                ],
                finish_reason="tool_use",
            ),
            _response(content="both done"),
        ]
    )
    session = Session()

    resp = await run_turn(
        session=session,
        user_input="do both",
        provider=provider,
        registry=_registry(_add_tool(), _echo_tool()),
    )

    assert resp.answer == "both done"
    assert [rec.name for rec in resp.tool_calls] == ["add", "echo"]
    assert resp.tool_calls[0].result == 5
    assert resp.tool_calls[1].result == "ok"

    assert [m.role for m in session.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert session.messages[2].tool_call_id == "t1"
    assert session.messages[3].tool_call_id == "t2"


async def test_multi_iteration_tool_chain() -> None:
    provider = FakeProvider(
        [
            _response(
                tool_calls=[ToolCall(id="t1", name="echo", arguments={"text": "one"})],
                finish_reason="tool_use",
            ),
            _response(
                tool_calls=[ToolCall(id="t2", name="echo", arguments={"text": "two"})],
                finish_reason="tool_use",
            ),
            _response(content="chained"),
        ]
    )
    session = Session()

    resp = await run_turn(
        session=session,
        user_input="chain",
        provider=provider,
        registry=_registry(_echo_tool()),
    )

    assert resp.answer == "chained"
    assert [rec.arguments["text"] for rec in resp.tool_calls] == ["one", "two"]
    assert len(provider.calls) == 3


async def test_tool_error_is_captured_and_fed_back() -> None:
    provider = FakeProvider(
        [
            _response(
                tool_calls=[ToolCall(id="t1", name="missing", arguments={})],
                finish_reason="tool_use",
            ),
            _response(content="recovered"),
        ]
    )
    session = Session()

    resp = await run_turn(
        session=session,
        user_input="x",
        provider=provider,
        registry=_registry(_echo_tool()),
    )

    assert resp.answer == "recovered"
    assert len(resp.tool_calls) == 1
    rec = resp.tool_calls[0]
    assert rec.result is None
    assert rec.error is not None
    assert "missing" in rec.error.lower()

    tool_msg = session.messages[2]
    assert tool_msg.role == "tool"
    assert tool_msg.tool_call_id == "t1"
    assert "missing" in tool_msg.content.lower()


async def test_max_iterations_guard_returns_stub() -> None:
    provider = FakeProvider(
        [
            _response(
                tool_calls=[ToolCall(id=f"t{i}", name="echo", arguments={"text": "x"})],
                finish_reason="tool_use",
            )
            for i in range(2)
        ]
    )
    session = Session()

    resp = await run_turn(
        session=session,
        user_input="spin",
        provider=provider,
        registry=_registry(_echo_tool()),
        max_iterations=2,
    )

    assert resp.answer == MAX_ITERATIONS_STUB
    assert len(resp.tool_calls) == 2


async def test_turn_is_appended_with_timestamps() -> None:
    provider = FakeProvider([_response(content="hi")])
    session = Session()

    await run_turn(session=session, user_input="hello", provider=provider, registry=_registry())

    turn = session.turns[0]
    assert turn.user_input == "hello"
    assert turn.final_answer == "hi"
    assert turn.finished_at is not None
    assert turn.finished_at >= turn.started_at


async def test_tool_specs_are_forwarded_to_provider() -> None:
    provider = FakeProvider([_response(content="ok")])
    session = Session()

    await run_turn(
        session=session,
        user_input="hi",
        provider=provider,
        registry=_registry(_echo_tool(), _add_tool()),
    )

    _, tools = provider.calls[0]
    assert tools is not None
    assert {s.name for s in tools} == {"echo", "add"}


async def test_provider_sees_growing_transcript_across_iterations() -> None:
    provider = FakeProvider(
        [
            _response(
                tool_calls=[ToolCall(id="t1", name="echo", arguments={"text": "a"})],
                finish_reason="tool_use",
            ),
            _response(content="final"),
        ]
    )
    session = Session()

    await run_turn(
        session=session,
        user_input="go",
        provider=provider,
        registry=_registry(_echo_tool()),
    )

    assert [m.role for m in provider.calls[0][0]] == ["user"]
    assert [m.role for m in provider.calls[1][0]] == ["user", "assistant", "tool"]


def test_default_max_iterations_is_eight() -> None:
    assert DEFAULT_MAX_ITERATIONS == 8
