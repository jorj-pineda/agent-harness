from __future__ import annotations

from harness.state import Session, ToolCallRecord, Turn, TurnResponse
from providers.base import ChatMessage, ToolCall


def test_session_defaults_are_empty_containers_with_a_generated_id() -> None:
    s = Session()
    assert s.session_id
    assert s.user_id is None
    assert s.messages == []
    assert s.turns == []


def test_session_ids_are_unique_across_instances() -> None:
    assert Session().session_id != Session().session_id


def test_turn_ids_are_unique_across_instances() -> None:
    assert Turn(user_input="a").turn_id != Turn(user_input="a").turn_id


def test_session_carries_raw_chat_messages_in_provider_shape() -> None:
    s = Session()
    s.messages.append(ChatMessage(role="user", content="hi"))
    s.messages.append(
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="t1", name="search_docs", arguments={"q": "hi"})],
        )
    )
    assert s.messages[0].role == "user"
    assert s.messages[1].tool_calls[0].name == "search_docs"


def test_turn_accumulates_tool_call_records_and_a_final_answer() -> None:
    t = Turn(user_input="how do returns work?")
    t.tool_calls.append(
        ToolCallRecord(
            name="search_docs",
            arguments={"query": "returns", "k": 2},
            result=[{"chunk_id": "returns#window"}],
            latency_ms=12.3,
        )
    )
    t.final_answer = "within 30 days"
    assert t.tool_calls[0].name == "search_docs"
    assert t.tool_calls[0].latency_ms == 12.3
    assert t.final_answer == "within 30 days"


def test_tool_call_record_captures_errors_separately_from_results() -> None:
    rec = ToolCallRecord(name="lookup_order", arguments={"id": "x"}, error="Unknown tool")
    assert rec.result is None
    assert rec.error == "Unknown tool"


def test_turn_response_carries_the_rule_5_metadata_shape() -> None:
    r = TurnResponse(answer="hello", provider="ollama", latency_ms=12.3)
    assert r.answer == "hello"
    assert r.confidence is None
    assert r.citations == []
    assert r.tool_calls == []
    assert r.memory_writes == []
    assert r.provider == "ollama"
    assert r.latency_ms == 12.3


def test_turn_response_round_trips_through_json() -> None:
    original = TurnResponse(
        answer="Returns accepted within 30 days.",
        confidence=0.82,
        citations=["returns#window"],
        tool_calls=[
            ToolCallRecord(
                name="search_docs",
                arguments={"query": "returns", "k": 1},
                result=[{"chunk_id": "returns#window"}],
                latency_ms=4.2,
            )
        ],
        memory_writes=[],
        provider="ollama",
        latency_ms=123.4,
    )
    parsed = TurnResponse.model_validate_json(original.model_dump_json())
    assert parsed == original


def test_turn_response_requires_provider_and_latency() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TurnResponse(answer="x")  # type: ignore[call-arg]


def test_session_round_trips_through_json_with_nested_turns() -> None:
    s = Session(user_id="u-1")
    s.messages.append(ChatMessage(role="user", content="hi"))
    s.turns.append(Turn(user_input="hi", final_answer="hello"))
    parsed = Session.model_validate_json(s.model_dump_json())
    assert parsed.user_id == "u-1"
    assert parsed.messages[0].content == "hi"
    assert parsed.turns[0].final_answer == "hello"
