from pathlib import Path

import pytest

from harness.config import Settings
from harness.loop import run_turn
from harness.runtime import run_configured_turn
from harness.state import Session
from harness.transcript import inspect_transcript
from memory import FactStore
from providers.base import ChatMessage, ToolCall
from tools import ToolRegistry

from .test_harness_loop import FakeProvider, _echo_tool, _response


def assistant(*ids: str) -> ChatMessage:
    return ChatMessage(role="assistant", tool_calls=[ToolCall(id=id, name="echo") for id in ids])


def result(id: str, name: str | None = "echo") -> ChatMessage:
    return ChatMessage(role="tool", tool_call_id=id, tool_name=name, content="fixture payload")


def test_valid_batches_allow_reused_ids_out_of_order_results_and_recovery_users():
    messages = [
        ChatMessage(role="system", content="rules"),
        ChatMessage(role="user", content="task"),
        assistant("a", "b"),
        result("b", None),
        result("a"),
        ChatMessage(role="assistant", content=""),
        ChatMessage(role="user", content="retry"),
        assistant("a"),
        result("a"),
        ChatMessage(role="assistant", content="done"),
    ]
    saved = [m.model_dump() for m in messages]
    assert inspect_transcript(messages).status == "valid"
    assert [m.model_dump() for m in messages] == saved
    assert inspect_transcript([]).status == "valid"


@pytest.mark.parametrize(
    "messages,code",
    [
        ([assistant("a")], "missing_tool_result"),
        (
            [assistant("a"), ChatMessage(role="user", content="continue"), result("a")],
            "missing_tool_result",
        ),
        ([assistant("a", "a"), result("a")], "duplicate_tool_call_id"),
        ([assistant(" ")], "empty_tool_call_id"),
        (
            [ChatMessage(role="assistant", tool_calls=[ToolCall(id="a", name=" ")])],
            "empty_tool_name",
        ),
        ([result("a")], "orphan_tool_result"),
        ([assistant("a"), result("a"), result("a")], "orphan_tool_result"),
        ([assistant("a"), result("a", "different")], "tool_name_mismatch"),
        (
            [ChatMessage(role="user", tool_calls=[ToolCall(id="a", name="echo")])],
            "tool_calls_on_non_assistant",
        ),
        ([ChatMessage(role="assistant", tool_call_id="a")], "tool_result_fields_on_non_tool"),
        ([ChatMessage(role="user"), ChatMessage(role="system")], "misplaced_system_message"),
    ],
)
def test_invalid_structure_reports_codes_without_payloads(messages, code):
    report = inspect_transcript(messages)
    assert report.status == "invalid"
    assert code in [issue.code for issue in report.issues]
    assert "fixture payload" not in report.model_dump_json()
    assert report.read_only is True
    assert "not permission to resume" in report.scope_note


def test_diagnostics_are_bounded_with_full_count():
    review = inspect_transcript([result(str(i)) for i in range(10)], max_issues=2)
    assert review.issue_count == 10
    assert len(review.issues) == 2
    assert review.issues_truncated
    with pytest.raises(ValueError):
        inspect_transcript([], max_issues=0)


async def test_invalid_history_blocks_loop_without_mutation_or_inference():
    session = Session(messages=[assistant("pending")])
    saved = session.model_dump()
    provider = FakeProvider([])
    response = await run_turn(
        session=session, user_input="continue", provider=provider, registry=ToolRegistry()
    )
    assert response.completion_status == "blocked"
    assert session.model_dump() == saved
    assert provider.calls == []


async def test_configured_guard_precedes_system_refresh_snapshots_and_finalization(
    tmp_path: Path, monkeypatch
):
    import harness.runtime as runtime

    def unexpected(*args, **kwargs):
        pytest.fail("No snapshot or callback may run for malformed history")

    monkeypatch.setattr(runtime, "_snapshot_or_reason", unexpected)
    session = Session(workspace_root=str(tmp_path), messages=[assistant("pending")])
    saved = session.model_dump()
    provider = FakeProvider([])
    store = FactStore(tmp_path / "facts.db")
    try:
        response = await run_configured_turn(
            settings=Settings(_env_file=None),
            session=session,
            user_id="dev",
            message="continue",
            provider=provider,
            fact_store=store,
            registry=ToolRegistry(),
            on_finalized=unexpected,
            on_workspace_review=unexpected,
        )
    finally:
        store.close()
    assert response.completion_status == "blocked"
    assert session.model_dump() == saved
    assert provider.calls == []


@pytest.mark.parametrize("ids", [("a", "a"), ("",), (" ",)])
async def test_malformed_response_batch_stops_before_any_tool_dispatch(ids):
    registry = ToolRegistry()
    registry.register(_echo_tool())
    provider = FakeProvider(
        [
            _response(
                tool_calls=[
                    ToolCall(id=id, name="echo", arguments={"text": "would execute"}) for id in ids
                ]
            )
        ]
    )
    session = Session()
    response = await run_turn(
        session=session, user_input="task", provider=provider, registry=registry
    )
    assert response.completion_status == "incomplete"
    assert response.tool_calls == []
    assert len(provider.calls) == 1
    assert len(session.messages) == 1  # Only the user's message; no fabricated results.
    assert inspect_transcript(session.messages).status == "valid"
    assert session.turns[-1].finished_at is not None
