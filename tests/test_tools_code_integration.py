"""Integration: code tools invoked through the harness ReAct loop."""

from __future__ import annotations

from pathlib import Path

from harness.grounding import Grounder
from harness.loop import run_turn
from harness.state import Session
from providers.base import ChatMessage, ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tools import ToolRegistry
from tools.code import register_code_tools
from workspace import Workspace

FIXTURE_REPO = Path(__file__).resolve().parent / "fixtures" / "tiny_repo"


async def test_run_turn_invokes_read_file_on_fixture_repo() -> None:
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "calc.py"})],
            finish_reason="tool_use",
        ),
        make_response(content="calc.py defines add and divide."),
    )

    registry = ToolRegistry()
    register_code_tools(registry, workspace=Workspace(root=FIXTURE_REPO))
    session = Session(user_id="dev")
    session.messages.append(ChatMessage(role="system", content="You are a coding agent."))

    response = await run_turn(
        session=session,
        user_input="What does calc.py export?",
        provider=provider,
        registry=registry,
        max_iterations=4,
        grounder=Grounder(escalation_threshold=0.55),
    )

    assert response.answer == "calc.py defines add and divide."
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].error is None
    result = response.tool_calls[0].result
    assert isinstance(result, dict)
    assert result["path"] == "calc.py"
    assert "def add" in result["content"]

    # Provider received tool specs including read_file.
    _messages, tools = provider.calls[0]
    assert tools is not None
    tool_names = {t.name for t in tools}
    assert "read_file" in tool_names
