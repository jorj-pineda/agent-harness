"""End-to-end bugfix on tiny_repo: read → write → pytest → answer."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from harness.grounding import Grounder
from harness.loop import run_turn
from harness.state import Session
from providers.base import ChatMessage, ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tools import ToolRegistry
from tools.code import register_code_tools
from workspace import Workspace

FIXTURE_REPO = Path(__file__).resolve().parent / "fixtures" / "tiny_repo"

FIXED_CALC = '''"""Minimal module for workspace tool tests."""


def add(a: int, b: int) -> int:
    return a + b


def divide(a: int, b: int) -> float:
    return a / b
'''


@pytest.fixture
def broken_repo(tmp_path: Path) -> Path:
    dest = tmp_path / "tiny_repo"
    shutil.copytree(FIXTURE_REPO, dest)
    return dest


async def test_bugfix_read_write_pytest_on_fixture_repo(broken_repo: Path) -> None:
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "calc.py"})],
            finish_reason="tool_use",
        ),
        make_response(
            tool_calls=[
                ToolCall(
                    id="t2",
                    name="write_file",
                    arguments={"path": "calc.py", "content": FIXED_CALC},
                )
            ],
            finish_reason="tool_use",
        ),
        make_response(
            tool_calls=[
                ToolCall(
                    id="t3",
                    name="run_command",
                    arguments={"argv": ["pytest", "test_calc.py", "-q"]},
                )
            ],
            finish_reason="tool_use",
        ),
        make_response(content="Fixed divide to use float division; pytest passes."),
    )

    registry = ToolRegistry()
    register_code_tools(registry, workspace=Workspace(root=broken_repo))
    session = Session(user_id="dev")
    session.messages.append(ChatMessage(role="system", content="Fix failing tests."))

    response = await run_turn(
        session=session,
        user_input="Fix the failing test in test_calc.py",
        provider=provider,
        registry=registry,
        max_iterations=8,
        grounder=Grounder(escalation_threshold=0.55),
    )

    assert response.answer == "Fixed divide to use float division; pytest passes."
    assert response.files_touched == ["calc.py"]
    assert response.verification_ran is True

    run_calls = [tc for tc in response.tool_calls if tc.name == "run_command"]
    assert len(run_calls) == 1
    assert run_calls[0].error is None
    result = run_calls[0].result
    assert isinstance(result, dict)
    assert result.get("success") is True

    assert (broken_repo / "calc.py").read_text(encoding="utf-8") == FIXED_CALC


async def test_require_verification_escalates_when_pytest_never_ran(broken_repo: Path) -> None:
    provider = ScriptedProvider()
    provider.script(make_response(content="done without running tests"))

    registry = ToolRegistry()
    register_code_tools(registry, workspace=Workspace(root=broken_repo))

    response = await run_turn(
        session=Session(),
        user_input="fix it",
        provider=provider,
        registry=registry,
        grounder=Grounder(escalation_threshold=0.55),
        require_verification_before_finish=True,
    )

    assert response.verification_ran is False
    assert response.escalated is True
