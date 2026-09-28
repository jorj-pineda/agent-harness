"""End-to-end bugfix on tiny_repo: read → write → pytest → answer."""

from __future__ import annotations

import hashlib
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
    assert response.patch_summary == ["calc.py (154 bytes written)"]

    run_calls = [tc for tc in response.tool_calls if tc.name == "run_command"]
    assert len(run_calls) == 1
    assert run_calls[0].error is None
    result = run_calls[0].result
    assert isinstance(result, dict)
    assert result.get("success") is True

    assert (broken_repo / "calc.py").read_text(encoding="utf-8") == FIXED_CALC


async def test_early_finish_gets_one_check_retry_and_can_complete(broken_repo: Path) -> None:
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "calc.py", "content": FIXED_CALC},
                )
            ]
        ),
        make_response(content="Fixed."),
        make_response(
            tool_calls=[
                ToolCall(id="check", name="run_command", arguments={"argv": ["pytest", "-q"]})
            ]
        ),
        make_response(content="Fixed; pytest passed."),
    )
    registry = ToolRegistry()
    register_code_tools(registry, workspace=Workspace(root=broken_repo))
    response = await run_turn(
        session=Session(),
        user_input="fix divide",
        provider=provider,
        registry=registry,
        require_verification_before_finish=True,
        max_completion_retries=1,
    )
    assert len(provider.calls) == 4
    assert "not verified" in provider.calls[2][0][-1].content
    assert response.completion_status == "completed"
    assert response.verification_status == "passed"
    assert response.verification_ran is True
    assert response.escalated is False


async def test_bugfix_read_replace_pytest_tracks_targeted_edit(broken_repo: Path) -> None:
    original = (broken_repo / "calc.py").read_bytes()
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[ToolCall(id="r1", name="read_file", arguments={"path": "calc.py"})],
            finish_reason="tool_use",
        ),
        make_response(
            tool_calls=[
                ToolCall(
                    id="r2",
                    name="replace_text",
                    arguments={
                        "path": "calc.py",
                        "old_text": "return 0.0  # intentional bug — Phase 4 e2e tests fix with write_file + pytest",
                        "new_text": "return a / b",
                        "expected_sha256": hashlib.sha256(original).hexdigest(),
                    },
                )
            ],
            finish_reason="tool_use",
        ),
        make_response(
            tool_calls=[
                ToolCall(
                    id="r3",
                    name="run_command",
                    arguments={"argv": ["pytest", "test_calc.py", "-q"]},
                )
            ],
            finish_reason="tool_use",
        ),
        make_response(content="Fixed the division bug; the test passes."),
    )
    registry = ToolRegistry()
    register_code_tools(registry, workspace=Workspace(root=broken_repo))
    response = await run_turn(
        session=Session(user_id="dev"),
        user_input="Fix divide",
        provider=provider,
        registry=registry,
        grounder=Grounder(escalation_threshold=0.55),
    )
    assert response.files_touched == ["calc.py"]
    assert response.patch_summary == ["calc.py (replaced 1 text span)"]
    assert response.verification_ran is True
    assert response.tool_calls[1].error is None
    assert (broken_repo / "calc.py").read_text(encoding="utf-8") == FIXED_CALC


async def test_read_only_turn_does_not_require_verification(broken_repo: Path) -> None:
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
    assert response.escalated is False
    assert response.completion_status == "completed"


async def test_emit_plan_appears_in_tool_calls_before_write(broken_repo: Path) -> None:
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="t0",
                    name="emit_plan",
                    arguments={
                        "steps": ["Read calc.py", "Fix divide", "Run pytest"],
                        "summary": "Fix divide bug",
                    },
                )
            ],
            finish_reason="tool_use",
        ),
        make_response(
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="write_file",
                    arguments={"path": "calc.py", "content": FIXED_CALC},
                )
            ],
            finish_reason="tool_use",
        ),
        make_response(content="Planned and applied fix."),
    )

    registry = ToolRegistry()
    register_code_tools(registry, workspace=Workspace(root=broken_repo))

    response = await run_turn(
        session=Session(),
        user_input="fix divide",
        provider=provider,
        registry=registry,
    )

    assert [tc.name for tc in response.tool_calls] == ["emit_plan", "write_file"]
    plan_call = response.tool_calls[0]
    assert isinstance(plan_call.result, dict)
    assert plan_call.result["step_count"] == 3


async def test_require_plan_before_edit_blocks_write_without_plan(broken_repo: Path) -> None:
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="write_file",
                    arguments={"path": "calc.py", "content": FIXED_CALC},
                )
            ],
            finish_reason="tool_use",
        ),
        make_response(content="Fixed without planning."),
    )

    registry = ToolRegistry()
    register_code_tools(registry, workspace=Workspace(root=broken_repo))

    response = await run_turn(
        session=Session(),
        user_input="fix divide",
        provider=provider,
        registry=registry,
        require_plan_before_edit=True,
    )

    assert response.files_touched == []
    assert response.tool_calls[0].error is not None
    assert response.tool_calls[0].error.startswith("Edit blocked:")
    assert response.escalated is True
    assert response.completion_status == "incomplete"


async def test_blocked_edit_can_recover_after_plan(broken_repo: Path) -> None:
    provider = ScriptedProvider()
    edit = ToolCall(
        id="edit", name="write_file", arguments={"path": "calc.py", "content": FIXED_CALC}
    )
    provider.script(
        make_response(tool_calls=[edit]),
        make_response(content="Fixed."),
        make_response(
            tool_calls=[ToolCall(id="plan", name="emit_plan", arguments={"steps": ["Fix divide"]})]
        ),
        make_response(tool_calls=[edit.model_copy(update={"id": "edit2"})]),
        make_response(content="Fixed after planning."),
    )
    registry = ToolRegistry()
    register_code_tools(registry, workspace=Workspace(root=broken_repo))
    response = await run_turn(
        session=Session(),
        user_input="fix divide",
        provider=provider,
        registry=registry,
        require_plan_before_edit=True,
        max_completion_retries=1,
    )
    assert response.tool_calls[0].error is not None
    assert response.tool_calls[-1].error is None
    assert response.files_touched == ["calc.py"]
    assert response.completion_status == "completed"
    assert (broken_repo / "calc.py").read_text(encoding="utf-8") == FIXED_CALC


async def test_require_plan_before_edit_allows_write_after_emit_plan(broken_repo: Path) -> None:
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="t0",
                    name="emit_plan",
                    arguments={"steps": ["Read calc.py", "Fix divide", "Run pytest"]},
                )
            ],
            finish_reason="tool_use",
        ),
        make_response(
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="write_file",
                    arguments={"path": "calc.py", "content": FIXED_CALC},
                )
            ],
            finish_reason="tool_use",
        ),
        make_response(content="Fixed with plan."),
    )

    registry = ToolRegistry()
    register_code_tools(registry, workspace=Workspace(root=broken_repo))

    response = await run_turn(
        session=Session(),
        user_input="fix divide",
        provider=provider,
        registry=registry,
        grounder=Grounder(escalation_threshold=0.55),
        require_plan_before_edit=True,
    )

    assert response.files_touched == ["calc.py"]
    assert response.escalated is False
