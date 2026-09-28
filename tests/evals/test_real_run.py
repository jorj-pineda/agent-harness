from __future__ import annotations

from pathlib import Path

import pytest

from evals.real_run import load_tasks, run_task, validate_task
from harness.config import Settings
from harness.prompts import BASE_SYSTEM_PROMPT, PROMPT_VERSION
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response


@pytest.mark.parametrize(
    "task_id",
    [
        "divide_zero",
        "slugify",
        "stable_dedupe",
        "decimal_total",
        "parse_flags",
    ],
)
async def test_each_task_fails_initially_and_reference_passes(task_id: str) -> None:
    task = next(task for task in load_tasks() if task.id == task_id)
    assert not task.acceptance.is_relative_to(task.fixture)
    assert not task.reference.is_relative_to(task.fixture)
    await validate_task(task)


async def test_real_tool_edit_is_scored_from_final_files(tmp_path: Path) -> None:
    task = load_tasks()[0]
    original = (task.fixture / "calc.py").read_bytes()
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="read",
                    name="read_file",
                    arguments={
                        "path": "calc.py",
                    },
                )
            ]
        ),
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={
                        "path": "calc.py",
                        "content": "def divide(a: float, b: float) -> float:\n    return a / b\n",
                    },
                )
            ]
        ),
        make_response(
            tool_calls=[
                ToolCall(
                    id="check",
                    name="run_command",
                    arguments={
                        "argv": ["pytest", "-q"],
                    },
                )
            ]
        ),
        make_response(content="Fixed division."),
    )
    result = await run_task(
        task,
        provider=provider,
        settings=Settings(_env_file=None),
        mode="harness",
    )
    assert result.passed is True
    assert result.termination == "accepted"
    assert result.modified == ("calc.py",)
    assert result.added == ()
    assert "-    return 0.0" in result.diff
    assert "+    return a / b" in result.diff
    assert [call["name"] for call in result.tool_trace] == [
        "read_file",
        "write_file",
        "run_command",
    ]
    assert result.tool_trace[-1]["result"]["success"] is True
    assert result.prompt_version == PROMPT_VERSION
    assert len(result.source_revision) == 64
    assert (task.fixture / "calc.py").read_bytes() == original
    assert provider.calls[0][0][0].content.startswith(BASE_SYSTEM_PROMPT)


async def test_editing_visible_test_does_not_change_acceptance() -> None:
    task = load_tasks()[0]
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="cheat",
                    name="write_file",
                    arguments={
                        "path": "test_visible.py",
                        "content": "def test_nothing(): pass\n",
                    },
                )
            ]
        ),
        make_response(content="done"),
    )
    result = await run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert result.passed is False
    assert result.termination == "acceptance_failed"
    assert result.modified == ("test_visible.py",)


async def test_minimal_baseline_uses_same_tools_and_budget() -> None:
    task = load_tasks()[0]
    settings = Settings(_env_file=None, max_tool_iterations=3)
    harness_provider = ScriptedProvider()
    minimal_provider = ScriptedProvider()
    harness_provider.script(make_response(content="not fixed"))
    minimal_provider.script(make_response(content="not fixed"))
    harness = await run_task(task, provider=harness_provider, settings=settings, mode="harness")
    minimal = await run_task(task, provider=minimal_provider, settings=settings, mode="minimal")
    assert not harness.passed and not minimal.passed
    assert harness.source_revision == minimal.source_revision
    assert harness.runtime_config == minimal.runtime_config
    assert harness_provider.calls[0][1] == minimal_provider.calls[0][1]
    assert harness_provider.calls[0][0][0].content != minimal_provider.calls[0][0][0].content


async def test_tool_budget_termination_is_separate_from_acceptance() -> None:
    task = load_tasks()[0]
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(id="read", name="read_file", arguments={"path": "calc.py"}),
                ToolCall(id="list", name="list_dir", arguments={"path": "."}),
            ]
        )
    )
    result = await run_task(
        task,
        provider=provider,
        settings=Settings(_env_file=None, max_tool_calls_per_turn=1),
        mode="harness",
    )
    assert result.passed is False
    assert result.termination == "budget_exhausted"
    assert result.completion_status == "budget_exhausted"
    assert result.tool_trace[0]["error"] is None
    assert "not executed" in result.tool_trace[1]["error"]
