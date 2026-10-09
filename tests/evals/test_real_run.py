from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import real_run
from evals.real_run import load_tasks, run_task, summarize, summary_markdown, validate_task
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
    responses = [r for r in result.turn_trace if r["kind"] == "response"]
    assert len(responses) == 4
    assert responses[0]["content"] == ""
    assert responses[0]["tool_calls"][0]["id"] == "read"
    tool_results = [r for r in result.turn_trace if r["kind"] == "tool_result"]
    assert [r["message"]["tool_call_id"] for r in tool_results] == ["read", "edit", "check"]
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


@pytest.mark.parametrize("toolset", ["full", "whole_file"])
async def test_minimal_baseline_uses_same_tools_and_budget(toolset: str) -> None:
    task = load_tasks()[0]
    settings = Settings(_env_file=None, max_tool_iterations=3, coding_toolset=toolset)
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
    specs = harness_provider.calls[0][1] or []
    assert harness.runtime_config["coding_toolset"] == toolset
    assert harness.runtime_config["tool_specs"] == [s.model_dump(mode="json") for s in specs]
    assert ("replace_text" in {s.name for s in specs}) == (toolset == "full")
    if toolset == "whole_file":
        assert "replace_text" not in harness_provider.calls[0][0][0].content
        assert "write_file" in harness_provider.calls[0][0][0].content
        assert harness.prompt_version == "coding-v2-whole-file-v1"
        assert minimal.prompt_version == "minimal-v1"


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
    assert result.prompt_tokens is None
    assert result.completion_tokens is None


async def test_summary_counts_acceptance_false_completion_and_terminations() -> None:
    task = load_tasks()[0]
    fixed = "def divide(a: float, b: float) -> float:\n    return a / b\n"
    solved_provider = ScriptedProvider()
    solved_provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "calc.py", "content": fixed},
                )
            ],
            prompt_tokens=10,
            completion_tokens=5,
        ),
        make_response(content="Fixed.", prompt_tokens=20, completion_tokens=5),
    )
    claimed_provider = ScriptedProvider()
    claimed_provider.script(make_response(content="Fixed it."))
    settings = Settings(_env_file=None)
    results = [
        await run_task(task, provider=solved_provider, settings=settings, mode="harness"),
        await run_task(
            task, provider=claimed_provider, settings=settings, mode="harness", attempt=2
        ),
    ]
    summary = summarize(results)
    assert list(summary) == ["harness"]
    harness = summary["harness"]
    assert harness["accepted"] == 1
    assert harness["attempts"] == 2
    assert harness["per_task"] == {"divide_zero": {"accepted": 1, "attempts": 2}}
    assert harness["terminations"] == {"acceptance_failed": 1, "accepted": 1}
    assert harness["false_completion"] == 1
    assert harness["median_total_tokens"] == 40
    assert harness["unreported_token_attempts"] == 1
    assert [r.attempt for r in results] == [1, 2]
    assert "| divide_zero | 1/2 |" in summary_markdown(summary)


def test_cli_repeats_both_modes_and_alternates_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = ScriptedProvider()
    provider.script(*[make_response(content="not fixed") for _ in range(4)])
    monkeypatch.setattr(real_run, "build_configured_provider", lambda name, settings: provider)
    monkeypatch.setattr(real_run, "get_settings", lambda: Settings(_env_file=None))
    report = tmp_path / "report.json"
    assert (
        real_run.main(
            ["--task", "divide_zero", "--mode", "both", "--repeats", "2", "--report", str(report)]
        )
        == 0
    )
    data = json.loads(report.read_text(encoding="utf-8"))
    assert [(r["mode"], r["attempt"]) for r in data["results"]] == [
        ("harness", 1),
        ("minimal", 1),
        ("minimal", 2),
        ("harness", 2),
    ]
    assert data["summary"]["harness"]["attempts"] == 2
    assert data["sampling"] == {"temperature": 0.0}
    assert data["trace_format"] == "normalized-turn-v1"
    assert all(
        [r["kind"] for r in row["turn_trace"]] == ["request", "response"] for row in data["results"]
    )
    assert set(data["harness_revision"]) == {"commit", "dirty"}


async def test_real_evaluator_records_opt_in_early_recovery() -> None:
    provider = ScriptedProvider()
    argv = ["python3", "-m", "pytest", "test_visible.py"]
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(id="bad", name="run_command", arguments={"argv": json.dumps(argv)})
            ]
        ),
        make_response(
            tool_calls=[ToolCall(id="valid", name="run_command", arguments={"argv": argv})]
        ),
        make_response(content="The initial check fails; the fixture is unchanged."),
    )
    result = await run_task(
        load_tasks()[0],
        provider=provider,
        mode="minimal",
        settings=Settings(_env_file=None, recover_string_argv=True, project_check_argv=argv),
    )
    assert result.runtime_config["recover_string_argv"] is True
    assert result.passed is False
    assert result.modified == ()
    recovery = [r for r in result.turn_trace if r["kind"] == "recovery"]
    assert len(recovery) == 1
    assert recovery[0]["iteration"] == 0
    assert (
        '"argv": ["python3", "-m", "pytest", "test_visible.py"]'
        in recovery[0]["message"]["content"]
    )
    assert result.tool_trace[0]["error"]
    assert result.tool_trace[1]["result"]["exit_code"] == 1
