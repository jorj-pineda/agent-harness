from __future__ import annotations

import asyncio
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
    [task.id for task in load_tasks()],
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
        settings=Settings(_env_file=None, prompt_estimate_bytes_per_token=4.5),
        mode="harness",
    )
    assert result.passed is True
    assert result.acceptance_output_truncated is False
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
    assert result.runtime_config["model_profile"] is None
    assert result.runtime_config["max_request_output_tokens"] == 0
    assert result.runtime_config["effective_max_context_tokens"] == 0
    assert result.runtime_config["context_limit_policy"] == "configured-min-v1"
    assert result.runtime_config["prompt_estimate_bytes_per_token"] == 4.5
    assert result.turn_trace[0]["footprint"]["prompt_estimate_bytes_per_token"] == 4.5
    assert result.runtime_config["max_workspace_diff_bytes"] == 64_000
    assert result.runtime_config["max_diff_snapshot_bytes"] == 2_000_000
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
@pytest.mark.parametrize("profile", ["legacy", "compact"])
async def test_minimal_baseline_uses_same_tools_and_budget(toolset: str, profile: str) -> None:
    task = load_tasks()[0]
    settings = Settings(
        _env_file=None, max_tool_iterations=3, coding_toolset=toolset, coding_prompt_profile=profile
    )
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
    assert harness.runtime_config["coding_prompt_profile"] == profile
    assert harness.runtime_config["tool_specs"] == [s.model_dump(mode="json") for s in specs]
    assert ("replace_text" in {s.name for s in specs}) == (toolset == "full")
    assert "semantic_search" not in {s.name for s in specs}
    if toolset == "whole_file":
        assert "replace_text" not in harness_provider.calls[0][0][0].content
        assert "write_file" in harness_provider.calls[0][0][0].content
    assert minimal.prompt_version == "minimal-v1"
    if profile == "compact":
        assert "senior" not in harness_provider.calls[0][0][0].content
        assert "ENABLE_SUPPORT_TOOLS" not in harness_provider.calls[0][0][0].content
        assert harness.prompt_version == (
            "coding-v3-compact-whole-file-v1" if toolset == "whole_file" else "coding-v3-compact-v1"
        )
    else:
        assert harness.prompt_version == (
            "coding-v2-whole-file-canonical-memory-v1"
            if toolset == "whole_file"
            else "coding-v2-canonical-memory-v1"
        )


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
    assert harness["per_task"] == {
        "divide_zero": {"accepted": 1, "completed_and_accepted": 1, "attempts": 2}
    }
    assert harness["completed_and_accepted"] == 1
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
    assert data["trace_format"] == "normalized-turn-v2"
    assert all(
        [r["kind"] for r in row["turn_trace"]] == ["request", "response"] for row in data["results"]
    )
    assert set(data["harness_revision"]) == {"commit", "dirty"}


async def test_go_binding_preserves_observation_and_real_tool_acceptance():
    import httpx

    from providers import compatible_transport_options, create_chat_provider

    captured = []
    task = load_tasks()[0]
    replies = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "read",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": "calc.py"})},
                }
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "write",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps(
                            {
                                "path": "calc.py",
                                "content": "def divide(a: float, b: float) -> float:\n    return a / b\n",
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "check",
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": json.dumps(
                            {"argv": ["python", "-m", "pytest", "-q", "test_visible.py"]}
                        ),
                    },
                }
            ],
        },
        {"role": "assistant", "content": "Fixed and checked."},
    ]

    def handler(request):
        captured.append(request)
        message = replies[len(captured) - 1]
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            },
        )

    settings = Settings(
        _env_file=None,
        openai_compatible_base_url="https://fixture.example/v1",
        openai_compatible_model="fixture",
        openai_compatible_api_key="fixture-private-key",
        openai_compatible_transport="opencode_go",
        require_verification_before_finish=True,
        project_check_argv=["python", "-m", "pytest", "-q", "test_visible.py"],
    )
    provider = create_chat_provider(
        "openai",
        api_key="fixture-private-key",
        model="fixture",
        provider_name="openai_compatible",
        base_url=settings.openai_compatible_base_url,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **compatible_transport_options(settings.openai_compatible_transport),
    )
    original = (task.fixture / "calc.py").read_bytes()
    try:
        result = await run_task(task, provider=provider, settings=settings, mode="harness")
    finally:
        await provider.aclose()
    assert result.passed and result.completion_status == "completed"
    assert result.verification_status == "passed"
    assert result.prompt_tokens == 40 and result.completion_tokens == 4
    assert len(captured) == 4
    assert len({request.headers["x-opencode-session"] for request in captured}) == 1
    assert result.runtime_config["openai_compatible_transport"] == "opencode_go"
    assert (task.fixture / "calc.py").read_bytes() == original


async def test_prevented_request_footprint_survives_real_task_report(tmp_path):
    task = load_tasks()[0]
    original = (task.fixture / "calc.py").read_bytes()
    provider = ScriptedProvider()
    result = await run_task(
        task,
        provider=provider,
        settings=Settings(_env_file=None, max_context_tokens=1),
        mode="harness",
    )
    assert result.completion_status == "budget_exhausted"
    assert result.passed is False
    assert provider.calls == [] and result.tool_trace == []
    assert result.prompt_tokens == 0 and result.completion_tokens == 0
    assert len(result.turn_trace) == 1
    stop = result.turn_trace[0]
    assert stop["kind"] == "budget_stop"
    assert stop["footprint"]["estimated_prompt_tokens"] > stop["max_context_tokens"]
    assert stop["footprint"]["tool_schema_estimate_tokens"] > 0
    path = tmp_path / "report.json"
    real_run.write_report(path, "fixture-start", [result])
    report = json.loads(path.read_text())
    assert report["trace_format"] == "normalized-turn-v2"
    assert report["results"][0]["turn_trace"][0] == stop
    assert (task.fixture / "calc.py").read_bytes() == original


@pytest.mark.parametrize("failure", [RuntimeError, TimeoutError, asyncio.CancelledError])
@pytest.mark.parametrize("has_prefix", [False, True])
async def test_observation_failure_invalidates_totals_but_keeps_prior_response(failure, has_prefix):
    class FailingProvider(ScriptedProvider):
        async def chat(self, *args, **kwargs):
            if self._queue:
                return await super().chat(*args, **kwargs)
            raise failure("private exception detail")

    provider = FailingProvider()
    if has_prefix:
        provider.script(make_response(content="prefix", prompt_tokens=10, completion_tokens=2))
    observed = real_run.ObservedProvider(provider)
    if has_prefix:
        response = await observed.chat([])
        assert response.usage.prompt_tokens == 10 and observed.prompt_tokens == 10
    with pytest.raises(failure):
        await observed.chat([])
    assert observed.prompt_tokens is None and observed.completion_tokens is None


async def test_failed_request_after_read_reports_unknown_usage_without_hiding_prefix():
    class FailingProvider(ScriptedProvider):
        async def chat(self, *args, **kwargs):
            if self._queue:
                return await super().chat(*args, **kwargs)
            raise RuntimeError("private exception detail")

    provider = FailingProvider()
    provider.script(
        make_response(
            tool_calls=[ToolCall(id="read", name="read_file", arguments={"path": "calc.py"})],
            prompt_tokens=10,
            completion_tokens=2,
        )
    )
    result = await run_task(
        load_tasks()[0], provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert result.termination == "error:RuntimeError"
    assert result.prompt_tokens is None and result.completion_tokens is None
    assert len(result.tool_trace) == 1 and result.tool_trace[0]["result"] is not None
    prefix = [r for r in result.turn_trace if r["kind"] == "response"]
    assert prefix[0]["usage"] == {"prompt_tokens": 10, "completion_tokens": 2}
    summary = summarize([result])["harness"]
    assert summary["unreported_token_attempts"] == 1 and summary["median_total_tokens"] is None
    assert "private exception detail" not in json.dumps(result.turn_trace)


async def test_real_task_runtime_records_active_memory_source(monkeypatch) -> None:
    from uuid import UUID

    from memory import FactStore

    captured = []
    original = FactStore.add

    def record_source(self, user_id, fact, source_turn_id=None, *, workspace_root=None):
        captured.append((user_id, source_turn_id))
        return original(self, user_id, fact, source_turn_id, workspace_root=workspace_root)

    monkeypatch.setattr(FactStore, "add", record_source)
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="remember", name="remember_fact", arguments={"fact": "fixture convention"}
                )
            ]
        ),
        make_response(content="Inspected"),
    )
    result = await run_task(
        load_tasks()[0], provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert result.passed is False  # Unedited fixture; source metadata is not acceptance.
    assert len(captured) == 1 and captured[0][0] == "eval"
    assert captured[0][1] is not None
    UUID(captured[0][1])


async def test_accepted_iteration_stop_stays_separate_from_completed_workflow(tmp_path):
    task = load_tasks()[0]
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={
                        "path": "calc.py",
                        "content": (task.reference / "calc.py").read_text(),
                    },
                )
            ]
        )
    )
    result = await run_task(
        task,
        provider=provider,
        settings=Settings(_env_file=None, max_tool_iterations=1),
        mode="minimal",
    )
    assert result.passed and result.completion_status == "budget_exhausted"
    assert result.verification_status == "not_run"
    counts = {"accepted": 1, "completed_and_accepted": 0, "attempts": 1}
    summary = summarize([result])
    assert summary["minimal"]["per_task"][task.id] == counts
    assert summary["minimal"]["per_category"]["bugfix"] == counts
    assert summary["minimal"]["completed_and_accepted"] == 0
    rendered = summary_markdown(summary)
    assert "Artifact acceptance (independent final-state checks)" in rendered
    assert "| divide_zero | 1/1 |" in rendered
    assert "| Completed and accepted | 0/1 |" in rendered
    report = tmp_path / "report.json"
    real_run.write_report(report, "fixture-start", [result])
    saved = json.loads(report.read_text())
    assert saved["summary"] == summary
    assert saved["results"][0]["completion_status"] == "budget_exhausted"
