import json
from pathlib import Path

import pytest

from evals.real_run import acceptance_check, load_tasks, run_task
from harness.config import Settings
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tools.process import run_process
from workspace import disposable_workspace


@pytest.mark.parametrize("solution", ["both", "parser_only", "client_only"])
async def test_both_public_interfaces_must_work_in_final_artifact(solution):
    task = next(task for task in load_tasks() if task.id == "retry_after_seconds")
    before = {path.name: path.read_bytes() for path in task.fixture.iterdir() if path.is_file()}
    files = (
        ["retry_after.py", "retry_client.py"]
        if solution == "both"
        else ["retry_after.py" if solution == "parser_only" else "retry_client.py"]
    )
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id=f"edit-{index}",
                    name="write_file",
                    arguments={"path": name, "content": (task.reference / name).read_text()},
                )
                for index, name in enumerate(files)
            ]
        ),
        make_response(content="Feature complete"),
    )
    result = await run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert result.passed is (solution == "both")
    assert result.category == "feature" and result.acceptance_policy == "owned-pytest-v1"
    assert result.acceptance_exit_code == (0 if solution == "both" else 1)
    assert before == {
        path.name: path.read_bytes() for path in task.fixture.iterdir() if path.is_file()
    }


@pytest.mark.parametrize("mode", ["harness", "minimal"])
async def test_archived_feature_failures_remain_visible_within_grader_cap(tmp_path, mode):
    # Offline artifact replay, not another model attempt or a repaired patch.
    report = (
        Path(__file__).parents[2] / "evals/results/2026-10-17-gemma4-12b-development-envelope.json"
    )
    original_report = report.read_bytes()
    row = next(
        row
        for row in json.loads(original_report)["results"]
        if row["task_id"] == "retry_after_seconds" and row["mode"] == mode
    )
    task = next(task for task in load_tasks() if task.id == row["task_id"])
    original_fixture = {
        path.name: path.read_bytes() for path in task.fixture.iterdir() if path.is_file()
    }
    patch = tmp_path / "saved.patch"
    patch.write_text(row["diff"])
    with disposable_workspace(task.fixture) as copy:
        applied = await run_process(
            ["git", "apply", str(patch)],
            cwd=copy.workspace.root,
            timeout_seconds=10,
            output_limit=12_000,
        )
        assert applied.exit_code == 0, applied.stderr
        graded = await acceptance_check(task, copy.workspace.root)
    assert graded.exit_code == row["acceptance_exit_code"] == 1
    assert not graded.stdout_truncated and not graded.stderr_truncated
    output = graded.stdout + graded.stderr
    assert len(output) <= 12_000
    for case in (
        "leading-zeros-5000",
        "leading-zeros-5000-plus-one",
        "arabic-indic-digits",
        "fullwidth-digits",
    ):
        assert f"[{case}]" in output
    assert "4 failed, 24 passed" in output
    assert report.read_bytes() == original_report
    assert {
        path.name: path.read_bytes() for path in task.fixture.iterdir() if path.is_file()
    } == original_fixture
