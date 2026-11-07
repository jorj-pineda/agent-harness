import json
from dataclasses import replace

import pytest

import evals.public_admission as admission
import evals.real_run as runner
from evals.heldout_comparison import PLAN_PATH
from evals.public_workspaces import _selection, prepare_public_cell
from harness.config import Settings
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response


def response(**kwargs):
    return make_response(prompt_tokens=10, completion_tokens=10, **kwargs)


def setup(tmp_path, ordinal=1):
    receipt, task, _ = _selection(ordinal)
    directory = tmp_path / "cell"
    prepare_public_cell(ordinal, directory)
    runtime = json.loads(PLAN_PATH.read_bytes())["runtime"]
    runtime.update(coding_prompt_profile="direct", coding_toolset="whole_file_check")
    settings = Settings(_env_file=None, **runtime)
    return (
        task,
        settings,
        admission.PublicCellWorkspace(ordinal=ordinal, directory=directory),
        receipt,
    )


def script_reference(provider, task):
    edits = [
        ToolCall(
            id=f"edit-{index}",
            name="write_file",
            arguments={"path": str(path.relative_to(task.reference)), "content": path.read_text()},
        )
        for index, path in enumerate(sorted(task.reference.rglob("*.py")))
    ]
    provider.script(
        response(tool_calls=[*edits, ToolCall(id="check", name="run_project_check", arguments={})]),
        response(content="Applied the change and ran the project check."),
    )


@pytest.mark.parametrize("ordinal", [1, 2, 3, 4])
async def test_named_public_cell_uses_shared_runtime_and_original_grading(tmp_path, ordinal):
    task, settings, cell, receipt = setup(tmp_path, ordinal)
    provider = ScriptedProvider()
    script_reference(provider, task)
    row = await runner.run_task(
        task,
        provider=provider,
        settings=settings,
        mode="harness",
        attempt=receipt.repeat,
        public_cell=cell,
    )
    assert row.passed and row.completion_status == "completed"
    assert row.runtime_config["workspace_policy"] == "prepared-public-pilot-single-use-v1"
    assert row.runtime_config["prepared_workspace_root"] == str(cell.directory / "workspace")
    evidence = row.runtime_config["public_pilot_preservation"]
    assert evidence["status"] == "checks_passed" and evidence["ordinal"] == ordinal
    assert evidence["checkpoints"][-2:] == ["before_grading", "after_grading"]
    assert (cell.directory / "admitted.json").exists()
    assert row.heldout_preservation is None


@pytest.mark.parametrize("defect", ["task", "repeat", "mode", "settings", "both"])
async def test_invalid_selection_refuses_before_marker_or_request(tmp_path, defect):
    task, settings, cell, _ = setup(tmp_path)
    provider = ScriptedProvider()
    kwargs = {"mode": "harness", "attempt": 1, "public_cell": cell}
    if defect == "task":
        task = replace(task, request=task.request + " changed")
    elif defect == "repeat":
        kwargs["attempt"] = 2
    elif defect == "mode":
        kwargs["mode"] = "minimal"
    elif defect == "settings":
        settings.max_tool_iterations = 7
    else:
        kwargs["prepared_directory"] = cell.directory
    with pytest.raises(ValueError):
        await runner.run_task(task, provider=provider, settings=settings, **kwargs)
    assert provider.calls == [] and not (cell.directory / "admitted.json").exists()


async def test_midrun_policy_drift_stops_before_second_request(tmp_path):
    task, settings, cell, _ = setup(tmp_path)

    class DriftProvider(ScriptedProvider):
        async def chat(self, *args, **kwargs):
            response = await super().chat(*args, **kwargs)
            settings.max_tool_iterations = 7
            return response

    provider = DriftProvider()
    provider.script(response(tool_calls=[ToolCall(id="tree", name="tree", arguments={})]))
    row = await runner.run_task(
        task, provider=provider, settings=settings, mode="harness", public_cell=cell
    )
    assert len(provider.calls) == 1 and not row.passed
    assert row.termination == "error:PublicPilotDriftError"
    evidence = row.runtime_config["public_pilot_preservation"]
    assert evidence["status"] == "failed" and evidence["failed_checkpoint"] == "before_request"


def test_observed_failure_remains_latched_after_restoration(tmp_path, monkeypatch):
    task, settings, cell, _ = setup(tmp_path)
    guard = admission.PublicTaskGuard(task, cell, mode="harness", attempt=1, settings=settings)
    original = admission._selection
    monkeypatch.setattr(
        admission, "_selection", lambda ordinal: (_ for _ in ()).throw(ValueError("drift"))
    )
    with pytest.raises(admission.PublicPilotDriftError):
        guard()
    monkeypatch.setattr(admission, "_selection", original)
    with pytest.raises(admission.PublicPilotDriftError, match="previously failed"):
        guard()


async def test_after_grading_drift_invalidates_acceptance_but_retains_grader(tmp_path, monkeypatch):
    task, settings, cell, _ = setup(tmp_path)
    provider = ScriptedProvider()
    script_reference(provider, task)
    original_grade = runner.acceptance_check
    original_select = admission._selection
    changed = False

    def select(ordinal):
        if changed:
            raise ValueError("grading drift")
        return original_select(ordinal)

    async def grade(*args, **kwargs):
        nonlocal changed
        result = await original_grade(*args, **kwargs)
        changed = True
        return result

    monkeypatch.setattr(admission, "_selection", select)
    monkeypatch.setattr(runner, "acceptance_check", grade)
    row = await runner.run_task(
        task, provider=provider, settings=settings, mode="harness", public_cell=cell
    )
    assert not row.passed and row.acceptance_exit_code == 0
    assert row.termination == "error:PublicPilotDriftError"
    assert row.runtime_config["public_pilot_preservation"]["failed_checkpoint"] == "after_grading"


async def test_default_disposable_policy_and_metadata_remain_unchanged(tmp_path):
    task, settings, cell, _ = setup(tmp_path)
    provider = ScriptedProvider()
    script_reference(provider, task)
    row = await runner.run_task(task, provider=provider, settings=settings, mode="harness")
    assert row.passed and row.runtime_config["workspace_policy"] == "disposable-v1"
    assert "public_pilot_preservation" not in row.runtime_config
    assert row.runtime_config["prepared_workspace_root"] is None
    assert not (cell.directory / "admitted.json").exists()


async def test_old_prepared_directory_still_refuses_public_task(tmp_path):
    task, settings, cell, _ = setup(tmp_path)
    provider = ScriptedProvider()
    with pytest.raises(ValueError, match="registered held-out"):
        await runner.run_task(
            task,
            provider=provider,
            settings=settings,
            mode="harness",
            prepared_directory=cell.directory,
        )
    assert provider.calls == [] and not (cell.directory / "admitted.json").exists()


async def test_before_grading_drift_skips_acceptance(tmp_path, monkeypatch):
    task, settings, cell, _ = setup(tmp_path)

    class DriftAfterAnswer(ScriptedProvider):
        async def chat(self, *args, **kwargs):
            result = await super().chat(*args, **kwargs)
            if len(self.calls) == 2:
                settings.max_tool_iterations = 7
            return result

    provider = DriftAfterAnswer()
    script_reference(provider, task)

    async def forbidden(*args, **kwargs):
        pytest.fail("graded after public preservation failed")

    monkeypatch.setattr(runner, "acceptance_check", forbidden)
    row = await runner.run_task(
        task, provider=provider, settings=settings, mode="harness", public_cell=cell
    )
    assert not row.passed and row.termination == "error:PublicPilotDriftError"
    assert row.runtime_config["public_pilot_preservation"]["failed_checkpoint"] == "before_grading"
