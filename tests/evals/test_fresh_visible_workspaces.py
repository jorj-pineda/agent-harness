"""Scripted real-file fresh-arm evaluation; no native lifecycle or model quality."""

import shutil
from dataclasses import replace

import pytest

from evals import fresh_visible_workspaces as fresh
from evals import real_run as runner
from evals.public_envelope import public_settings
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tests.evals.test_fresh_visible_tasks import PARTIAL


def response(**kwargs):
    return make_response(prompt_tokens=10, completion_tokens=5, **kwargs)


@pytest.mark.parametrize("ordinal", range(1, 13))
def test_all_cells_are_fresh_arm_specific_and_single_use(tmp_path, ordinal):
    directory = tmp_path / "cell"
    receipt = fresh.prepare_fresh_cell(ordinal, directory)
    root = fresh.verify_fresh_cell(directory, ordinal=ordinal)
    cell, task, visible, _ = fresh.selected_cell(ordinal)
    assert receipt.cell == cell and (root / "test_visible.py").read_bytes() == visible
    for path in task.fixture.iterdir():
        if path.name != "test_visible.py":
            assert (root / path.name).read_bytes() == path.read_bytes()
    with fresh.admit_fresh_cell(directory, ordinal=ordinal) as copied:
        assert not copied.changes().modified
    assert root.exists() and (directory / "admitted.json").is_file()
    with pytest.raises(ValueError), fresh.admit_fresh_cell(directory, ordinal=ordinal):
        pytest.fail("Replayed cell")
    with pytest.raises(FileExistsError):
        fresh.prepare_fresh_cell(ordinal, directory)


def setup(tmp_path, ordinal=2):
    directory = tmp_path / "cell"
    receipt = fresh.prepare_fresh_cell(ordinal, directory)
    _, task, _, _ = fresh.selected_cell(ordinal)
    return task, fresh.FreshCellWorkspace(ordinal=ordinal, directory=directory), receipt


def script(provider, task, *, partial=False, weaken=False):
    name, content = PARTIAL[task.id]
    if not partial:
        content = (task.reference / name).read_text()
    calls = [ToolCall(id="write", name="write_file", arguments={"path": name, "content": content})]
    if weaken:
        calls.append(
            ToolCall(
                id="weaken",
                name="write_file",
                arguments={
                    "path": "test_visible.py",
                    "content": (task.fixture / "test_visible.py").read_text(),
                },
            )
        )
    calls.append(ToolCall(id="check", name="run_project_check", arguments={}))
    provider.script(
        response(tool_calls=calls),
        response(content="Observed check result."),
        response(content="Observed check result."),
    )


@pytest.mark.parametrize("ordinal", [1, 2, 3, 4, 5, 6, 9, 12])
async def test_shared_runtime_reference_succeeds_and_arm_setup_is_not_model_change(
    tmp_path, ordinal
):
    task, cell, receipt = setup(tmp_path, ordinal)
    provider = ScriptedProvider()
    script(provider, task)
    row = await runner.run_task(
        task,
        provider=provider,
        settings=public_settings(),
        mode="harness",
        attempt=receipt.cell.repeat,
        fresh_cell=cell,
    )
    assert row.passed and row.completion_status == "completed"
    assert "test_visible.py" not in row.modified and "test_visible.py" not in row.diff
    assert row.runtime_config["workspace_policy"] == "prepared-fresh-visible-single-use-v1"
    assert row.runtime_config["prepared_workspace_root"] == str(cell.directory / "workspace")
    evidence = row.runtime_config["fresh_visible_preservation"]
    assert evidence["status"] == "checks_passed"
    assert evidence["checkpoints"][-2:] == ["before_grading", "after_grading"]
    assert "public_feedback_preservation" not in row.runtime_config


@pytest.mark.parametrize("ordinal", range(1, 7))
async def test_partial_patch_acceptance_independent_of_visible_arm(tmp_path, ordinal):
    task, cell, receipt = setup(tmp_path, ordinal)
    provider = ScriptedProvider()
    script(provider, task, partial=True)
    row = await runner.run_task(
        task, provider=provider, settings=public_settings(), mode="harness", fresh_cell=cell
    )
    assert not row.passed and row.acceptance_exit_code != 0
    assert row.verification_status == ("passed" if receipt.cell.arm == "weak" else "failed")


@pytest.mark.parametrize("ordinal", [2, 3, 6])
async def test_weakened_stronger_tests_are_reported_but_cannot_redefine_acceptance(
    tmp_path, ordinal
):
    task, cell, _ = setup(tmp_path, ordinal)
    provider = ScriptedProvider()
    script(provider, task, partial=True, weaken=True)
    row = await runner.run_task(
        task, provider=provider, settings=public_settings(), mode="harness", fresh_cell=cell
    )
    assert not row.passed and row.verification_status == "passed"
    assert "test_visible.py" in row.modified and "test_visible.py" in row.diff


@pytest.mark.parametrize(
    "defect", ["request", "repeat", "mode", "settings", "prepared", "feedback"]
)
async def test_wrong_task_or_policy_refuses_before_provider_and_marker(tmp_path, defect):
    task, cell, _ = setup(tmp_path)
    settings = public_settings()
    provider = ScriptedProvider()
    kwargs = {"mode": "harness", "attempt": 1, "fresh_cell": cell}
    if defect == "request":
        task = replace(task, request=task.request + " changed")
    elif defect == "repeat":
        kwargs["attempt"] = 2
    elif defect == "mode":
        kwargs["mode"] = "minimal"
    elif defect == "settings":
        settings.max_tool_iterations = 7
    elif defect == "prepared":
        kwargs["prepared_directory"] = cell.directory
    else:
        from evals.public_feedback_workspaces import FeedbackCellWorkspace

        kwargs["feedback_cell"] = FeedbackCellWorkspace(ordinal=2, directory=cell.directory)
    with pytest.raises(ValueError):
        await runner.run_task(task, provider=provider, settings=settings, **kwargs)
    assert not provider.calls and not (cell.directory / "admitted.json").exists()


@pytest.mark.parametrize(
    "source", ["selection", "catalog", "initial", "acceptance", "reference", "stronger"]
)
async def test_source_drift_stops_next_request_and_latches(tmp_path, monkeypatch, source):
    cloned = tmp_path / "catalog"
    shutil.copytree(fresh.ROOT, cloned)
    monkeypatch.setattr(fresh, "ROOT", cloned)
    task, cell, _ = setup(tmp_path)
    targets = {
        "selection": cloned / "selection.json",
        "catalog": cloned / "tasks.yaml",
        "initial": task.fixture / "lookup.py",
        "acceptance": task.acceptance / "test_acceptance.py",
        "reference": task.reference / "lookup.py",
        "stronger": cloned / "stronger" / task.id / "test_visible.py",
    }

    class Drift(ScriptedProvider):
        async def chat(self, *args, **kwargs):
            response = await super().chat(*args, **kwargs)
            target = targets[source]
            target.write_text(target.read_text() + "\n# drift\n")
            return response

    provider = Drift()
    provider.script(response(tool_calls=[ToolCall(id="tree", name="tree", arguments={})]))
    row = await runner.run_task(
        task, provider=provider, settings=public_settings(), mode="harness", fresh_cell=cell
    )
    assert len(provider.calls) == 1 and not row.passed
    assert row.termination == "error:FreshDriftError"
    assert row.runtime_config["fresh_visible_preservation"]["status"] == "failed"


@pytest.mark.parametrize("mutation", ["implementation", "test", "extra", "mode", "receipt", "cell"])
def test_pre_admission_changes_refuse(tmp_path, mutation):
    _, cell, _ = setup(tmp_path)
    ordinal = 2
    if mutation == "implementation":
        (cell.directory / "workspace/lookup.py").write_text("# drift\n")
    elif mutation == "test":
        (cell.directory / "workspace/test_visible.py").write_text("# drift\n")
    elif mutation == "extra":
        (cell.directory / "extra").touch()
    elif mutation == "mode":
        cell.directory.chmod(0o755)
    elif mutation == "receipt":
        (cell.directory / "prepared.json").write_text("{}")
    else:
        ordinal = 1
    with pytest.raises(ValueError), fresh.admit_fresh_cell(cell.directory, ordinal=ordinal):
        pytest.fail("Drifted admission")
    assert not (cell.directory / "admitted.json").exists()


@pytest.mark.parametrize("ordinal", [True, 0, 13, "1"])
def test_invalid_ordinals_refuse_without_allocation(tmp_path, ordinal):
    with pytest.raises(ValueError):
        fresh.prepare_fresh_cell(ordinal, tmp_path / "cell")
    assert not (tmp_path / "cell").exists()


def test_git_repository_refused(tmp_path):
    (tmp_path / ".git").mkdir()
    with pytest.raises(ValueError, match="outside Git"):
        fresh.prepare_fresh_cell(1, tmp_path / "cell")
    assert not (tmp_path / "cell").exists()


async def test_after_grading_policy_drift_keeps_grade_but_invalidates_acceptance(
    tmp_path, monkeypatch
):
    task, cell, _ = setup(tmp_path)
    settings = public_settings()
    provider = ScriptedProvider()
    script(provider, task)
    original = runner.acceptance_check

    async def drift(*args, **kwargs):
        result = await original(*args, **kwargs)
        settings.max_tool_iterations = 7
        return result

    monkeypatch.setattr(runner, "acceptance_check", drift)
    row = await runner.run_task(
        task, provider=provider, settings=settings, mode="harness", fresh_cell=cell
    )
    assert row.acceptance_exit_code == 0 and not row.passed
    assert row.termination == "error:FreshDriftError"
    assert row.runtime_config["fresh_visible_preservation"]["failed_checkpoint"] == "after_grading"


def test_failed_guard_remains_latched_after_source_restoration(tmp_path, monkeypatch):
    task, cell, _ = setup(tmp_path)
    guard = fresh.FreshTaskGuard(task, cell, mode="harness", attempt=1, settings=public_settings())
    selected = fresh._selection
    monkeypatch.setattr(fresh, "_selection", lambda _: (_ for _ in ()).throw(ValueError("drift")))
    with pytest.raises(fresh.FreshDriftError):
        guard()
    monkeypatch.setattr(fresh, "_selection", selected)
    with pytest.raises(fresh.FreshDriftError, match="previously failed"):
        guard()


def test_post_marker_failure_is_retained_and_consumes_cell(tmp_path, monkeypatch):
    _, cell, _ = setup(tmp_path)
    selected = fresh._selection
    calls = 0

    def fail_after_marker(ordinal):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("post-marker drift")
        return selected(ordinal)

    monkeypatch.setattr(fresh, "_selection", fail_after_marker)
    with (
        pytest.raises(ValueError, match="post-marker"),
        fresh.admit_fresh_cell(cell.directory, ordinal=2),
    ):
        pytest.fail("Failed admission yielded")
    assert (cell.directory / "admitted.json").is_file()
    assert (cell.directory / "workspace/lookup.py").is_file()
