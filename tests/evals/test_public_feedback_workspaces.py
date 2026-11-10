from dataclasses import replace

import pytest

import evals.public_feedback_workspaces as feedback
import evals.real_run as runner
from evals.public_envelope import public_settings
from evals.public_feedback import visible_check_bytes
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response


def response(**kwargs):
    return make_response(prompt_tokens=10, completion_tokens=10, **kwargs)


@pytest.mark.parametrize("ordinal", range(1, 9))
def test_prepare_and_single_use_admission_select_exact_arm(tmp_path, ordinal):
    directory = tmp_path / "cell"
    receipt = feedback.prepare_feedback_cell(ordinal, directory)
    root = feedback.verify_feedback_cell(directory, ordinal=ordinal)
    assert (root / "test_visible.py").read_bytes() == visible_check_bytes(receipt.cell.arm)
    assert (root / "pagination.py").read_bytes() == (
        runner.TASKS_ROOT / "fixtures/paginate_items/pagination.py"
    ).read_bytes()
    with feedback.admit_feedback_cell(directory, ordinal=ordinal) as copy:
        assert not copy.changes().modified
    assert root.exists() and (directory / "admitted.json").exists()
    with pytest.raises(ValueError), feedback.admit_feedback_cell(directory, ordinal=ordinal):
        pytest.fail("Replayed admission")
    with pytest.raises(FileExistsError):
        feedback.prepare_feedback_cell(ordinal, directory)


@pytest.mark.parametrize(
    "mutation", ["implementation", "test", "receipt", "extra", "mode", "wrong_cell"]
)
def test_pre_admission_drift_refuses(tmp_path, mutation):
    directory = tmp_path / "cell"
    feedback.prepare_feedback_cell(2, directory)
    ordinal = 2
    if mutation in {"implementation", "test"}:
        name = "pagination.py" if mutation == "implementation" else "test_visible.py"
        (directory / "workspace" / name).write_text("# changed\n")
    elif mutation == "receipt":
        (directory / "prepared.json").write_text("{}")
    elif mutation == "extra":
        (directory / "extra").touch()
    elif mutation == "mode":
        directory.chmod(0o755)
    else:
        ordinal = 1
    with pytest.raises(ValueError), feedback.admit_feedback_cell(directory, ordinal=ordinal):
        pytest.fail("Drifted admission")
    assert not (directory / "admitted.json").exists()


def setup(tmp_path, ordinal=2):
    directory = tmp_path / "cell"
    receipt = feedback.prepare_feedback_cell(ordinal, directory)
    _, task = feedback._selection(ordinal)
    cell = feedback.FeedbackCellWorkspace(ordinal=ordinal, directory=directory)
    return task, cell, receipt, public_settings()


def script(provider, task, *, correct=True, weaken=False):
    content = (task.reference / "pagination.py").read_text()
    if not correct:
        content = content.replace("type(page) is not int", "not isinstance(page, int)")
        content = content.replace("type(page_size) is not int", "not isinstance(page_size, int)")
    edits = [
        ToolCall(
            id="write", name="write_file", arguments={"path": "pagination.py", "content": content}
        )
    ]
    if weaken:
        edits.append(
            ToolCall(
                id="weaken",
                name="write_file",
                arguments={
                    "path": "test_visible.py",
                    "content": visible_check_bytes("control").decode(),
                },
            )
        )
    edits.append(ToolCall(id="check", name="run_project_check", arguments={}))
    provider.script(
        response(tool_calls=edits),
        response(content="Observed project check."),
        response(content="Observed project check."),
    )


@pytest.mark.parametrize("ordinal", [1, 2, 7, 8])
async def test_shared_evaluator_preserves_original_grading_and_treatment_baseline(
    tmp_path, ordinal
):
    task, cell, receipt, settings = setup(tmp_path, ordinal)
    provider = ScriptedProvider()
    script(provider, task)
    result = await runner.run_task(
        task,
        provider=provider,
        settings=settings,
        mode="harness",
        attempt=receipt.cell.repeat,
        feedback_cell=cell,
    )
    assert result.passed and result.completion_status == "completed"
    assert result.modified == ("pagination.py",) and "test_visible.py" not in result.diff
    assert result.runtime_config["workspace_policy"] == "prepared-public-feedback-single-use-v1"
    assert result.runtime_config["prepared_workspace_root"] == str(cell.directory / "workspace")
    evidence = result.runtime_config["public_feedback_preservation"]
    assert evidence["status"] == "checks_passed"
    assert evidence["checkpoints"][-2:] == ["before_grading", "after_grading"]
    assert "public_pilot_preservation" not in result.runtime_config


@pytest.mark.parametrize("ordinal", [1, 2])
async def test_visible_check_never_redefines_independent_acceptance(tmp_path, ordinal):
    task, cell, receipt, settings = setup(tmp_path, ordinal)
    provider = ScriptedProvider()
    script(provider, task, correct=False)
    result = await runner.run_task(
        task,
        provider=provider,
        settings=settings,
        mode="harness",
        attempt=receipt.cell.repeat,
        feedback_cell=cell,
    )
    assert not result.passed and result.acceptance_exit_code == 1
    assert result.verification_status == ("passed" if receipt.cell.arm == "control" else "failed")


async def test_weakened_treatment_is_diffed_against_selected_initial_test(tmp_path):
    task, cell, _, settings = setup(tmp_path)
    provider = ScriptedProvider()
    script(provider, task, correct=False, weaken=True)
    result = await runner.run_task(
        task, provider=provider, settings=settings, mode="harness", feedback_cell=cell
    )
    assert not result.passed and result.verification_status == "passed"
    assert set(result.modified) == {"pagination.py", "test_visible.py"}
    assert "-def test_bool_arguments_are_invalid" in result.diff


@pytest.mark.parametrize("defect", ["request", "repeat", "mode", "settings", "public", "heldout"])
async def test_invalid_evaluator_selection_refuses_before_admission(tmp_path, defect):
    task, cell, _, settings = setup(tmp_path)
    provider = ScriptedProvider()
    kwargs = {"mode": "harness", "attempt": 1, "feedback_cell": cell}
    if defect == "request":
        task = replace(task, request=task.request + " changed")
    elif defect == "repeat":
        kwargs["attempt"] = 2
    elif defect == "mode":
        kwargs["mode"] = "minimal"
    elif defect == "settings":
        settings.max_tool_iterations = 7
    elif defect == "public":
        from evals.public_admission import PublicCellWorkspace

        kwargs["public_cell"] = PublicCellWorkspace(ordinal=2, directory=cell.directory)
    else:
        kwargs["prepared_directory"] = cell.directory
    with pytest.raises(ValueError):
        await runner.run_task(task, provider=provider, settings=settings, **kwargs)
    assert not provider.calls and not (cell.directory / "admitted.json").exists()


async def test_midrun_settings_drift_stops_before_next_request(tmp_path):
    task, cell, _, settings = setup(tmp_path)

    class DriftProvider(ScriptedProvider):
        async def chat(self, *args, **kwargs):
            response = await super().chat(*args, **kwargs)
            settings.max_tool_iterations = 7
            return response

    provider = DriftProvider()
    provider.script(response(tool_calls=[ToolCall(id="tree", name="tree", arguments={})]))
    row = await runner.run_task(
        task, provider=provider, settings=settings, mode="harness", feedback_cell=cell
    )
    assert len(provider.calls) == 1 and not row.passed
    assert row.termination == "error:FeedbackDriftError"
    assert row.runtime_config["public_feedback_preservation"]["status"] == "failed"


def test_guard_failure_is_latched_after_source_restoration(tmp_path, monkeypatch):
    task, cell, _, settings = setup(tmp_path)
    guard = feedback.FeedbackTaskGuard(task, cell, mode="harness", attempt=1, settings=settings)
    selection = feedback._selection
    monkeypatch.setattr(
        feedback, "_selection", lambda ordinal: (_ for _ in ()).throw(ValueError("drift"))
    )
    with pytest.raises(feedback.FeedbackDriftError):
        guard()
    monkeypatch.setattr(feedback, "_selection", selection)
    with pytest.raises(feedback.FeedbackDriftError, match="previously failed"):
        guard()


async def test_after_grading_drift_retains_grader_result_but_rejects_acceptance(
    tmp_path, monkeypatch
):
    task, cell, _, settings = setup(tmp_path)
    provider = ScriptedProvider()
    script(provider, task)
    grade = runner.acceptance_check

    async def drift(*args, **kwargs):
        result = await grade(*args, **kwargs)
        settings.max_tool_iterations = 7
        return result

    monkeypatch.setattr(runner, "acceptance_check", drift)
    row = await runner.run_task(
        task, provider=provider, settings=settings, mode="harness", feedback_cell=cell
    )
    assert row.acceptance_exit_code == 0 and not row.passed
    assert row.termination == "error:FeedbackDriftError"
    assert (
        row.runtime_config["public_feedback_preservation"]["failed_checkpoint"] == "after_grading"
    )


@pytest.mark.parametrize("ordinal", [True, 0, 9, "1"])
def test_invalid_ordinal_refuses_without_allocation(tmp_path, ordinal):
    directory = tmp_path / "cell"
    with pytest.raises(ValueError):
        feedback.prepare_feedback_cell(ordinal, directory)
    assert not directory.exists()


def test_git_workspace_preparation_refuses(tmp_path):
    (tmp_path / ".git").mkdir()
    with pytest.raises(ValueError, match="outside Git"):
        feedback.prepare_feedback_cell(1, tmp_path / "cell")
    assert not (tmp_path / "cell").exists()


def test_post_marker_failure_consumes_admission_and_retains_partial_state(tmp_path, monkeypatch):
    directory = tmp_path / "cell"
    feedback.prepare_feedback_cell(2, directory)
    selection = feedback._selection
    calls = 0

    def fail_after_marker(ordinal):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("post marker drift")
        return selection(ordinal)

    monkeypatch.setattr(feedback, "_selection", fail_after_marker)
    with (
        pytest.raises(ValueError, match="post marker"),
        feedback.admit_feedback_cell(directory, ordinal=2),
    ):
        pytest.fail("Failed admission yielded")
    assert (directory / "admitted.json").exists()
    assert (directory / "workspace/pagination.py").exists()
