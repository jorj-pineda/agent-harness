import json
import shutil
import sys

import pytest

import evals.public_envelope as original
import evals.public_feedback as feedback
import evals.real_run as real_run
from tools.process import run_process


def test_selected_plan_preserves_original_task_and_balances_eight_cells():
    plan = feedback.validate_feedback_plan()
    assert plan.inference_authorized is False
    assert plan.task == next(
        t for t in original.validate_public_plan().tasks if t.id == "paginate_items"
    )
    assert [c.arm for c in plan.cells] == [
        "control",
        "bool_check",
        "bool_check",
        "control",
        "control",
        "bool_check",
        "bool_check",
        "control",
    ]
    assert [c.ordinal for c in plan.cells] == list(range(1, 9))
    assert [c.repeat for c in plan.cells] == [1, 1, 2, 2, 3, 3, 4, 4]


@pytest.mark.parametrize("field", ["cells", "inference_authorized", "task", "treatment"])
def test_plan_drift_refuses(tmp_path, field):
    data = json.loads(feedback.PLAN_PATH.read_bytes())
    data[field] = None
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="reviewed bytes"):
        feedback.validate_feedback_plan(path)


@pytest.mark.parametrize("change", ["content", "mode", "symlink", "oversized"])
def test_treatment_drift_refuses(tmp_path, monkeypatch, change):
    path = tmp_path / "test_visible.py"
    shutil.copy2(feedback.TREATMENT_PATH, path)
    if change == "content":
        path.write_bytes(path.read_bytes() + b"\n# changed\n")
    elif change == "mode":
        path.chmod(0o600)
    elif change == "symlink":
        path.unlink()
        path.symlink_to(feedback.TREATMENT_PATH)
    else:
        path.write_bytes(b" " * 16_001)
    monkeypatch.setattr(feedback, "TREATMENT_PATH", path)
    with pytest.raises((ValueError, OSError)):
        feedback.visible_check_bytes("bool_check")


def test_fixture_selection_drift_refuses(tmp_path, monkeypatch):
    path = tmp_path / "fixture.json"
    path.write_bytes(feedback.FIXTURE_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(feedback, "FIXTURE_PATH", path)
    with pytest.raises(ValueError, match="fixture selection"):
        feedback.validate_feedback_plan()


@pytest.mark.parametrize("tree", ["fixtures", "acceptance", "reference"])
def test_original_task_drift_refuses(tmp_path, monkeypatch, tree):
    root = tmp_path / "tasks"
    shutil.copytree(original.TASKS_ROOT, root)
    monkeypatch.setattr(original, "TASKS_ROOT", root)
    monkeypatch.setattr(real_run, "TASKS_ROOT", root)
    path = next((root / tree / "paginate_items").glob("*.py"))
    path.write_bytes(path.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="task pins"):
        feedback.validate_feedback_plan()


def test_control_is_original_and_invalid_arm_refuses():
    path = real_run.TASKS_ROOT / "fixtures/paginate_items/test_visible.py"
    assert feedback.visible_check_bytes("control") == path.read_bytes()
    with pytest.raises(ValueError, match="Select"):
        feedback.visible_check_bytes("other")


@pytest.mark.parametrize("correct", [False, True])
@pytest.mark.parametrize("arm", ["control", "bool_check"])
async def test_selected_visible_test_discriminates_bool_defect(tmp_path, correct, arm):
    task = next(t for t in real_run.load_tasks() if t.id == "paginate_items")
    content = (task.reference / "pagination.py").read_text()
    if not correct:
        # Representative observed defect: isinstance accepts True as an integer.
        content = content.replace("type(page) is not int", "not isinstance(page, int)")
        content = content.replace("type(page_size) is not int", "not isinstance(page_size, int)")
    (tmp_path / "pagination.py").write_text(content)
    (tmp_path / "test_visible.py").write_bytes(feedback.visible_check_bytes(arm))
    result = await run_process(
        [sys.executable, "-B", "-m", "pytest", "-q", "test_visible.py"],
        cwd=tmp_path,
        timeout_seconds=30,
        output_limit=12000,
        env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )
    assert result.exit_code == (1 if arm == "bool_check" and not correct else 0)
    assert (
        "4 failed, 6 passed"
        if result.exit_code
        else ("10 passed" if arm == "bool_check" else "2 passed")
    ) in result.stdout
