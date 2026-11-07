import json
import shutil

import pytest

import evals.public_envelope as pilot
import evals.real_run as real_run


def test_reviewed_plan_has_six_fixed_public_development_cells():
    plan = pilot.validate_public_plan()
    assert plan.inference_authorized is False
    assert [task.category for task in plan.tasks] == ["bugfix", "feature", "refactor"]
    assert [cell.task_id for cell in plan.cells] == list(pilot.TASK_IDS) * 2
    assert [cell.repeat for cell in plan.cells] == [1, 1, 1, 2, 2, 2]
    assert all(cell.mode == "harness" for cell in plan.cells)


@pytest.mark.parametrize("mutation", ["authorization", "order", "extra", "model", "toolset"])
def test_changed_plan_bytes_refuse(tmp_path, mutation):
    raw = json.loads(pilot.PLAN_PATH.read_bytes())
    if mutation == "authorization":
        raw["inference_authorized"] = True
    elif mutation == "order":
        raw["cells"].reverse()
    elif mutation == "extra":
        raw["cells"].append(raw["cells"][0])
    elif mutation == "model":
        raw["model"] = "other"
    else:
        raw["coding_toolset"] = "full"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="reviewed bytes"):
        pilot.validate_public_plan(path)


@pytest.fixture
def copied_catalog(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    shutil.copytree(pilot.TASKS_ROOT, root)
    monkeypatch.setattr(pilot, "TASKS_ROOT", root)
    monkeypatch.setattr(real_run, "TASKS_ROOT", root)
    return root


@pytest.mark.parametrize("tree", ["fixtures", "acceptance", "reference"])
def test_task_tree_drift_refuses(copied_catalog, tree):
    file = next((copied_catalog / tree / "divide_zero").glob("*.py"))
    file.write_text(file.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="task pins"):
        pilot.validate_public_plan()


def test_catalog_drift_refuses(copied_catalog):
    path = copied_catalog / "tasks.yaml"
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="catalog"):
        pilot.validate_public_plan()


def test_oversized_plan_refuses(tmp_path):
    path = tmp_path / "plan.json"
    path.write_bytes(b" " * 64_001)
    with pytest.raises(ValueError):
        pilot.validate_public_plan(path)


def test_symlink_tree_entry_refuses(copied_catalog):
    root = copied_catalog / "fixtures/divide_zero"
    (root / "linked.py").symlink_to(root / "calc.py")
    with pytest.raises(ValueError):
        pilot.validate_public_plan()


def test_selected_file_mode_drift_refuses(copied_catalog):
    path = copied_catalog / "fixtures/divide_zero/calc.py"
    path.chmod(path.stat().st_mode ^ 0o100)
    with pytest.raises(ValueError, match="task pins"):
        pilot.validate_public_plan()


def test_original_budget_drift_refuses(tmp_path, monkeypatch):
    path = tmp_path / "budget.json"
    raw = json.loads(pilot.BUDGET_SOURCE.read_bytes())
    raw["runtime"]["max_tool_iterations"] = 7
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(pilot, "BUDGET_SOURCE", path)
    with pytest.raises(ValueError):
        pilot.validate_public_plan()
