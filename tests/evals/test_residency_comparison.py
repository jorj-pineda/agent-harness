"""Fixed offline schedule validation, not inference or residency evidence."""

import json

import pytest

from evals.residency_comparison import RESIDENCY_PLAN_PATH, validate_residency_plan


def test_fixed_schedule_preserves_all_pairs_and_budget():
    plan = validate_residency_plan()
    assert len(plan.cells) == 16 and not plan.inference_authorized
    assert [cell.keep_alive_seconds for cell in plan.cells] == [0, 300] * 4 + [300, 0] * 4
    for task in {cell.task_id for cell in plan.cells}:
        assert {
            (cell.repeat, cell.keep_alive_seconds) for cell in plan.cells if cell.task_id == task
        } == {(1, 0), (1, 300), (2, 0), (2, 300)}


@pytest.mark.parametrize(
    "fault", ["hash", "authorization", "label", "order", "task", "repeat", "mode", "arm", "count"]
)
def test_schedule_drift_refuses_without_execution(tmp_path, fault):
    doc = json.loads(RESIDENCY_PLAN_PATH.read_text())
    if fault == "hash":
        doc["base_plan_sha256"] = "0" * 64
    elif fault == "authorization":
        doc["inference_authorized"] = True
    elif fault == "label":
        doc["evidence_label"] = "held-out"
    elif fault == "order":
        doc["cells"].reverse()
    elif fault == "count":
        doc["cells"].pop()
    else:
        key, value = {
            "task": ("task_id", "other"),
            "repeat": ("repeat", 2),
            "mode": ("mode", "minimal"),
            "arm": ("keep_alive_seconds", 1),
        }[fault]
        doc["cells"][0][key] = value
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        validate_residency_plan(path)


def test_oversized_schedule_refuses(tmp_path):
    path = tmp_path / "plan.json"
    path.write_bytes(b" " * 64_001)
    with pytest.raises(ValueError):
        validate_residency_plan(path)
