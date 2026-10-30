"""Offline fixed-protocol contracts, not product-model attempts."""

import json
from copy import deepcopy

import pytest

from evals import heldout_comparison as comparison
from evals import real_run
from harness import providers


def test_reviewed_plan_validates_without_provider_and_retains_sixteen_cells(monkeypatch):
    monkeypatch.setattr(providers, "build_configured_provider", lambda *a: pytest.fail("provider"))
    plan = comparison.validate_plan()
    tasks = real_run.load_tasks(comparison.REGISTRATIONS[0].root / "tasks.yaml")
    ledger, _ = real_run._plan_cohort(tasks, real_run.MODES, plan.repeats)
    assert len(ledger.cells) == 16
    assert {c.mode for c in ledger.cells} == {"harness", "minimal"}
    assert all(c.status == "planned" for c in ledger.cells)
    assert plan.inference_authorized is False
    assert sum(plan.runtime.max_turn_wall_seconds for c in ledger.cells) == 2400
    assert sum(plan.runtime.max_total_tokens_per_turn for c in ledger.cells) == 512000


@pytest.mark.parametrize(
    "defect",
    [
        "model",
        "digest",
        "provider",
        "authorization",
        "seal",
        "order",
        "request",
        "category",
        "modes",
        "repeats",
        "budget",
        "check",
        "extra",
    ],
)
def test_changed_protocol_refuses(tmp_path, defect):
    document = json.loads(comparison.PLAN_PATH.read_text())
    changed = deepcopy(document)
    if defect == "model":
        changed["model"] = "another-model"
    elif defect == "digest":
        changed["model_digest"] = "0" * 64
    elif defect == "provider":
        changed["provider"] = "openai"
    elif defect == "authorization":
        changed["inference_authorized"] = True
    elif defect == "seal":
        changed["seal_sha256"] = "0" * 64
    elif defect == "order":
        changed["tasks"].reverse()
    elif defect == "request":
        changed["tasks"][0]["request_sha256"] = "0" * 64
    elif defect == "category":
        changed["tasks"][0]["category"] = "feature"
    elif defect == "modes":
        changed["modes"].reverse()
    elif defect == "repeats":
        changed["repeats"] = 3
    elif defect == "budget":
        changed["runtime"]["max_total_tokens_per_turn"] = 64000
    elif defect == "check":
        changed["runtime"]["project_check_argv"] = ["true"]
    else:
        changed["runtime"]["unexpected"] = True
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        comparison.validate_plan(path)


def test_oversized_protocol_refuses_without_catalog_reads(monkeypatch, tmp_path):
    path = tmp_path / "plan.json"
    path.write_bytes(b"x" * 64001)
    monkeypatch.setattr(comparison, "verify_catalog", lambda *a: pytest.fail("catalog read"))
    with pytest.raises(ValueError, match="byte limit"):
        comparison.validate_plan(path)
