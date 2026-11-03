import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import heldout_execution_preflight as preflight
from evals import residency_execution as executor
from evals.heldout_execution_manifest import ExecutionManifest, load_execution
from evals.heldout_execution_provider import build_cell_provider
from evals.residency_runner import run_schedule
from tests.evals.test_execution_manifest import selected as selected
from tests.evals.test_execution_preflight import prepared as prepared
from tests.evals.test_residency_execution import setup as setup


def manifest(selected):
    raw = json.loads(selected[0].read_bytes())
    raw.update(residency_plan_sha256="0" * 64, residency_feasibility_ordinal=2)
    return raw


def test_feasibility_selection_round_trips_and_cannot_reuse_old_approval(selected):
    model = ExecutionManifest.model_validate(manifest(selected))
    assert ExecutionManifest.model_validate_json(model.model_dump_json()) == model
    assert model.residency_feasibility_ordinal == 2
    selected[0].write_text(model.model_dump_json())
    with pytest.raises(ValueError, match="bytes/mode differ"):
        load_execution(
            selected[0],
            manifest_sha256=selected[1],
            approval_path=selected[2],
            approval_sha256=selected[3],
        )


@pytest.mark.parametrize("ordinal", [0, 1, 3, 16, True, 2.0, "2"])
def test_only_the_strict_selected_ordinal_is_accepted(selected, ordinal):
    raw = manifest(selected)
    raw["residency_feasibility_ordinal"] = ordinal
    with pytest.raises(ValueError):
        ExecutionManifest.model_validate(raw)


@pytest.mark.parametrize("update", [{"residency_plan_sha256": None}, {"keep_alive_seconds": 300}])
def test_feasibility_requires_schedule_without_global_override(selected, update):
    with pytest.raises(ValueError, match="Feasibility requires"):
        ExecutionManifest.model_validate({**manifest(selected), **update})


def test_constructed_invalid_selection_refuses_before_preflight_paths(selected, monkeypatch):
    model = ExecutionManifest.model_validate(manifest(selected)).model_copy(
        update={"residency_feasibility_ordinal": 1}
    )
    monkeypatch.setattr(preflight, "execution_paths", lambda *a: pytest.fail("paths inspected"))
    with pytest.raises(ValueError):
        preflight.preflight(model)


@pytest.mark.parametrize("ordinal", [1, 3, 16])
def test_provider_refuses_unselected_ordinal_before_pins_or_ownership(selected, ordinal):
    admission = SimpleNamespace(manifest=ExecutionManifest.model_validate(manifest(selected)))
    with pytest.raises(ValueError, match="outside the selected single-cell"):
        build_cell_provider(
            admission,
            ordinal,
            settings=None,
            transport=None,
            check_owned=lambda: pytest.fail("ownership checked"),
        )


@pytest.mark.parametrize("fail", [False, True])
def test_feasibility_ledger_plans_only_one_attempt_and_never_reenters(tmp_path, fail):
    calls, snapshots = [], []

    def execute(cell):
        calls.append(cell)
        if fail:
            raise ValueError("unknown usage")

    marker = tmp_path / "started.json"
    if fail:
        with pytest.raises(ValueError, match="unknown usage"):
            run_schedule(
                marker, execute_and_settle=execute, persist=snapshots.append, feasibility=True
            )
    else:
        run_schedule(marker, execute_and_settle=execute, persist=snapshots.append, feasibility=True)
    assert [(c.ordinal, c.task_id, c.keep_alive_seconds) for c in calls] == [
        (2, "ho_iso_week", 300)
    ]
    final = snapshots[-1]
    assert final.format == "residency-feasibility-ledger-v1" and len(final.cells) == 1
    assert final.status == ("stopped" if fail else "finished")
    assert final.cells[0].status == ("started" if fail else "recorded")
    with pytest.raises(FileExistsError):
        run_schedule(marker, execute_and_settle=execute, persist=snapshots.append, feasibility=True)
    assert len(calls) == 1


def select_feasibility(admission):
    admission.manifest = admission.manifest.model_copy(update={"residency_feasibility_ordinal": 2})
    raw = admission.manifest.model_dump_json().encode()
    (admission.directory / "manifest.json").write_bytes(raw)
    admission.approval.manifest_sha256 = hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("prepared", [True], indirect=True)
def test_single_cell_uses_shared_execution_and_leaves_other_fixtures_unadmitted(setup, monkeypatch):
    admission, events, _state, transport = setup
    controls = []
    validate = executor.validate_task

    async def checked(task, **kwargs):
        assert events == []
        await validate(task, **kwargs)
        controls.append(task.id)

    monkeypatch.setattr(executor, "validate_task", checked)
    with pytest.raises(ValueError, match="matching explicit"):
        executor.run_residency_feasibility(admission, transport_factory=transport, scripted=True)
    assert not (admission.directory / "cohort-started.json").exists()
    select_feasibility(admission)
    with pytest.raises(ValueError, match="matching explicit"):
        executor.run_residency_cohort(admission, transport_factory=transport, scripted=True)
    assert events == [] and not (admission.directory / "cohort-started.json").exists()
    ledger = executor.run_residency_feasibility(
        admission, transport_factory=transport, scripted=True
    )
    assert ledger.status == "finished" and len(ledger.cells) == 1
    assert controls == ["ho_iso_week", "ho_rolling_mean", "ho_endpoint_record", "ho_cache_policy"]
    assert events == [("open", 300), ("settle", 300)]
    report = json.loads((admission.directory / "report.json").read_bytes())
    assert len(report["results"]) == 1 and report["results"][0]["passed"]
    assert report["cohort"]["cells"][0]["cell"]["ordinal"] == 2
    staged = Path(admission.manifest.staged_directory)
    assert [n for n in range(1, 17) if (staged / f"cell-{n:02d}/admitted.json").exists()] == [2]
    assert sorted(p.name for p in admission.directory.glob("cell-*")) == ["cell-02"]
    with pytest.raises((ValueError, FileExistsError)):
        executor.run_residency_feasibility(admission, transport_factory=transport, scripted=True)
    assert events == [("open", 300), ("settle", 300)]


@pytest.mark.parametrize("prepared", [True], indirect=True)
def test_single_cell_preserves_result_and_partial_ledger_on_settlement_failure(setup):
    admission, events, state, transport = setup
    select_feasibility(admission)
    state["cleanup_failure"] = True
    with pytest.raises(ValueError, match="settlement failed"):
        executor.run_residency_feasibility(admission, transport_factory=transport, scripted=True)
    report = json.loads((admission.directory / "report.json").read_bytes())
    assert len(report["results"]) == 1 and report["cohort"]["status"] == "stopped"
    assert len(report["cohort"]["cells"]) == 1
    assert report["cohort"]["cells"][0]["status"] == "started"
    assert events == [("open", 300), ("settle", 300)]
