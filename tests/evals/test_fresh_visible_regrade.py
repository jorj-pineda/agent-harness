"""Synthetic records and copied real grading; never inference or approval."""

import hashlib
import json
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import evals.fresh_visible_regrade as regrade
from evals.fresh_visible_runner import CellState, FreshLedger
from evals.fresh_visible_workspaces import ROOT, SELECTION_SHA256, selected_cell
from evals.heldout_command_environment import environment_bytes
from evals.heldout_inputs import _save
from evals.real_run import _task_command_environment, load_tasks, run_task
from harness.config import Settings
from tests.api.conftest import ScriptedProvider, make_response
from tests.evals.test_fresh_visible_execution import values
from tools.process import ProcessResult
from workspace.changes import SnapshotLimitError


def make_directory(path):
    path.mkdir(mode=0o700)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(source):
    files, modes = regrade._artifact(source)
    return hashlib.sha256(
        json.dumps({"files": files, "modes": modes}, sort_keys=True).encode()
    ).hexdigest()


@pytest.fixture
async def selected(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    task = next(task for task in load_tasks(ROOT / "tasks.yaml") if task.id == "fv_lookup_option")
    provider = ScriptedProvider()
    provider.script(
        make_response(content="Scripted no patch", prompt_tokens=10, completion_tokens=5)
    )
    row = await run_task(task, provider=provider, settings=Settings(_env_file=None), mode="harness")
    data = values(root)
    data["command_environment"] = _task_command_environment(
        {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin"}
    )
    staged = Path(data["staged_directory"])
    source = staged / "cell-01/workspace"
    shutil.copytree(task.fixture, source)
    (source / "lookup.py").write_bytes((task.reference / "lookup.py").read_bytes())
    archive = Path(data["archive_directory"]) / "captured"
    archive.mkdir(parents=True)
    data["command_environment_sha256"] = _save(
        archive / "command-environment.json", environment_bytes(data["command_environment"])
    )
    data["runtime_inventory_sha256"] = _save(
        archive / "runtime-inventory.json",
        json.dumps({"executables": {"python": {"selected_path": sys.executable}}}).encode(),
    )
    monkeypatch.setattr(regrade, "check_runtime_selection", lambda *a: None)
    monkeypatch.setattr(regrade, "check_dependency_selection", lambda *a: None)
    row = replace(
        row,
        passed=True,
        acceptance_exit_code=0,
        completion_status="completed",
        runtime_config={
            **row.runtime_config,
            "fresh_visible_preservation": {
                "format": "fresh-visible-preservation-v1",
                "ordinal": 1,
                "selection_sha256": SELECTION_SHA256,
                "status": "checks_passed",
            },
        },
    )
    ledger = FreshLedger(
        status="stopped",
        cells=[
            CellState(cell=cell, status="started" if cell.ordinal == 1 else "unrun")
            for cell in [selected_cell(i)[0] for i in range(1, 13)]
        ],
    )
    output = Path(data["output_directory"])
    make_directory(output)
    report = output / "report.json"
    _save(
        report,
        json.dumps(
            {
                "mode": "scripted-real-tools-rehearsal",
                "cohort": ledger.model_dump(),
                "results": [asdict(row)],
            }
        ).encode(),
    )
    manifest = root / "manifest.json"
    _save(manifest, json.dumps(data).encode())
    kwargs = {
        "manifest_sha256": sha(manifest),
        "report_sha256": sha(report),
        "artifact_pins": {1: pin(source)},
        "directory": root / "regrades",
    }
    return manifest, kwargs, source, report


async def test_copies_grade_without_promoting_started_cells_or_missing_slots(selected):
    manifest, kwargs, source, report = selected
    before = regrade._artifact(source), report.read_bytes()
    result = await regrade.regrade_fresh_run(manifest, **kwargs)
    cell = result.cells[0]
    assert cell.regrade_exit_code == 0 and cell.matches_recorded_exit_code
    assert cell.ledger_status == "started" and cell.scoped_source_unchanged
    assert all(c.status == "no_result" and c.regrade_exit_code is None for c in result.cells[1:])
    assert result.benefit_assessment == "not_assessed" and not result.inference_ran
    assert result.source_report_mode == "scripted-real-tools-rehearsal"
    assert (regrade._artifact(source), report.read_bytes()) == before
    assert (kwargs["directory"] / "regrades.json").exists()
    with pytest.raises(FileExistsError):
        await regrade.regrade_fresh_run(manifest, **kwargs)


async def test_weakened_visible_tests_cannot_redefine_authoritative_acceptance(selected):
    manifest, kwargs, source, report = selected
    text = (
        (source / "lookup.py")
        .read_text()
        .replace("options.get(key, default)", "options.get(key) or default")
    )
    (source / "lookup.py").write_text(text)
    (source / "test_visible.py").write_text("def test_weakened(): assert True\n")
    kwargs["artifact_pins"] = {1: pin(source)}
    result = await regrade.regrade_fresh_run(manifest, **kwargs)
    assert result.cells[0].recorded_passed is True
    assert result.cells[0].regrade_exit_code == 1 and not result.cells[0].matches_recorded_exit_code
    assert json.loads(report.read_bytes())["results"][0]["passed"] is True


@pytest.mark.parametrize(
    "fault",
    [
        "manifest_hash",
        "report_hash",
        "ordinals",
        "criterion",
        "overlap",
        "environment",
        "interpreter",
    ],
)
async def test_invalid_selection_refuses_before_output_or_grader(selected, monkeypatch, fault):
    manifest, kwargs, source, _report = selected

    async def forbidden(*a, **k):
        pytest.fail("grader before valid selection")

    monkeypatch.setattr(regrade, "acceptance_check", forbidden)
    if fault == "manifest_hash":
        kwargs["manifest_sha256"] = "0" * 64
    elif fault == "report_hash":
        kwargs["report_sha256"] = "0" * 64
    elif fault == "ordinals":
        kwargs["artifact_pins"] = {True: kwargs["artifact_pins"][1]}
    elif fault == "criterion":
        monkeypatch.setattr(regrade, "_acceptance_digest", lambda *a: "f" * 64)
    elif fault == "overlap":
        kwargs["directory"] = source.parent / "new"
    elif fault == "interpreter":
        monkeypatch.setattr(regrade.sys, "executable", "/different/venv/python")
    else:
        data = json.loads(manifest.read_bytes())
        data["command_environment"]["LC_ALL"] = "POSIX"
        manifest.write_text(json.dumps(data))
        kwargs["manifest_sha256"] = sha(manifest)
    with pytest.raises(ValueError):
        await regrade.regrade_fresh_run(manifest, **kwargs)
    assert not kwargs["directory"].exists()


@pytest.mark.parametrize("fault", ["pin", "symlink", "bytes", "file_count"])
async def test_copy_refusal_retains_failure_without_running_grader(selected, monkeypatch, fault):
    manifest, kwargs, source, _report = selected

    async def forbidden(*a, **k):
        pytest.fail("grader before bounded matching copy")

    monkeypatch.setattr(regrade, "acceptance_check", forbidden)
    if fault == "pin":
        kwargs["artifact_pins"] = {1: "0" * 64}
    elif fault == "symlink":
        (source / "link.py").symlink_to(source / "lookup.py")
    elif fault == "bytes":
        (source / "large.py").write_bytes(b"x" * 1_000_001)
    else:
        for i in range(101):
            (source / f"extra-{i}.py").write_text("")
    with pytest.raises((ValueError, OSError, SnapshotLimitError)):
        await regrade.regrade_fresh_run(manifest, **kwargs)
    assert (kwargs["directory"] / "failure.json").exists()
    assert not (kwargs["directory"] / "regrades.json").exists()


async def test_external_source_change_is_retained_and_prevents_success(selected, monkeypatch):
    manifest, kwargs, source, _report = selected

    async def grade(*a, **k):
        (source / "later.py").write_text("external write retained\n")
        return ProcessResult([], 0, "passed", "", False, False)

    monkeypatch.setattr(regrade, "acceptance_check", grade)
    with pytest.raises(ValueError, match="Original artifact changed"):
        await regrade.regrade_fresh_run(manifest, **kwargs)
    assert (source / "later.py").read_text() == "external write retained\n"
    assert not (kwargs["directory"] / "regrades.json").exists()


async def test_cancellation_retains_failure_and_preserves_original(selected, monkeypatch):
    import asyncio

    manifest, kwargs, source, report = selected
    before = regrade._artifact(source), report.read_bytes()

    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(regrade, "acceptance_check", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await regrade.regrade_fresh_run(manifest, **kwargs)
    assert (regrade._artifact(source), report.read_bytes()) == before
    assert (
        json.loads((kwargs["directory"] / "failure.json").read_bytes())["error_type"]
        == "CancelledError"
    )
    assert not (kwargs["directory"] / "regrades.json").exists()


async def test_later_artifact_failure_keeps_earlier_regrade_and_original_report(selected):
    manifest, kwargs, source, report = selected
    payload = json.loads(report.read_bytes())
    second = json.loads(json.dumps(payload["results"][0]))
    second["runtime_config"]["fresh_visible_preservation"]["ordinal"] = 2
    payload["results"].append(second)
    payload["cohort"]["cells"][0]["status"] = "recorded"
    payload["cohort"]["cells"][1]["status"] = "started"
    report.write_text(json.dumps(payload))
    kwargs["report_sha256"] = sha(report)
    second_source = source.parent.parent / "cell-02/workspace"
    shutil.copytree(source, second_source)
    kwargs["artifact_pins"][2] = "0" * 64
    original = report.read_bytes()
    with pytest.raises(ValueError, match="independently selected snapshot"):
        await regrade.regrade_fresh_run(manifest, **kwargs)
    assert json.loads((kwargs["directory"] / "cell-01.json").read_bytes())["regrade_exit_code"] == 0
    assert (kwargs["directory"] / "failure.json").exists()
    assert not (kwargs["directory"] / "regrades.json").exists()
    assert report.read_bytes() == original


async def test_all_twelve_cells_bind_each_tasks_own_original_criterion(selected):
    manifest, kwargs, source, report = selected
    payload = json.loads(report.read_bytes())
    original = payload["results"][0]
    by_id = {task.id: task for task in load_tasks(ROOT / "tasks.yaml")}
    rows = []
    snapshots = {}
    for ordinal in range(1, 13):
        cell = selected_cell(ordinal)[0]
        task = by_id[cell.task_id]
        artifact = source.parent.parent / f"cell-{ordinal:02d}/workspace"
        if ordinal != 1:
            shutil.copytree(task.fixture, artifact)
        for path in task.reference.iterdir():
            if path.is_file():
                (artifact / path.name).write_bytes(path.read_bytes())
        item = json.loads(json.dumps(original))
        item.update(task_id=task.id, category=task.category, attempt=cell.repeat)
        criterion = regrade._acceptance_digest(task.acceptance)
        item["acceptance_identity"].update(
            status="matching", before_sha256=criterion, after_sha256=criterion
        )
        item["runtime_config"]["fresh_visible_preservation"]["ordinal"] = ordinal
        rows.append(item)
        snapshots[ordinal] = pin(artifact)
        payload["cohort"]["cells"][ordinal - 1]["status"] = "recorded"
    payload["cohort"]["status"] = "finished"
    payload["results"] = rows
    report.write_text(json.dumps(payload))
    kwargs.update(report_sha256=sha(report), artifact_pins=snapshots)
    before = report.read_bytes()
    result = await regrade.regrade_fresh_run(manifest, **kwargs)
    assert len(result.cells) == 12 and all(cell.regrade_exit_code == 0 for cell in result.cells)
    assert len(result.criteria_sha256) == 3
    assert len(set(result.criteria_sha256.values())) == 3
    assert report.read_bytes() == before
    for ordinal, expected in snapshots.items():
        assert pin(source.parent.parent / f"cell-{ordinal:02d}/workspace") == expected
