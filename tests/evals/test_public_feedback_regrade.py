"""Synthetic records and copied real grading; never inference or approval."""

import hashlib
import json
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import evals.public_feedback_regrade as regrade
from evals.heldout_command_environment import environment_bytes
from evals.heldout_inputs import _save
from evals.public_feedback import PLAN_SHA256, validate_feedback_plan
from evals.public_feedback_runner import CellState, FeedbackLedger
from evals.real_run import _task_command_environment, load_tasks, run_task
from harness.config import Settings
from tests.api.conftest import ScriptedProvider, make_response
from tests.evals.test_public_feedback_execution import values
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
    task = next(task for task in load_tasks() if task.id == "paginate_items")
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
    (source / "pagination.py").write_bytes((task.reference / "pagination.py").read_bytes())
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
            "public_feedback_preservation": {
                "format": "public-feedback-preservation-v1",
                "ordinal": 1,
                "plan_sha256": PLAN_SHA256,
                "status": "checks_passed",
            },
        },
    )
    ledger = FeedbackLedger(
        status="stopped",
        cells=[
            CellState(cell=cell, status="started" if cell.ordinal == 1 else "unrun")
            for cell in validate_feedback_plan().cells
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
    result = await regrade.regrade_feedback_run(manifest, **kwargs)
    cell = result.cells[0]
    assert cell.regrade_exit_code == 0 and cell.matches_recorded_exit_code
    assert cell.ledger_status == "started" and cell.scoped_source_unchanged
    assert all(c.status == "no_result" and c.regrade_exit_code is None for c in result.cells[1:])
    assert result.benefit_assessment == "not_assessed" and not result.inference_ran
    assert result.source_report_mode == "scripted-real-tools-rehearsal"
    assert (regrade._artifact(source), report.read_bytes()) == before
    assert (kwargs["directory"] / "regrades.json").exists()
    with pytest.raises(FileExistsError):
        await regrade.regrade_feedback_run(manifest, **kwargs)


async def test_weakened_visible_tests_cannot_redefine_authoritative_acceptance(selected):
    manifest, kwargs, source, report = selected
    text = (
        (source / "pagination.py")
        .read_text()
        .replace("type(page) is not int", "not isinstance(page, int)")
    )
    (source / "pagination.py").write_text(text)
    (source / "test_visible.py").write_text("def test_weakened(): assert True\n")
    kwargs["artifact_pins"] = {1: pin(source)}
    result = await regrade.regrade_feedback_run(manifest, **kwargs)
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
        await regrade.regrade_feedback_run(manifest, **kwargs)
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
        (source / "link.py").symlink_to(source / "pagination.py")
    elif fault == "bytes":
        (source / "large.py").write_bytes(b"x" * 1_000_001)
    else:
        for i in range(101):
            (source / f"extra-{i}.py").write_text("")
    with pytest.raises((ValueError, OSError, SnapshotLimitError)):
        await regrade.regrade_feedback_run(manifest, **kwargs)
    assert (kwargs["directory"] / "failure.json").exists()
    assert not (kwargs["directory"] / "regrades.json").exists()


async def test_external_source_change_is_retained_and_prevents_success(selected, monkeypatch):
    manifest, kwargs, source, _report = selected

    async def grade(*a, **k):
        (source / "later.py").write_text("external write retained\n")
        return ProcessResult([], 0, "passed", "", False, False)

    monkeypatch.setattr(regrade, "acceptance_check", grade)
    with pytest.raises(ValueError, match="Original artifact changed"):
        await regrade.regrade_feedback_run(manifest, **kwargs)
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
        await regrade.regrade_feedback_run(manifest, **kwargs)
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
    second["runtime_config"]["public_feedback_preservation"]["ordinal"] = 2
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
        await regrade.regrade_feedback_run(manifest, **kwargs)
    assert json.loads((kwargs["directory"] / "cell-01.json").read_bytes())["regrade_exit_code"] == 0
    assert (kwargs["directory"] / "failure.json").exists()
    assert not (kwargs["directory"] / "regrades.json").exists()
    assert report.read_bytes() == original
