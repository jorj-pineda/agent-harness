"""Bounded copied-artifact acceptance regrades; no inference or benefit decision."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import Field, TypeAdapter

from evals.fresh_visible_execution_manifest import FreshExecutionManifest
from evals.fresh_visible_runner import FreshLedger
from evals.fresh_visible_summary import fresh_summary
from evals.fresh_visible_workspaces import ROOT, FreshCell, selected_cell
from evals.heldout_command_environment import environment_bytes, load_environment_pin
from evals.heldout_execution_manifest import Digest, FrozenModel, _selected_bytes
from evals.heldout_guard import _read_regular
from evals.heldout_inputs import _save, check_dependency_selection, check_runtime_selection
from evals.public_workspaces import _destination
from evals.real_run import (
    RealTaskResult,
    _acceptance_digest,
    _harness_revision,
    acceptance_check,
    load_tasks,
)
from workspace.changes import snapshot
from workspace.disposable import COPY_IGNORE_PATTERNS


class RegradedCell(FrozenModel):
    cell: FreshCell
    ledger_status: Literal["unrun", "started", "recorded"]
    status: Literal["no_result", "graded"]
    recorded_passed: bool | None = None
    recorded_completion: str | None = None
    recorded_acceptance_exit_code: int | None = None
    regrade_exit_code: int | None = None
    acceptance_output: str | None = None
    acceptance_output_truncated: bool | None = None
    matches_recorded_exit_code: bool | None = None
    artifact_sha256: Digest | None = None
    scoped_source_unchanged: bool | None = None


class FreshRegrades(FrozenModel):
    format: Literal["fresh-visible-copied-regrades-v1"] = "fresh-visible-copied-regrades-v1"
    manifest_sha256: Digest
    report_sha256: Digest
    criteria_sha256: dict[str, Digest] = Field(min_length=3, max_length=3)
    source_report_mode: Literal["scripted-real-tools-rehearsal", "real-tools-trusted-fixtures"]
    grader_revision: dict[str, str | bool | None]
    cells: list[RegradedCell] = Field(min_length=12, max_length=12)
    inference_ran: Literal[False] = False
    benefit_assessment: Literal["not_assessed"] = "not_assessed"
    process_isolation: Literal[False] = False


def _artifact(root: Path) -> tuple[dict[str, str], dict[str, int]]:
    if root.is_symlink() or root.resolve(strict=True) != root or _destination(root) != root:
        raise ValueError("Regrade source must be a canonical artifact directory")
    modes: dict[str, int] = {}
    files = snapshot(
        root,
        ignore=COPY_IGNORE_PATTERNS,
        max_files=100,
        max_bytes=1_000_000,
        strict=True,
        file_modes=modes,
    )
    if any(value.startswith("symlink:") for value in files.values()):
        raise ValueError("Regrade artifacts cannot contain symlinks")
    return files, modes


def _output_directory(
    directory: Path, manifest: FreshExecutionManifest, manifest_path: Path
) -> Path:
    destination = _destination(directory)
    roots = [
        Path(value)
        for value in (
            manifest.staged_directory,
            manifest.archive_directory,
            manifest.output_directory,
            manifest.image_directory,
        )
    ]
    if any(_destination(root) != root or root.resolve() != root for root in roots):
        raise ValueError("Regrade sources must be canonical evaluation-owned paths")
    if (
        directory != destination
        or directory.resolve() != directory
        or directory.is_symlink()
        or any(directory.is_relative_to(root) or root.is_relative_to(directory) for root in roots)
        or manifest_path.is_relative_to(directory)
    ):
        raise ValueError("Regrade output must be canonical and separate from selected inputs")
    return destination


@contextmanager
def _temporary_copy() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="fresh_regrade_") as temporary:
        root = Path(temporary).resolve() / "artifact"
        root.mkdir(mode=0o700)
        yield root


async def regrade_fresh_run(
    manifest_path: Path,
    *,
    manifest_sha256: str,
    report_sha256: str,
    artifact_pins: Mapping[int, str],
    directory: Path,
) -> FreshRegrades:
    """Grade trusted retained code on copies, preserving the selected report and sources.

    Copies are bounded to 100 nonignored files/1MB, excluding caches/environment
    files. Imported code still has host permissions: this is not process isolation.
    The trusted caller must independently select the expected artifact snapshots;
    Pins bind bytes, not provenance or writers. No-result slots stay unknown,
    not failed attempts. A graded started cell does
    not certify settlement, original completion, interventions or model quality.
    """
    if not manifest_path.is_absolute() or _destination(manifest_path) != manifest_path:
        raise ValueError("Select an absolute evaluation-owned manifest")
    manifest = FreshExecutionManifest.model_validate_json(
        _selected_bytes(manifest_path, manifest_sha256, 64_000)
    )
    report_path = Path(manifest.output_directory) / "report.json"
    raw = _selected_bytes(report_path, report_sha256, 2_000_000)
    report = json.loads(raw)
    mode = report["mode"]
    if mode not in {"scripted-real-tools-rehearsal", "real-tools-trusted-fixtures"}:
        raise ValueError("Unknown fresh report evidence label")
    ledger = FreshLedger.model_validate(report["cohort"])
    rows = [
        TypeAdapter(RealTaskResult).validate_json(json.dumps(row), strict=True)
        for row in report["results"]
    ]
    fresh_summary(ledger, rows, scripted=mode == "scripted-real-tools-rehearsal")
    pins = dict(artifact_pins)
    if any(type(key) is not int for key in pins) or set(pins) != set(range(1, len(rows) + 1)):
        raise ValueError("Select exactly the recorded result artifact ordinals")
    for pin in pins.values():
        TypeAdapter(Digest).validate_python(pin, strict=True)

    cells = [selected_cell(i)[0] for i in range(1, 13)]
    if [state.cell for state in ledger.cells] != cells:
        raise ValueError("Regrade schedule differs from selected fresh plan")
    by_id = {task.id: task for task in load_tasks(ROOT / "tasks.yaml")}
    criteria = {task.id: _acceptance_digest(task.acceptance) for task in by_id.values()}
    if any(value is None for value in criteria.values()) or any(
        row.acceptance_policy != "owned-pytest-v1"
        or (row.acceptance_identity or {}).get("status") != "matching"
        or (row.acceptance_identity or {}).get("before_sha256") != criteria[row.task_id]
        or (row.acceptance_identity or {}).get("after_sha256") != criteria[row.task_id]
        for row in rows
    ):
        raise ValueError("Regrade requires unchanged original criterion/grader bytes")
    destination = _output_directory(directory, manifest, manifest_path)
    capture = Path(manifest.archive_directory) / "captured"

    def check() -> None:
        _selected_bytes(manifest_path, manifest_sha256, 64_000)
        _selected_bytes(report_path, report_sha256, 2_000_000)
        for ordinal in range(1, 13):
            selected_cell(ordinal)
        if any(_acceptance_digest(task.acceptance) != criteria[task.id] for task in by_id.values()):
            raise ValueError("Independent criterion changed during regrade")
        check_runtime_selection(
            capture / "runtime-inventory.json", manifest.runtime_inventory_sha256
        )
        runtime = json.loads(
            _selected_bytes(
                capture / "runtime-inventory.json", manifest.runtime_inventory_sha256, 2_000_000
            )
        )
        if Path(sys.executable) != Path(runtime["executables"]["python"]["selected_path"]):
            raise ValueError("Regrade requires the selected existing interpreter environment")
        check_dependency_selection(
            capture / "dependency-files.json", manifest.dependency_files_sha256
        )
        if environment_bytes(manifest.command_environment) != load_environment_pin(
            capture / "command-environment.json", manifest.command_environment_sha256
        ):
            raise ValueError("Regrade command environment differs from retained selection")

    check()
    destination.mkdir(mode=0o700, exist_ok=False)
    try:
        cells = []
        for index, state in enumerate(ledger.cells):
            check()
            if index >= len(rows):
                cell = RegradedCell(cell=state.cell, ledger_status=state.status, status="no_result")
            else:
                row = rows[index]
                task = by_id[state.cell.task_id]
                source = (
                    Path(manifest.staged_directory) / f"cell-{state.cell.ordinal:02d}/workspace"
                )
                before, modes = _artifact(source)
                artifact_digest = hashlib.sha256(
                    json.dumps({"files": before, "modes": modes}, sort_keys=True).encode()
                ).hexdigest()
                if artifact_digest != pins[state.cell.ordinal]:
                    raise ValueError("Artifact differs from independently selected snapshot")
                with _temporary_copy() as copied:
                    used = 0
                    for name, expected in before.items():
                        path = source / name
                        if path.resolve(strict=True) != path:
                            raise ValueError("Regrade source path changed")
                        content, file_mode = _read_regular(path, 1_000_000 - used)
                        used += len(content)
                        if (
                            hashlib.sha256(content).hexdigest() != expected
                            or file_mode != modes[name]
                        ):
                            raise ValueError("Regrade source changed while copying")
                        target = copied / name
                        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                        target.write_bytes(content)
                        target.chmod(file_mode)
                    if _artifact(copied) != (before, modes) or _artifact(source) != (before, modes):
                        raise ValueError("Copied artifact differs from source snapshot")
                    result = await acceptance_check(
                        task, copied, command_env=manifest.command_environment
                    )
                    check()
                    if _artifact(source) != (before, modes):
                        raise ValueError("Original artifact changed during regrade")
                cell = RegradedCell(
                    cell=state.cell,
                    ledger_status=state.status,
                    status="graded",
                    recorded_passed=row.passed,
                    recorded_completion=row.completion_status,
                    recorded_acceptance_exit_code=row.acceptance_exit_code,
                    regrade_exit_code=result.exit_code,
                    acceptance_output=result.stdout + result.stderr,
                    acceptance_output_truncated=result.stdout_truncated or result.stderr_truncated,
                    matches_recorded_exit_code=result.exit_code == row.acceptance_exit_code,
                    artifact_sha256=artifact_digest,
                    scoped_source_unchanged=True,
                )
            _save(
                destination / f"cell-{state.cell.ordinal:02d}.json",
                cell.model_dump_json(indent=2).encode(),
            )
            cells.append(cell)
        check()
        summary = FreshRegrades(
            manifest_sha256=manifest_sha256,
            report_sha256=report_sha256,
            criteria_sha256=criteria,
            source_report_mode=mode,
            grader_revision=_harness_revision(),
            cells=cells,
        )
        _save(destination / "regrades.json", summary.model_dump_json(indent=2).encode())
        return summary
    except BaseException as error:
        try:
            _save(
                destination / "failure.json",
                json.dumps({"status": "failed", "error_type": type(error).__name__}).encode(),
            )
        except BaseException as secondary:
            error.add_note(f"Regrade failure recording also failed: {type(secondary).__name__}")
        raise
