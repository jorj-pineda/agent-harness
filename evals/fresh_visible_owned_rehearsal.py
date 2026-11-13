"""Synchronous owned metadata lifecycle around mandatory-mock fresh-cell capture."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from itertools import islice
from pathlib import Path
from typing import Literal

from pydantic import Field

from evals.fresh_visible_rehearsal import FreshRehearsal, _FreshCapture
from evals.fresh_visible_workspaces import SELECTION_SHA256, FreshCell
from evals.heldout_backend_rehearsal import owned_backend
from evals.heldout_comparison import validate_plan
from evals.heldout_dependency_files import record_files
from evals.heldout_execution_manifest import Commit, Digest, FrozenModel
from evals.heldout_guard import _read_regular
from evals.heldout_inputs import _save, check_dependency_selection, check_runtime_selection
from evals.heldout_model_copy import prepare_image
from evals.heldout_preparation import check_existing_interpreter
from evals.heldout_runtime_inventory import record_runtime
from evals.public_envelope import _digest
from evals.public_workspaces import _destination
from evals.real_run import _harness_revision


class OwnedFreshCell(FrozenModel):
    cell: FreshCell
    image_receipt_sha256: Digest
    session_sha256: Digest
    cleanup_sha256: Digest


class OwnedFreshRehearsal(FrozenModel):
    format: Literal["fresh-visible-owned-mock-rehearsal-v1"] = (
        "fresh-visible-owned-mock-rehearsal-v1"
    )
    runtime_commit: Commit
    selection_sha256: Digest
    capture_index_sha256: Digest
    report_sha256: Digest
    runtime_inventory_sha256: Digest
    dependency_files_sha256: Digest
    cells: list[OwnedFreshCell] = Field(min_length=12, max_length=12)
    mock_generation_requests: Literal[36] = 36
    generation_forwarded: Literal[0] = 0
    inference_ran: Literal[False] = False
    parent_approval_issued: Literal[False] = False
    evidence_label: Literal["scripted-real-tools-rehearsal"] = "scripted-real-tools-rehearsal"


def _private(path: Path) -> bytes:
    raw, mode = _read_regular(path, 2_000_000)
    if mode != 0o600:
        raise ValueError("Owned fresh evidence must stay private")
    return raw


def rehearse_fresh_owned(
    directory: Path,
    *,
    expected_python: Path,
    expected_commit: str,
    executables: dict[str, Path],
    dependency_files: list[Path],
    source_models: Path,
    command_env: Mapping[str, str],
) -> OwnedFreshRehearsal:
    """New consumed rehearsal only, using an existing external Python and real owned backends.

    Shared ownership sends metadata GETs; generation stays in the capture driver's
    internal mock. No actual/synthetic inference receipt or execution authority is issued.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise ValueError("Enter owned fresh rehearsal outside an active event loop")
    check_existing_interpreter(expected_python)
    observed = {"commit": expected_commit, "dirty": False}
    if _harness_revision() != observed:
        raise ValueError("Select the clean retained owned rehearsal source")
    selected = dict(executables)
    dependencies = list(dependency_files)
    if selected.get("python") != expected_python or "ollama" not in selected:
        raise ValueError("Select existing Python and Ollama executables")
    destination = _destination(directory)
    protected = [source_models.resolve(strict=True), *selected.values(), *dependencies]
    if (
        directory != destination
        or directory.resolve() != directory
        or directory.is_symlink()
        or any(
            path.resolve().is_relative_to(destination) or destination.is_relative_to(path.resolve())
            for path in protected
        )
    ):
        raise ValueError("Owned rehearsal destination overlaps inputs or is not canonical")
    destination.mkdir(mode=0o700, exist_ok=False)
    capture = None
    try:
        runtime = destination / "runtime.json"
        dependency = destination / "dependencies.json"
        inventory = record_runtime(selected, runtime)
        record_files(dependencies, dependency)
        runtime_sha, dependency_sha = _digest(_private(runtime)), _digest(_private(dependency))
        binary = Path(inventory["executables"]["ollama"]["resolved_path"])
        model_digest = validate_plan().model_digest
        capture = _FreshCapture(
            destination / "capture",
            command_env=command_env,
            runtime_inventory=runtime,
            runtime_sha256=runtime_sha,
            dependency_manifest=dependency,
            dependency_sha256=dependency_sha,
        )
        if capture.runtime.commit != expected_commit:
            raise ValueError("Capture and owned source differ")
        resources = destination / "resources"
        resources.mkdir(mode=0o700)
        completed = []

        def check() -> None:
            if _harness_revision() != observed:
                raise ValueError("Owned fresh source changed")
            check_runtime_selection(runtime, runtime_sha)
            check_dependency_selection(dependency, dependency_sha)
            capture.check()
            for evidence in completed:
                prior = resources / f"cell-{evidence.cell.ordinal:02d}"
                for name, expected in (
                    ("image/prepared-image.json", evidence.image_receipt_sha256),
                    ("backend/session.json", evidence.session_sha256),
                    ("backend/cleanup.json", evidence.cleanup_sha256),
                ):
                    if _digest(_private(prior / name)) != expected:
                        raise ValueError("Owned fresh earlier settlement evidence changed")

        for cell in capture.cells:
            check()
            output = resources / f"cell-{cell.ordinal:02d}"
            output.mkdir(mode=0o700)
            prepare_image(source_models, output / "image", expected_manifest_sha256=model_digest)
            check()
            with owned_backend(
                output / "image", output / "backend", binary, keep_alive_seconds=300
            ) as session:
                # Listener ownership is checked before and after every mocked request.
                asyncio.run(capture.capture_cell(cell, check_owned=session.check_owned))
            # No next cell or success record until shared lifecycle verifies settlement.
            check()
            cleanup = json.loads(_private(output / "backend/cleanup.json"))
            session_record = json.loads(_private(output / "backend/session.json"))
            if (
                cleanup.get("passed") is not True
                or cleanup.get("settled") is not True
                or cleanup.get("forced") is not False
                or cleanup.get("returncode") != 0
                or cleanup.get("errors") != {}
                or session_record.get("format") != "heldout-owned-backend-session-v1"
                or session_record.get("inference_authorized") is not False
                or session_record.get("cleanup") != cleanup
            ):
                raise ValueError("Owned fresh settlement did not pass")
            completed.append(
                OwnedFreshCell(
                    cell=cell,
                    image_receipt_sha256=_digest(_private(output / "image/prepared-image.json")),
                    session_sha256=_digest(_private(output / "backend/session.json")),
                    cleanup_sha256=_digest(_private(output / "backend/cleanup.json")),
                )
            )
        if {p.name for p in islice(resources.iterdir(), 13)} != {
            f"cell-{i:02d}" for i in range(1, 13)
        }:
            raise ValueError("Owned fresh resource inventory changed")
        check()
        captured: FreshRehearsal = capture.finish()
        summary = OwnedFreshRehearsal(
            runtime_commit=expected_commit,
            selection_sha256=SELECTION_SHA256,
            capture_index_sha256=_digest(_private(destination / "capture/index.json")),
            report_sha256=captured.report_sha256,
            runtime_inventory_sha256=runtime_sha,
            dependency_files_sha256=dependency_sha,
            cells=completed,
        )
        _save(destination / "owned-rehearsal.json", summary.model_dump_json(indent=2).encode())
        return summary
    except BaseException as error:
        if capture is not None:
            capture.fail(error)
        try:
            _save(
                destination / "owned-failure.json",
                json.dumps(
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "inference_authorized": False,
                        "retry_authorized": False,
                    }
                ).encode(),
            )
        except BaseException as secondary:
            error.add_note(f"Owned fresh failure recording also failed: {type(secondary).__name__}")
        raise
