"""One-shot offline preparation with the selected existing interpreter; never installs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from evals.heldout_comparison import PLAN_PATH, validate_plan
from evals.heldout_dependency_files import observe_files, record_files
from evals.heldout_execution_manifest import ExecutionManifest
from evals.heldout_execution_preflight import preflight
from evals.heldout_guard import _read_regular
from evals.heldout_inputs import _save, capture_inputs
from evals.heldout_model_copy import prepare_image
from evals.heldout_runtime_inventory import observe_runtime, record_runtime
from evals.heldout_staging import stage_capture
from evals.heldout_workspaces import _owned_location
from evals.real_run import _harness_revision, _task_command_environment

RUNTIME_ROOT = Path(__file__).resolve().parent.parent


def check_existing_interpreter(expected: Path) -> None:
    # resolve() alone is insufficient: two venv Python links can share one base
    # executable while selecting different environments/site-packages.
    local_environment = RUNTIME_ROOT / ".venv"
    if (
        local_environment.exists()
        or local_environment.is_symlink()
        or not expected.is_absolute()
        or ".." in expected.parts
        or Path(sys.executable) != expected
        or expected.is_relative_to(RUNTIME_ROOT)
        or Path(sys.prefix).resolve().is_relative_to(RUNTIME_ROOT)
    ):
        raise ValueError("Select the running existing interpreter outside the retained source tree")


def prepare_candidate(
    directory: Path,
    *,
    expected_python: Path,
    expected_commit: str,
    executables: dict[str, Path],
    dependency_files: list[Path],
    source_models: Path,
    command_env: Mapping[str, str],
) -> ExecutionManifest:
    """Preserve partial failures; no resolver, installer, backend, parent receipt or inference.

    The caller must start this module with the trusted existing absolute Python
    from the clean retained source. No CLI launches or changes the interpreter.
    """
    check_existing_interpreter(expected_python)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise ValueError("Enter preparation outside an active event loop")
    runtime = {"commit": expected_commit, "dirty": False}
    if _harness_revision() != runtime:
        raise ValueError("Preparation requires the selected clean retained revision")
    selected = dict(executables)
    dependencies = list(dependency_files)
    if selected.get("python") != expected_python or "ollama" not in selected:
        raise ValueError("Select the running Python and existing Ollama executable")
    environment = _task_command_environment(command_env)
    assert environment is not None
    plan = validate_plan()
    initial_runtime = observe_runtime(selected)
    initial_dependencies = observe_files(dependencies)
    binary = Path(initial_runtime["executables"]["ollama"]["resolved_path"])
    destination = _owned_location(directory)
    if directory != destination or directory.resolve() != directory or directory.is_symlink():
        raise ValueError("Preparation requires a canonical evaluation-owned directory")
    # Inputs must not be captured inside or overwritten by the new trial.
    protected = [source_models.resolve(strict=True), *selected.values(), *dependencies]
    if any(
        path.resolve().is_relative_to(destination) or destination.is_relative_to(path.resolve())
        for path in protected
    ):
        raise ValueError("Preparation destination overlaps selected inputs")
    destination.mkdir(mode=0o700, exist_ok=False)
    try:
        observed = record_runtime(selected, destination / "runtime.json")
        observed_dependencies = record_files(dependencies, destination / "dependencies.json")
        if observed != initial_runtime or observed_dependencies != initial_dependencies:
            raise ValueError("Selected runtime/dependencies changed during preparation")
        prepare_image(
            source_models, destination / "image", expected_manifest_sha256=plan.model_digest
        )

        def digest(path: Path) -> str:
            # Only bounded, privately written small reports are read here.
            raw, _mode = _read_regular(path, 2_000_000)
            return hashlib.sha256(raw).hexdigest()

        runtime_sha = digest(destination / "runtime.json")
        dependency_sha = digest(destination / "dependencies.json")
        cells = destination / "cells"
        asyncio.run(
            capture_inputs(
                cells,
                prepared=True,
                command_env=environment,
                runtime_inventory=destination / "runtime.json",
                runtime_sha256=runtime_sha,
                dependency_manifest=destination / "dependencies.json",
                dependency_sha256=dependency_sha,
            )
        )
        index_sha = digest(cells / "index.json")
        stage_capture(cells, destination / "archive", expected_index_sha256=index_sha)
        manifest = ExecutionManifest(
            format="heldout-execution-manifest-v1",
            runtime_commit=expected_commit,
            protocol_sha256=digest(PLAN_PATH),
            capture_index_sha256=index_sha,
            runtime_inventory_sha256=runtime_sha,
            dependency_files_sha256=dependency_sha,
            image_receipt_sha256=digest(destination / "image/prepared-image.json"),
            binary_sha256=initial_runtime["executables"]["ollama"]["sha256"],
            staged_directory=str(cells),
            capture_directory=str(destination / "archive/captured"),
            output_directory=str(destination / "run"),
            image_directory=str(destination / "image"),
            ollama_binary=str(binary),
            command_environment=environment,
            inference_authorized=False,
        )
        preflight(manifest)
        check_existing_interpreter(expected_python)
        if _harness_revision() != runtime:
            raise ValueError("Retained source changed during preparation")
        manifest_sha = _save(
            destination / "manifest-candidate.json", manifest.model_dump_json(indent=2).encode()
        )
        _save(
            destination / "preparation.json",
            json.dumps(
                {
                    "format": "offline-heldout-preparation-v1",
                    "manifest_sha256": manifest_sha,
                    "runtime": runtime,
                    "python": str(expected_python),
                    "inference_ran": False,
                    "parent_approval_issued": False,
                    "import_closure_verified": False,
                },
                indent=2,
            ).encode(),
        )
        return manifest
    except BaseException as error:
        try:
            _save(
                destination / "WITHDRAWN.json",
                b'{"eligible_for_live_launch":false,"reason":"preparation failed"}',
            )
            _save(
                destination / "preparation-failure.json",
                json.dumps({"error_type": type(error).__name__}).encode(),
            )
        except BaseException as recording_error:
            error.add_note(
                f"Preparation failure recording also failed: {type(recording_error).__name__}"
            )
        raise
