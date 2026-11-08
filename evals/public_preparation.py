"""One-shot offline preparation with the selected existing interpreter; never installs."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from evals.heldout_dependency_files import observe_files, record_files
from evals.heldout_guard import _read_regular
from evals.heldout_inputs import _save
from evals.heldout_model_copy import prepare_image
from evals.heldout_preparation import check_existing_interpreter
from evals.heldout_runtime_inventory import observe_runtime, record_runtime
from evals.public_envelope import PLAN_SHA256, validate_public_plan
from evals.public_execution_manifest import PublicExecutionManifest
from evals.public_execution_preflight import preflight_public as preflight
from evals.public_inputs import capture_public_inputs
from evals.public_staging import stage_public_capture
from evals.public_workspaces import _destination as _owned_location
from evals.real_run import _harness_revision, _task_command_environment


def prepare_public_candidate(
    directory: Path,
    *,
    expected_python: Path,
    expected_commit: str,
    executables: dict[str, Path],
    dependency_files: list[Path],
    source_models: Path,
    command_env: Mapping[str, str],
) -> PublicExecutionManifest:
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
    plan = validate_public_plan()
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
        index = asyncio.run(
            capture_public_inputs(
                cells,
                command_env=environment,
                runtime_inventory=destination / "runtime.json",
                runtime_sha256=runtime_sha,
                dependency_manifest=destination / "dependencies.json",
                dependency_sha256=dependency_sha,
            )
        )
        index_sha = digest(cells / "index.json")
        stage_public_capture(cells, destination / "archive", expected_index_sha256=index_sha)
        manifest = PublicExecutionManifest(
            format="public-pilot-execution-v1",
            runtime_commit=expected_commit,
            plan_sha256=PLAN_SHA256,
            settings_sha256=index.settings_sha256,
            command_environment_sha256=index.command_environment_sha256,
            capture_index_sha256=index_sha,
            runtime_inventory_sha256=runtime_sha,
            dependency_files_sha256=dependency_sha,
            image_receipt_sha256=digest(destination / "image/prepared-image.json"),
            binary_sha256=initial_runtime["executables"]["ollama"]["sha256"],
            staged_directory=str(cells),
            archive_directory=str(destination / "archive"),
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
                    "format": "offline-public-preparation-v1",
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
