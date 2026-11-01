"""Read-only semantic checks of a selected execution manifest; no launch authority."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path

from evals.heldout_backend_rehearsal import _prepared_models
from evals.heldout_command_environment import environment_bytes, load_environment_pin
from evals.heldout_comparison import PLAN_PATH, validate_plan
from evals.heldout_execution_manifest import ExecutionManifest, _selected_bytes
from evals.heldout_inputs import check_dependency_selection, check_runtime_selection
from evals.heldout_model_image import _hash_blob
from evals.heldout_operator_rehearsal import _runtime_matches, _staged_inventory
from evals.heldout_settings import check_settings, load_settings_pin, protocol_settings
from evals.heldout_staging import CaptureIndex, verify_capture
from evals.heldout_workspaces import _owned_location, verify_prepared


def execution_paths(manifest: ExecutionManifest) -> tuple[Path, Path, Path, Path, Path]:
    """Refuse aliases, overlap and occupied output before inspecting large artifacts."""
    directories = [
        Path(value)
        for value in (
            manifest.staged_directory,
            manifest.capture_directory,
            manifest.output_directory,
            manifest.image_directory,
        )
    ]
    for path in directories:
        if path.is_symlink() or _owned_location(path) != path or path.resolve() != path:
            raise ValueError("Execution directories must be canonical owned paths")
    for offset, path in enumerate(directories):
        for other in directories[offset + 1 :]:
            if path.is_relative_to(other) or other.is_relative_to(path):
                raise ValueError("Execution directories overlap")
    binary = Path(manifest.ollama_binary)
    if binary.is_symlink() or binary.resolve(strict=True) != binary:
        raise ValueError("Execution binary must be a canonical regular file")
    if any(binary.is_relative_to(path) for path in directories):
        raise ValueError("Execution binary overlaps managed directories")
    if directories[2].exists():
        raise ValueError("Execution output already exists; no retry or reuse")
    return *directories, binary


def preflight(manifest: ExecutionManifest) -> CaptureIndex:
    """Observe current pins without allocating output, admitting cells or starting a child.

    Callable before approval to prepare an exact freeze. The caller still owns
    approval provenance, fresh resource gates and exclusive execution admission.
    """
    staged, capture, _, image, binary = execution_paths(manifest)
    runtime = {"commit": manifest.runtime_commit, "dirty": False}
    _runtime_matches(runtime)
    if hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest() != manifest.protocol_sha256:
        raise ValueError("Execution protocol differs from selected hash")
    plan = validate_plan()
    index = verify_capture(
        capture, expected_index_sha256=manifest.capture_index_sha256, original_directory=staged
    )
    if (
        index.runtime.model_dump() != runtime
        or index.runtime_inventory_sha256 != manifest.runtime_inventory_sha256
        or index.dependency_files_sha256 != manifest.dependency_files_sha256
        or index.settings_sha256 is None
        or index.command_environment_sha256 is None
    ):
        raise ValueError("Capture lacks matching execution runtime/settings/environment pins")
    expected_settings = load_settings_pin(
        capture / "settings.json", expected_sha256=index.settings_sha256
    )
    check_settings(protocol_settings(plan), expected_settings)
    expected_env = load_environment_pin(
        capture / "command-environment.json", index.command_environment_sha256
    )
    if environment_bytes(manifest.command_environment) != expected_env:
        raise ValueError("Execution command environment differs from capture")
    check_runtime_selection(capture / "runtime-inventory.json", manifest.runtime_inventory_sha256)
    check_dependency_selection(capture / "dependency-files.json", manifest.dependency_files_sha256)
    _staged_inventory(staged, capture, manifest.capture_index_sha256, index.cells)
    for cell in index.cells:
        verify_prepared(staged / f"cell-{cell.ordinal:02d}", task_id=cell.task_id)
    receipt = image / "prepared-image.json"
    original_receipt = _selected_bytes(receipt, manifest.image_receipt_sha256, 64_000)
    _prepared_models(image)
    info = binary.stat()
    if (
        not stat.S_ISREG(info.st_mode)
        or not info.st_mode & 0o111
        or not 1 <= info.st_size <= 256_000_000
    ):
        raise ValueError("Execution binary type/size/mode is invalid")
    if _hash_blob(binary, info.st_size)["sha256"] != manifest.binary_sha256:
        raise ValueError("Execution binary differs from selected hash")
    if _selected_bytes(receipt, manifest.image_receipt_sha256, 64_000) != original_receipt:
        raise ValueError("Prepared image receipt changed during preflight")
    _runtime_matches(runtime)
    execution_paths(manifest)
    return index
