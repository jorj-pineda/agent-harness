"""Read-only fresh execution preflight; no admission, services or launch authority."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

from evals.fresh_visible_execution_manifest import FreshExecutionManifest
from evals.fresh_visible_rehearsal import FreshRehearsal
from evals.fresh_visible_staging import verify_staged_fresh
from evals.heldout_backend_rehearsal import _prepared_models
from evals.heldout_command_environment import (
    EnvironmentPin,
    environment_bytes,
    load_environment_pin,
)
from evals.heldout_execution_manifest import _selected_bytes
from evals.heldout_inputs import check_dependency_selection
from evals.heldout_model_image import _hash_blob
from evals.heldout_operator_rehearsal import _runtime_matches
from evals.heldout_runtime_guard import verify_runtime
from evals.heldout_settings import check_settings, load_settings_pin
from evals.public_envelope import public_settings
from evals.public_workspaces import _destination


def check_fresh_runtime(path: Path, expected_sha256: str) -> None:
    inventory = verify_runtime(path, expected_sha256=expected_sha256)
    if Path(sys.executable) != Path(inventory.executables["python"].selected_path):
        raise ValueError("Fresh execution requires the exact selected Python entrypoint")


def fresh_execution_paths(
    manifest: FreshExecutionManifest,
) -> tuple[Path, Path, Path, Path, Path]:
    roots = [
        Path(value)
        for value in (
            manifest.staged_directory,
            manifest.archive_directory,
            manifest.output_directory,
            manifest.image_directory,
        )
    ]
    for path in roots:
        if path.is_symlink() or _destination(path) != path or path.resolve() != path:
            raise ValueError("Fresh execution directories must be canonical owned paths")
    for offset, path in enumerate(roots):
        if any(
            path.is_relative_to(other) or other.is_relative_to(path)
            for other in roots[offset + 1 :]
        ):
            raise ValueError("Fresh execution directories overlap")
    binary = Path(manifest.ollama_binary)
    if binary.is_symlink() or binary.resolve(strict=True) != binary:
        raise ValueError("Fresh execution binary must be a canonical regular file")
    if any(binary.is_relative_to(path) for path in roots):
        raise ValueError("Fresh execution binary overlaps managed directories")
    if roots[2].exists():
        raise ValueError("Fresh execution output already exists; no retry or reuse")
    return roots[0], roots[1], roots[2], roots[3], binary


def preflight_fresh(manifest: FreshExecutionManifest) -> FreshRehearsal:
    """Observe fresh selected pins without consuming fixtures or constructing providers.

    This can prepare a freeze before approval. It is neither a lease nor a proof
    of authority; fresh ownership, trusted approval and exclusive admission remain.
    """
    manifest = FreshExecutionManifest.model_validate(manifest.model_dump())
    staged, archive, _, image, binary = fresh_execution_paths(manifest)
    runtime = {"commit": manifest.runtime_commit, "dirty": False}
    _runtime_matches(runtime)
    index = verify_staged_fresh(
        staged, archive, expected_index_sha256=manifest.capture_index_sha256
    )
    if index.runtime.model_dump() != runtime or any(
        getattr(index, field) != getattr(manifest, field)
        for field in (
            "selection_sha256",
            "settings_sha256",
            "command_environment_sha256",
            "runtime_inventory_sha256",
            "dependency_files_sha256",
        )
    ):
        raise ValueError("Fresh capture differs from execution runtime/settings/artifact pins")
    capture = archive / "captured"
    check_settings(
        public_settings(),
        load_settings_pin(capture / "settings.json", expected_sha256=manifest.settings_sha256),
    )
    if environment_bytes(manifest.command_environment) != load_environment_pin(
        capture / "command-environment.json", manifest.command_environment_sha256
    ):
        raise ValueError("Fresh execution command environment differs from capture")
    check_fresh_runtime(capture / "runtime-inventory.json", manifest.runtime_inventory_sha256)
    check_dependency_selection(capture / "dependency-files.json", manifest.dependency_files_sha256)
    receipt = image / "prepared-image.json"
    receipt_raw = _selected_bytes(receipt, manifest.image_receipt_sha256, 64_000)
    _prepared_models(image)
    info = binary.stat()
    if (
        not stat.S_ISREG(info.st_mode)
        or not info.st_mode & 0o111
        or not 1 <= info.st_size <= 256_000_000
    ):
        raise ValueError("Fresh execution binary type/size/mode is invalid")
    if _hash_blob(binary, info.st_size)["sha256"] != manifest.binary_sha256:
        raise ValueError("Fresh execution binary differs from selected hash")
    if _selected_bytes(receipt, manifest.image_receipt_sha256, 64_000) != receipt_raw:
        raise ValueError("Fresh prepared image receipt changed during preflight")
    _runtime_matches(runtime)
    fresh_execution_paths(manifest)
    verify_staged_fresh(staged, archive, expected_index_sha256=manifest.capture_index_sha256)
    return index


def select_fresh_execution(
    staged: Path,
    archive: Path,
    *,
    expected_index_sha256: str,
    expected_runtime_commit: str,
    image_directory: Path,
    image_receipt_sha256: str,
    ollama_binary: Path,
    binary_sha256: str,
    output_directory: Path,
) -> FreshExecutionManifest:
    """Select existing pinned artifacts and preflight without allocating or granting authority."""
    capture = verify_staged_fresh(staged, archive, expected_index_sha256=expected_index_sha256)
    environment = EnvironmentPin.model_validate_json(
        load_environment_pin(
            archive / "captured/command-environment.json", capture.command_environment_sha256
        )
    ).values
    manifest = FreshExecutionManifest(
        format="fresh-visible-execution-v1",
        runtime_commit=expected_runtime_commit,
        selection_sha256=capture.selection_sha256,
        capture_index_sha256=expected_index_sha256,
        settings_sha256=capture.settings_sha256,
        command_environment_sha256=capture.command_environment_sha256,
        runtime_inventory_sha256=capture.runtime_inventory_sha256,
        dependency_files_sha256=capture.dependency_files_sha256,
        image_receipt_sha256=image_receipt_sha256,
        binary_sha256=binary_sha256,
        staged_directory=str(staged),
        archive_directory=str(archive),
        output_directory=str(output_directory),
        image_directory=str(image_directory),
        ollama_binary=str(ollama_binary),
        command_environment=environment,
        inference_authorized=False,
    )
    preflight_fresh(manifest)
    return manifest
