"""Read-only feedback execution preflight; no admission, services or launch authority."""

from __future__ import annotations

import stat
from pathlib import Path

from evals.heldout_backend_rehearsal import _prepared_models
from evals.heldout_command_environment import environment_bytes, load_environment_pin
from evals.heldout_execution_manifest import _selected_bytes
from evals.heldout_inputs import check_dependency_selection, check_runtime_selection
from evals.heldout_model_image import _hash_blob
from evals.heldout_operator_rehearsal import _runtime_matches
from evals.heldout_settings import check_settings, load_settings_pin
from evals.public_envelope import public_settings
from evals.public_feedback_execution_manifest import FeedbackExecutionManifest
from evals.public_feedback_rehearsal import FeedbackRehearsal
from evals.public_feedback_staging import verify_staged_feedback
from evals.public_workspaces import _destination


def feedback_execution_paths(
    manifest: FeedbackExecutionManifest,
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
            raise ValueError("Feedback execution directories must be canonical owned paths")
    for offset, path in enumerate(roots):
        if any(
            path.is_relative_to(other) or other.is_relative_to(path)
            for other in roots[offset + 1 :]
        ):
            raise ValueError("Feedback execution directories overlap")
    binary = Path(manifest.ollama_binary)
    if binary.is_symlink() or binary.resolve(strict=True) != binary:
        raise ValueError("Feedback execution binary must be a canonical regular file")
    if any(binary.is_relative_to(path) for path in roots):
        raise ValueError("Feedback execution binary overlaps managed directories")
    if roots[2].exists():
        raise ValueError("Feedback execution output already exists; no retry or reuse")
    return roots[0], roots[1], roots[2], roots[3], binary


def preflight_feedback(manifest: FeedbackExecutionManifest) -> FeedbackRehearsal:
    """Observe fresh selected pins without consuming fixtures or constructing providers.

    This can prepare a freeze before approval. It is neither a lease nor a proof
    of authority; fresh ownership, trusted approval and exclusive admission remain.
    """
    manifest = FeedbackExecutionManifest.model_validate(manifest.model_dump())
    staged, archive, _, image, binary = feedback_execution_paths(manifest)
    runtime = {"commit": manifest.runtime_commit, "dirty": False}
    _runtime_matches(runtime)
    index = verify_staged_feedback(
        staged, archive, expected_index_sha256=manifest.capture_index_sha256
    )
    if index.runtime.model_dump() != runtime or any(
        getattr(index, field) != getattr(manifest, field)
        for field in (
            "plan_sha256",
            "settings_sha256",
            "command_environment_sha256",
            "runtime_inventory_sha256",
            "dependency_files_sha256",
        )
    ):
        raise ValueError("Feedback capture differs from execution runtime/settings/artifact pins")
    capture = archive / "captured"
    check_settings(
        public_settings(),
        load_settings_pin(capture / "settings.json", expected_sha256=manifest.settings_sha256),
    )
    if environment_bytes(manifest.command_environment) != load_environment_pin(
        capture / "command-environment.json", manifest.command_environment_sha256
    ):
        raise ValueError("Feedback execution command environment differs from capture")
    check_runtime_selection(capture / "runtime-inventory.json", manifest.runtime_inventory_sha256)
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
        raise ValueError("Feedback execution binary type/size/mode is invalid")
    if _hash_blob(binary, info.st_size)["sha256"] != manifest.binary_sha256:
        raise ValueError("Feedback execution binary differs from selected hash")
    if _selected_bytes(receipt, manifest.image_receipt_sha256, 64_000) != receipt_raw:
        raise ValueError("Feedback prepared image receipt changed during preflight")
    _runtime_matches(runtime)
    feedback_execution_paths(manifest)
    verify_staged_feedback(staged, archive, expected_index_sha256=manifest.capture_index_sha256)
    return index
