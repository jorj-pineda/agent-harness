"""Single-use artifact admission; no provider, backend or approval generator."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from evals.heldout_execution_manifest import (
    ExecutionManifest,
    ParentApproval,
    _selected_bytes,
    load_execution,
)
from evals.heldout_execution_preflight import preflight
from evals.heldout_inputs import _save
from evals.heldout_staging import CaptureIndex
from evals.heldout_workspaces import _owned_location


@dataclass(frozen=True)
class ExecutionAdmission:
    directory: Path
    manifest: ExecutionManifest
    approval: ParentApproval
    capture: CaptureIndex


def admit_execution(
    manifest_path: Path,
    *,
    manifest_sha256: str,
    approval_path: Path,
    approval_sha256: str,
    verify_parent: Callable[[ExecutionManifest, ParentApproval], None],
) -> ExecutionAdmission:
    """Reserve one output only after trusted provenance and semantic checks.

    verify_parent MUST raise unless actual parent authority, exact reviewed/merged
    freeze and CI have been independently established. This callback has no default
    and is trusted operator code, not a JSON authenticator or model supervisor.
    Failure after allocation retains partial evidence, permanently consuming output.
    """
    manifest, approval = load_execution(
        manifest_path,
        manifest_sha256=manifest_sha256,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
    )
    # Keep the selected source files outside every managed tree. Canonical paths
    # make the later preserved-byte reads select the same named locations.
    roots = [
        Path(value)
        for value in (
            manifest.staged_directory,
            manifest.capture_directory,
            manifest.output_directory,
            manifest.image_directory,
        )
    ]
    for path in (manifest_path, approval_path):
        if (
            not path.is_absolute()
            or path.is_symlink()
            or path.resolve(strict=True) != path
            or _owned_location(path) != path
            or any(path.is_relative_to(root) for root in roots)
        ):
            raise ValueError(
                "Execution selection files must be canonical and outside managed trees"
            )
    if manifest_path == approval_path:
        raise ValueError("Manifest and parent approval must be separately selected files")
    # Pydantic frozen models still contain mutable dicts; don't let callback
    # mutations change the values used by preflight or returned to the runner.
    verify_parent(manifest.model_copy(deep=True), approval.model_copy(deep=True))
    capture = preflight(manifest)
    manifest_raw = _selected_bytes(manifest_path, manifest_sha256, 64_000)
    approval_raw = _selected_bytes(approval_path, approval_sha256, 16_000)
    directory = _owned_location(Path(manifest.output_directory))
    if directory != Path(manifest.output_directory) or directory.resolve() != directory:
        raise ValueError("Execution output changed after preflight")
    directory.mkdir(mode=0o700, exist_ok=False)
    try:
        _save(directory / "manifest.json", manifest_raw)
        _save(directory / "parent-approval.json", approval_raw)
        # Refuse source or preservation drift without rebinding after allocation.
        _selected_bytes(manifest_path, manifest_sha256, 64_000)
        _selected_bytes(approval_path, approval_sha256, 16_000)
        _selected_bytes(directory / "manifest.json", manifest_sha256, 64_000)
        _selected_bytes(directory / "parent-approval.json", approval_sha256, 16_000)
        _save(
            directory / "admission.json",
            json.dumps(
                {
                    "format": "heldout-execution-admission-v1",
                    "status": "admitted",
                    "manifest_sha256": manifest_sha256,
                    "approval_sha256": approval_sha256,
                    "approval_id": approval.approval_id,
                    "runtime_commit": manifest.runtime_commit,
                    "capture_index_sha256": manifest.capture_index_sha256,
                    "execution_completed": False,
                    "inference_performed_by_admission": False,
                },
                indent=2,
            ).encode(),
        )
    except BaseException as error:
        try:
            _save(
                directory / "admission-failure.json",
                json.dumps({"status": "failed", "error_type": type(error).__name__}).encode(),
            )
        except BaseException as recording_error:
            error.add_note(
                f"Admission failure recording also failed: {type(recording_error).__name__}"
            )
        raise
    return ExecutionAdmission(directory, manifest, approval, capture)
