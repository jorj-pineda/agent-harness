"""Read-only execution/parent-receipt binding; not provenance or launch verification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evals.heldout_guard import _read_regular
from evals.real_run import _task_command_environment

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Commit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)


class ExecutionManifest(FrozenModel):
    format: Literal["heldout-execution-manifest-v1"]
    runtime_commit: Commit
    protocol_sha256: Digest
    capture_index_sha256: Digest
    runtime_inventory_sha256: Digest
    dependency_files_sha256: Digest
    image_receipt_sha256: Digest
    binary_sha256: Digest
    staged_directory: str = Field(min_length=1, max_length=4096)
    capture_directory: str = Field(min_length=1, max_length=4096)
    output_directory: str = Field(min_length=1, max_length=4096)
    image_directory: str = Field(min_length=1, max_length=4096)
    ollama_binary: str = Field(min_length=1, max_length=4096)
    command_environment: dict[str, str] = Field(min_length=1, max_length=10)
    inference_authorized: Literal[False]

    @field_validator(
        "staged_directory",
        "capture_directory",
        "output_directory",
        "image_directory",
        "ollama_binary",
    )
    @classmethod
    def absolute_path(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts or "\x00" in value:
            raise ValueError("Execution paths must be explicit absolute paths")
        return value

    @field_validator("command_environment")
    @classmethod
    def effective_environment(cls, value: dict[str, str]) -> dict[str, str]:
        effective = _task_command_environment(value)
        if effective != value:
            raise ValueError("Manifest must include effective pytest startup flags")
        return value


class ParentApproval(FrozenModel):
    format: Literal["heldout-parent-approval-v1"]
    approval_id: str = Field(min_length=1, max_length=256)
    manifest_sha256: Digest
    runtime_commit: Commit
    freeze_reviewed_head: Commit
    freeze_merge_commit: Commit
    exact_head_review_passed: Literal[True]
    ci_passed: Literal[True]
    merged: Literal[True]
    inference_authorized: Literal[True]


def _selected_bytes(path: Path, digest: str, limit: int) -> bytes:
    raw, mode = _read_regular(path, limit)
    if mode != 0o600 or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Execution artifact bytes/mode differ from selected hash")
    return raw


def load_execution(
    manifest_path: Path,
    *,
    manifest_sha256: str,
    approval_path: Path,
    approval_sha256: str,
) -> tuple[ExecutionManifest, ParentApproval]:
    """Validate bindings only; caller owns receipt provenance and every fresh gate."""
    manifest_raw = _selected_bytes(manifest_path, manifest_sha256, 64_000)
    manifest = ExecutionManifest.model_validate_json(manifest_raw)
    approval_raw = _selected_bytes(approval_path, approval_sha256, 16_000)
    approval = ParentApproval.model_validate_json(approval_raw)
    if (
        approval.manifest_sha256 != manifest_sha256
        or approval.runtime_commit != manifest.runtime_commit
    ):
        raise ValueError("Parent approval does not bind this exact execution manifest/runtime")
    if (
        _selected_bytes(manifest_path, manifest_sha256, 64_000) != manifest_raw
        or _selected_bytes(approval_path, approval_sha256, 16_000) != approval_raw
    ):
        raise ValueError("Execution artifacts changed during binding validation")
    return manifest, approval
