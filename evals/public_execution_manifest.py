"""Public-pilot artifact binding only; affirmative receipt fields are not authentication."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from evals.heldout_execution_manifest import Commit, Digest, FrozenModel, _selected_bytes
from evals.public_envelope import PLAN_SHA256
from evals.real_run import _task_command_environment


class PublicExecutionManifest(FrozenModel):
    format: Literal["public-pilot-execution-v1"]
    runtime_commit: Commit
    plan_sha256: Digest
    capture_index_sha256: Digest
    settings_sha256: Digest
    command_environment_sha256: Digest
    runtime_inventory_sha256: Digest
    dependency_files_sha256: Digest
    image_receipt_sha256: Digest
    binary_sha256: Digest
    staged_directory: str = Field(min_length=1, max_length=4096)
    archive_directory: str = Field(min_length=1, max_length=4096)
    output_directory: str = Field(min_length=1, max_length=4096)
    image_directory: str = Field(min_length=1, max_length=4096)
    ollama_binary: str = Field(min_length=1, max_length=4096)
    command_environment: dict[str, str] = Field(min_length=1, max_length=10)
    coding_prompt_profile: Literal["direct"] = "direct"
    coding_toolset: Literal["whole_file_check"] = "whole_file_check"
    keep_alive_seconds: Literal[300] = 300
    inference_authorized: Literal[False]

    @model_validator(mode="after")
    def selected_plan(self) -> PublicExecutionManifest:
        if self.plan_sha256 != PLAN_SHA256:
            raise ValueError("Public execution requires the exact reviewed public plan")
        return self

    @field_validator(
        "staged_directory",
        "archive_directory",
        "output_directory",
        "image_directory",
        "ollama_binary",
    )
    @classmethod
    def absolute_path(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts or "\x00" in value:
            raise ValueError("Public execution paths must be explicit absolute paths")
        return value

    @field_validator("command_environment")
    @classmethod
    def effective_environment(cls, value: dict[str, str]) -> dict[str, str]:
        if _task_command_environment(value) != value:
            raise ValueError("Manifest must include effective pytest startup flags")
        return value


class PublicParentApproval(FrozenModel):
    format: Literal["public-pilot-parent-approval-v1"]
    approval_id: str = Field(min_length=1, max_length=256)
    manifest_sha256: Digest
    runtime_commit: Commit
    freeze_reviewed_head: Commit
    freeze_merge_commit: Commit
    exact_head_review_passed: Literal[True]
    ci_passed: Literal[True]
    merged: Literal[True]
    inference_authorized: Literal[True]


def load_public_execution(
    manifest_path: Path, *, manifest_sha256: str, approval_path: Path, approval_sha256: str
) -> tuple[PublicExecutionManifest, PublicParentApproval]:
    """Bind selected private bytes; no provenance check, admission or resource allocation.

    The trusted parent must independently establish actual authority, exact freeze
    review/CI/merge and fresh launch gates. True JSON fields do not prove those facts.
    """
    manifest_raw = _selected_bytes(manifest_path, manifest_sha256, 64_000)
    approval_raw = _selected_bytes(approval_path, approval_sha256, 16_000)
    manifest = PublicExecutionManifest.model_validate_json(manifest_raw)
    approval = PublicParentApproval.model_validate_json(approval_raw)
    if (
        approval.manifest_sha256 != manifest_sha256
        or approval.runtime_commit != manifest.runtime_commit
    ):
        raise ValueError("Public parent receipt does not bind the selected manifest/runtime")
    if (
        _selected_bytes(manifest_path, manifest_sha256, 64_000) != manifest_raw
        or _selected_bytes(approval_path, approval_sha256, 16_000) != approval_raw
    ):
        raise ValueError("Public execution artifacts changed during binding validation")
    return manifest, approval
