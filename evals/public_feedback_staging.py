"""Preserve verified mock inputs and stage fresh exact-path fixtures; no launch authority."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from evals.heldout_comparison import ProtocolModel
from evals.heldout_guard import _read_regular
from evals.heldout_inputs import _save
from evals.public_feedback import PLAN_SHA256
from evals.public_feedback_archive import verify_feedback_capture
from evals.public_feedback_rehearsal import FeedbackRehearsal
from evals.public_feedback_workspaces import prepare_feedback_cell, verify_feedback_cell
from evals.public_inputs import Digest
from evals.public_staging import _inventory, _private_directory
from evals.public_workspaces import _destination


class StagedFeedback(ProtocolModel):
    format: Literal["staged-public-feedback-v1"] = "staged-public-feedback-v1"
    plan_sha256: Digest
    capture_index_sha256: Digest
    archive: str
    inference_authorized: Literal[False] = False


def _paths(directory: Path, archive: Path) -> tuple[Path, Path]:
    source, target = _destination(directory), _destination(archive)
    if source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Feedback staged and archive paths overlap")
    return source, target


def stage_feedback_capture(directory: Path, archive: Path, *, expected_index_sha256: str) -> None:
    source, target = _paths(directory, archive)
    index = verify_feedback_capture(source, expected_index_sha256=expected_index_sha256)
    target.mkdir(mode=0o700, exist_ok=False)
    _inventory(target, set())
    # Preserve the whole consumed tree, including unselected caches. No reset/delete.
    source.rename(target / "captured")
    verify_feedback_capture(
        target / "captured",
        expected_index_sha256=expected_index_sha256,
        original_directory=source,
    )
    _private_directory(target)
    _inventory(target, {"captured"})
    source.mkdir(mode=0o700, exist_ok=False)
    for row in index.cells:
        container = source / f"cell-{row.cell.ordinal:02d}"
        prepare_feedback_cell(row.cell.ordinal, container)
        if str(verify_feedback_cell(container, ordinal=row.cell.ordinal)) != row.workspace_root:
            raise ValueError("Feedback fresh workspace differs from captured absolute path")
    _inventory(source, {f"cell-{row.cell.ordinal:02d}" for row in index.cells})
    verify_feedback_capture(
        target / "captured",
        expected_index_sha256=expected_index_sha256,
        original_directory=source,
    )
    for row in index.cells:
        verify_feedback_cell(source / f"cell-{row.cell.ordinal:02d}", ordinal=row.cell.ordinal)
    _private_directory(source)
    _private_directory(target)
    _inventory(target, {"captured"})
    _save(
        source / "staged.json",
        StagedFeedback(
            plan_sha256=PLAN_SHA256,
            capture_index_sha256=expected_index_sha256,
            archive=str(target / "captured"),
        )
        .model_dump_json(indent=2)
        .encode(),
    )


def verify_staged_feedback(
    directory: Path, archive: Path, *, expected_index_sha256: str
) -> FeedbackRehearsal:
    source, target = _paths(directory, archive)
    _private_directory(source)
    _private_directory(target)
    _inventory(target, {"captured"})
    raw, mode = _read_regular(source / "staged.json", 4096)
    expected = StagedFeedback(
        plan_sha256=PLAN_SHA256,
        capture_index_sha256=expected_index_sha256,
        archive=str(target / "captured"),
    )
    if mode != 0o600 or StagedFeedback.model_validate_json(raw) != expected:
        raise ValueError("Feedback staged receipt changed")
    index = verify_feedback_capture(
        target / "captured",
        expected_index_sha256=expected_index_sha256,
        original_directory=source,
    )
    _inventory(source, {"staged.json", *(f"cell-{row.cell.ordinal:02d}" for row in index.cells)})
    for row in index.cells:
        verify_feedback_cell(source / f"cell-{row.cell.ordinal:02d}", ordinal=row.cell.ordinal)
    return index
