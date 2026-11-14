"""Preserve verified mock inputs and stage fresh exact-path fixtures; no launch authority."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from evals.fresh_visible_archive import verify_fresh_capture
from evals.fresh_visible_rehearsal import FreshRehearsal
from evals.fresh_visible_workspaces import SELECTION_SHA256, prepare_fresh_cell, verify_fresh_cell
from evals.heldout_comparison import ProtocolModel
from evals.heldout_guard import _read_regular
from evals.heldout_inputs import _save
from evals.public_inputs import Digest
from evals.public_staging import _inventory, _private_directory
from evals.public_workspaces import _destination


class StagedFresh(ProtocolModel):
    format: Literal["staged-fresh-visible-v1"] = "staged-fresh-visible-v1"
    selection_sha256: Digest
    capture_index_sha256: Digest
    archive: str
    inference_authorized: Literal[False] = False


def _paths(directory: Path, archive: Path) -> tuple[Path, Path]:
    source, target = _destination(directory), _destination(archive)
    if source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Fresh staged and archive paths overlap")
    return source, target


def stage_fresh_capture(directory: Path, archive: Path, *, expected_index_sha256: str) -> None:
    source, target = _paths(directory, archive)
    index = verify_fresh_capture(source, expected_index_sha256=expected_index_sha256)
    target.mkdir(mode=0o700, exist_ok=False)
    _inventory(target, set())
    # Preserve the whole consumed tree, including unselected caches. No reset/delete.
    source.rename(target / "captured")
    verify_fresh_capture(
        target / "captured",
        expected_index_sha256=expected_index_sha256,
        original_directory=source,
    )
    _private_directory(target)
    _inventory(target, {"captured"})
    source.mkdir(mode=0o700, exist_ok=False)
    for row in index.cells:
        container = source / f"cell-{row.cell.ordinal:02d}"
        prepare_fresh_cell(row.cell.ordinal, container)
        if str(verify_fresh_cell(container, ordinal=row.cell.ordinal)) != row.workspace_root:
            raise ValueError("Fresh fresh workspace differs from captured absolute path")
    _inventory(source, {f"cell-{row.cell.ordinal:02d}" for row in index.cells})
    verify_fresh_capture(
        target / "captured",
        expected_index_sha256=expected_index_sha256,
        original_directory=source,
    )
    for row in index.cells:
        verify_fresh_cell(source / f"cell-{row.cell.ordinal:02d}", ordinal=row.cell.ordinal)
    _private_directory(source)
    _private_directory(target)
    _inventory(target, {"captured"})
    _save(
        source / "staged.json",
        StagedFresh(
            selection_sha256=SELECTION_SHA256,
            capture_index_sha256=expected_index_sha256,
            archive=str(target / "captured"),
        )
        .model_dump_json(indent=2)
        .encode(),
    )


def verify_staged_fresh(
    directory: Path, archive: Path, *, expected_index_sha256: str
) -> FreshRehearsal:
    source, target = _paths(directory, archive)
    _private_directory(source)
    _private_directory(target)
    _inventory(target, {"captured"})
    raw, mode = _read_regular(source / "staged.json", 4096)
    expected = StagedFresh(
        selection_sha256=SELECTION_SHA256,
        capture_index_sha256=expected_index_sha256,
        archive=str(target / "captured"),
    )
    if mode != 0o600 or StagedFresh.model_validate_json(raw) != expected:
        raise ValueError("Fresh staged receipt changed")
    index = verify_fresh_capture(
        target / "captured",
        expected_index_sha256=expected_index_sha256,
        original_directory=source,
    )
    _inventory(source, {"staged.json", *(f"cell-{row.cell.ordinal:02d}" for row in index.cells)})
    for row in index.cells:
        verify_fresh_cell(source / f"cell-{row.cell.ordinal:02d}", ordinal=row.cell.ordinal)
    return index
