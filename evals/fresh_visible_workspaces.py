"""Pinned fresh-development arm fixtures in the shared evaluator; no launch authority."""

from __future__ import annotations

import json
import os
import shutil
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Literal

from pydantic import Field

from evals.heldout_comparison import ProtocolModel
from evals.heldout_execution_manifest import Digest
from evals.heldout_guard import _read_regular
from evals.public_admission import PublicPilotDriftError
from evals.public_envelope import _digest, _tree_digest, policy_settings_values
from evals.public_workspaces import _destination
from evals.real_run import Task, load_tasks
from harness.config import Settings
from workspace import DisposableWorkspace, Workspace
from workspace.disposable import _snapshot

ROOT = Path(__file__).parent / "fresh_visible_tasks/v1"
SELECTION_SHA256 = "5208deb02901c0909598525b7458502b64f74b8c1aa6622471f73eb417b3bbd4"
TaskId = Literal["fv_lookup_option", "fv_merge_windows", "fv_header_pair"]


class FreshCell(ProtocolModel):
    ordinal: int = Field(ge=1, le=12)
    task_id: TaskId
    arm: Literal["weak", "stronger"]
    repeat: int = Field(ge=1, le=2)


class TaskPin(ProtocolModel):
    task_id: TaskId
    fixture_sha256: Digest
    acceptance_sha256: Digest
    reference_sha256: Digest
    stronger_sha256: Digest


class FreshSelection(ProtocolModel):
    format: Literal["fresh-visible-selection-v1"]
    inference_authorized: Literal[False]
    catalog_sha256: Digest
    tasks: list[TaskPin] = Field(min_length=3, max_length=3)
    cells: list[FreshCell] = Field(min_length=12, max_length=12)


def selected_cell(ordinal: int) -> tuple[FreshCell, Task, bytes, str]:
    """Reobserve selected source/criteria bytes and modes; no allocation or providers."""
    if type(ordinal) is not int or not 1 <= ordinal <= 12:
        raise ValueError("Select one of the twelve fresh visible cells")
    raw, _ = _read_regular(ROOT / "selection.json", 32_000)
    if _digest(raw) != SELECTION_SHA256:
        raise ValueError("Fresh visible selection changed")
    selection = FreshSelection.model_validate_json(raw)
    catalog, _ = _read_regular(ROOT / "tasks.yaml", 32_000)
    if _digest(catalog) != selection.catalog_sha256:
        raise ValueError("Fresh visible catalog changed")
    tasks = load_tasks(ROOT / "tasks.yaml")
    for task, pin in zip(tasks, selection.tasks, strict=True):
        if task.id != pin.task_id or any(
            _tree_digest(path) != expected
            for path, expected in (
                (task.fixture, pin.fixture_sha256),
                (task.acceptance, pin.acceptance_sha256),
                (task.reference, pin.reference_sha256),
                (ROOT / "stronger" / task.id, pin.stronger_sha256),
            )
        ):
            raise ValueError("Fresh visible task/criterion bytes or modes changed")
    cell = selection.cells[ordinal - 1]
    task = next(task for task in tasks if task.id == cell.task_id)
    test = (
        ROOT / "stronger" / task.id / "test_visible.py"
        if cell.arm == "stronger"
        else task.fixture / "test_visible.py"
    )
    content, _ = _read_regular(test, 16_000)
    # Match the existing fixture digest's bounded file+mode encoding, replacing only the test.
    from workspace.changes import snapshot

    modes: dict[str, int] = {}
    files = snapshot(
        task.fixture, ignore=(), max_files=100, max_bytes=1_000_000, strict=True, file_modes=modes
    )
    files["test_visible.py"] = _digest(content)
    if cell.arm == "stronger":
        modes["test_visible.py"] = 0o644
    fixture_sha = _digest(json.dumps({"files": files, "modes": modes}, sort_keys=True).encode())
    return cell, task, content, fixture_sha


class FreshFixture(ProtocolModel):
    format: Literal["prepared-fresh-visible-v1"] = "prepared-fresh-visible-v1"
    selection_sha256: Literal[
        "5208deb02901c0909598525b7458502b64f74b8c1aa6622471f73eb417b3bbd4"
    ] = SELECTION_SHA256
    cell: FreshCell
    fixture_sha256: Digest


class FreshCellWorkspace(ProtocolModel):
    ordinal: int = Field(ge=1, le=12)
    directory: Path


def _write(path: Path, receipt: FreshFixture) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(receipt.model_dump_json(indent=2) + "\n")


def _selection(ordinal: int) -> tuple[FreshFixture, Task, bytes]:
    cell, task, content, fixture_sha = selected_cell(ordinal)
    return FreshFixture(cell=cell, fixture_sha256=fixture_sha), task, content


def prepare_fresh_cell(ordinal: int, directory: Path) -> FreshFixture:
    receipt, task, content = _selection(ordinal)
    destination = _destination(directory)
    destination.mkdir(mode=0o700, exist_ok=False)
    root = destination / "workspace"
    shutil.copytree(task.fixture, root, symlinks=True)
    if receipt.cell.arm == "stronger":
        (root / "test_visible.py").write_bytes(content)
        (root / "test_visible.py").chmod(0o644)
    after, _, _ = _selection(ordinal)
    if after != receipt or _tree_digest(root) != receipt.fixture_sha256:
        raise ValueError("Fresh visible fixture changed during preparation")
    _write(destination / "prepared.json", receipt)
    return receipt


def verify_fresh_cell(directory: Path, *, ordinal: int) -> Path:
    expected, _, _ = _selection(ordinal)
    destination = _destination(directory)
    if (
        destination.is_symlink()
        or not destination.is_dir()
        or stat.S_IMODE(destination.stat().st_mode) != 0o700
        or {p.name for p in islice(destination.iterdir(), 3)} != {"workspace", "prepared.json"}
    ):
        raise ValueError("Invalid fresh visible container/inventory")
    raw, mode = _read_regular(destination / "prepared.json", 4096)
    if mode != 0o600 or FreshFixture.model_validate_json(raw) != expected:
        raise ValueError("Fresh visible receipt differs from selected cell")
    root = destination / "workspace"
    if _tree_digest(root) != expected.fixture_sha256:
        raise ValueError("Fresh visible fixture differs from selected arm")
    return root


@contextmanager
def admit_fresh_cell(directory: Path, *, ordinal: int) -> Iterator[DisposableWorkspace]:
    root = verify_fresh_cell(directory, ordinal=ordinal)
    baseline = _snapshot(root)
    receipt, _, _ = _selection(ordinal)
    _write(root.parent / "admitted.json", receipt)
    after, _, _ = _selection(ordinal)
    if after != receipt or _tree_digest(root) != receipt.fixture_sha256:
        raise ValueError("Fresh visible source changed during admission")
    yield DisposableWorkspace(workspace=Workspace(root=root), baseline=baseline)


class FreshDriftError(PublicPilotDriftError):
    """Latched source or policy drift invalidates shared evaluator acceptance."""


class FreshTaskGuard:
    def __init__(
        self, task: Task, cell: FreshCellWorkspace, *, mode: str, attempt: int, settings: Settings
    ) -> None:
        self.task, self.cell = task, cell
        self.mode, self.attempt, self.settings = mode, attempt, settings
        self.expected_settings = policy_settings_values()
        self.failed = False
        self.failed_checkpoint: str | None = None
        self.checkpoints: list[str] = []
        self.check_at("initial")

    def check_at(self, stage: str) -> None:
        if self.failed:
            raise FreshDriftError("Fresh visible preservation previously failed")
        try:
            receipt, task, _ = _selection(self.cell.ordinal)
            if policy_settings_values() != self.expected_settings:
                raise ValueError("Fresh visible policy source changed")
            if (
                self.task != task
                or self.mode != "harness"
                or type(self.attempt) is not int
                or self.attempt != receipt.cell.repeat
            ):
                raise ValueError("Fresh task/mode/repeat differs from selection")
            actual = self.settings.model_dump()
            if any(actual[key] != value for key, value in self.expected_settings.items()):
                raise ValueError("Fresh visible runtime policy changed")
        except Exception as error:
            self.failed = True
            self.failed_checkpoint = stage
            raise FreshDriftError("Fresh visible source or policy changed") from error
        self.checkpoints.append(stage)

    def __call__(self) -> None:
        self.check_at("before_request")

    def evidence(self) -> dict[str, object]:
        return {
            "format": "fresh-visible-preservation-v1",
            "selection_sha256": SELECTION_SHA256,
            "ordinal": self.cell.ordinal,
            "status": "failed" if self.failed else "checks_passed",
            "checkpoints": list(self.checkpoints),
            "failed_checkpoint": self.failed_checkpoint,
        }
