"""Fresh single-use comparison fixtures and shared-evaluator guards; no launch approval."""

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
from evals.heldout_guard import _read_regular
from evals.public_admission import PublicPilotDriftError
from evals.public_envelope import _digest, _tree_digest, policy_settings_values
from evals.public_feedback import (
    PLAN_SHA256,
    FeedbackCell,
    validate_feedback_plan,
    visible_check_bytes,
)
from evals.public_workspaces import _destination
from evals.real_run import TASKS_ROOT, Task, load_tasks
from harness.config import Settings
from workspace import DisposableWorkspace, Workspace
from workspace.changes import snapshot
from workspace.disposable import _snapshot


class FeedbackFixture(ProtocolModel):
    format: Literal["prepared-public-feedback-v1"] = "prepared-public-feedback-v1"
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cell: FeedbackCell
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FeedbackCellWorkspace(ProtocolModel):
    ordinal: int = Field(ge=1, le=8)
    directory: Path


def _selection(ordinal: int) -> tuple[FeedbackFixture, Task]:
    if type(ordinal) is not int or not 1 <= ordinal <= 8:
        raise ValueError("Select one of the eight public feedback cells")
    plan = validate_feedback_plan()
    cell = plan.cells[ordinal - 1]
    task = next(t for t in load_tasks(TASKS_ROOT / "tasks.yaml") if t.id == cell.task_id)
    modes: dict[str, int] = {}
    files = snapshot(
        task.fixture, ignore=(), max_files=100, max_bytes=1_000_000, strict=True, file_modes=modes
    )
    files["test_visible.py"] = _digest(visible_check_bytes(cell.arm))
    if cell.arm == "bool_check":
        modes["test_visible.py"] = 0o644
    digest = _digest(json.dumps({"files": files, "modes": modes}, sort_keys=True).encode())
    return FeedbackFixture(plan_sha256=PLAN_SHA256, cell=cell, fixture_sha256=digest), task


def _write(path: Path, receipt: FeedbackFixture) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(receipt.model_dump_json(indent=2) + "\n")


def prepare_feedback_cell(ordinal: int, directory: Path) -> FeedbackFixture:
    receipt, task = _selection(ordinal)
    destination = _destination(directory)
    destination.mkdir(mode=0o700, exist_ok=False)
    root = destination / "workspace"
    shutil.copytree(task.fixture, root, symlinks=True)
    if receipt.cell.arm == "bool_check":
        (root / "test_visible.py").write_bytes(visible_check_bytes(receipt.cell.arm))
        (root / "test_visible.py").chmod(0o644)
    after, _ = _selection(ordinal)
    if after != receipt or _tree_digest(root) != receipt.fixture_sha256:
        raise ValueError("Feedback fixture or source changed during preparation")
    _write(destination / "prepared.json", receipt)
    return receipt


def verify_feedback_cell(directory: Path, *, ordinal: int) -> Path:
    expected, _ = _selection(ordinal)
    destination = _destination(directory)
    if (
        destination.is_symlink()
        or not destination.is_dir()
        or stat.S_IMODE(destination.stat().st_mode) != 0o700
        or {p.name for p in islice(destination.iterdir(), 3)} != {"workspace", "prepared.json"}
    ):
        raise ValueError("Invalid feedback container or inventory")
    raw, mode = _read_regular(destination / "prepared.json", 4096)
    if mode != 0o600 or FeedbackFixture.model_validate_json(raw) != expected:
        raise ValueError("Feedback receipt differs from selected cell")
    root = destination / "workspace"
    if _tree_digest(root) != expected.fixture_sha256:
        raise ValueError("Feedback fixture differs from selected arm")
    return root


@contextmanager
def admit_feedback_cell(directory: Path, *, ordinal: int) -> Iterator[DisposableWorkspace]:
    root = verify_feedback_cell(directory, ordinal=ordinal)
    baseline = _snapshot(root)
    receipt, _ = _selection(ordinal)
    _write(root.parent / "admitted.json", receipt)
    after, _ = _selection(ordinal)
    if after != receipt or _tree_digest(root) != receipt.fixture_sha256:
        raise ValueError("Feedback fixture or source changed during admission")
    yield DisposableWorkspace(workspace=Workspace(root=root), baseline=baseline)


class FeedbackDriftError(PublicPilotDriftError):
    """Failed preservation latches and prevents accepted status in the shared grader."""


class FeedbackTaskGuard:
    def __init__(
        self,
        task: Task,
        cell: FeedbackCellWorkspace,
        *,
        mode: str,
        attempt: int,
        settings: Settings,
    ) -> None:
        self.task = task
        self.cell = cell
        self.mode = mode
        self.attempt = attempt
        self.settings = settings
        self.expected_settings = policy_settings_values()
        self.failed = False
        self.failed_checkpoint: str | None = None
        self.checkpoints: list[str] = []
        self.check_at("initial")

    def check_at(self, stage: str) -> None:
        if self.failed:
            raise FeedbackDriftError("Public feedback preservation previously failed")
        try:
            receipt, task = _selection(self.cell.ordinal)
            if (
                self.task != task
                or self.mode != "harness"
                or type(self.attempt) is not int
                or self.attempt != receipt.cell.repeat
            ):
                raise ValueError("Feedback task, mode or repeat differs from selection")
            actual = self.settings.model_dump()
            if any(actual[key] != value for key, value in self.expected_settings.items()):
                raise ValueError("Feedback runtime policy differs from selected comparison")
        except Exception as error:
            self.failed = True
            self.failed_checkpoint = stage
            raise FeedbackDriftError("Public feedback selection or source changed") from error
        self.checkpoints.append(stage)

    def __call__(self) -> None:
        self.check_at("before_request")

    def evidence(self) -> dict[str, object]:
        return {
            "format": "public-feedback-preservation-v1",
            "plan_sha256": PLAN_SHA256,
            "ordinal": self.cell.ordinal,
            "status": "failed" if self.failed else "checks_passed",
            "checkpoints": list(self.checkpoints),
            "failed_checkpoint": self.failed_checkpoint,
        }
