"""Named, single-use public-pilot fixture copies; no launch approval or isolation."""

from __future__ import annotations

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
from evals.heldout_workspaces import _owned_location
from evals.public_envelope import PLAN_SHA256, _tree_digest, validate_public_plan
from evals.real_run import TASKS_ROOT, Task, load_tasks
from workspace import DisposableWorkspace, Workspace
from workspace.disposable import _snapshot


class PublicFixture(ProtocolModel):
    format: Literal["prepared-public-pilot-v1"] = "prepared-public-pilot-v1"
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordinal: int = Field(ge=1, le=6)
    task_id: str
    repeat: int = Field(ge=1, le=2)


def _selection(ordinal: int) -> tuple[PublicFixture, Task, str]:
    if type(ordinal) is not int or not 1 <= ordinal <= 6:
        raise ValueError("Select one of the six public pilot cells")
    plan = validate_public_plan()
    cell = plan.cells[ordinal - 1]
    task = next(task for task in load_tasks(TASKS_ROOT / "tasks.yaml") if task.id == cell.task_id)
    pin = next(pin for pin in plan.tasks if pin.id == task.id)
    return (
        PublicFixture(
            plan_sha256=PLAN_SHA256, ordinal=ordinal, task_id=cell.task_id, repeat=cell.repeat
        ),
        task,
        pin.fixture_sha256,
    )


def _destination(directory: Path) -> Path:
    destination = _owned_location(directory)
    source = TASKS_ROOT.resolve(strict=True)
    if (
        directory != destination
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Public pilot copies require canonical paths outside source inputs")
    return destination


def _write(path: Path, receipt: PublicFixture) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(receipt.model_dump_json(indent=2) + "\n")


def prepare_public_cell(ordinal: int, directory: Path) -> PublicFixture:
    receipt, task, digest = _selection(ordinal)
    destination = _destination(directory)
    destination.mkdir(mode=0o700, exist_ok=False)
    # Preserve partial output on all failures; never reset an occupied destination.
    shutil.copytree(task.fixture, destination / "workspace", symlinks=True)
    if _tree_digest(destination / "workspace") != digest:
        raise ValueError("Public fixture copy differs from selected task")
    after, _, _ = _selection(ordinal)
    if after != receipt:
        raise ValueError("Public pilot selection changed during preparation")
    _write(destination / "prepared.json", receipt)
    return receipt


def verify_public_cell(directory: Path, *, ordinal: int) -> Path:
    expected, _, digest = _selection(ordinal)
    destination = _destination(directory)
    if (
        destination.is_symlink()
        or not destination.is_dir()
        or stat.S_IMODE(destination.stat().st_mode) != 0o700
    ):
        raise ValueError("Invalid public pilot directory")
    if {path.name for path in islice(destination.iterdir(), 3)} != {"workspace", "prepared.json"}:
        raise ValueError("Public pilot container inventory changed")
    raw, mode = _read_regular(destination / "prepared.json", 4096)
    if mode != 0o600 or PublicFixture.model_validate_json(raw) != expected:
        raise ValueError("Public pilot receipt differs from selected cell")
    if _tree_digest(destination / "workspace") != digest:
        raise ValueError("Public pilot fixture changed before admission")
    return destination / "workspace"


@contextmanager
def admit_public_cell(directory: Path, *, ordinal: int) -> Iterator[DisposableWorkspace]:
    """Retain editable copies on exit; caller still owns execution authority and guards."""
    root = verify_public_cell(directory, ordinal=ordinal)
    baseline = _snapshot(root)
    receipt, _, digest = _selection(ordinal)
    _write(root.parent / "admitted.json", receipt)
    # A failed post-marker check consumes admission and preserves evidence.
    _selection(ordinal)
    if _tree_digest(root) != digest:
        raise ValueError("Public pilot fixture changed during admission")
    yield DisposableWorkspace(workspace=Workspace(root=root), baseline=baseline)
