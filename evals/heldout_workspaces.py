"""Prepare named evaluation-owned fixture copies, without executing model work."""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evals.heldout_guard import (
    REGISTRATIONS,
    HeldoutDriftError,
    HeldoutRegistration,
    _read_regular,
    verify_catalog,
    verify_fixture,
)
from evals.real_run import Task, load_tasks
from workspace import DisposableWorkspace, Workspace
from workspace.disposable import _snapshot


class PreparedFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["prepared-heldout-fixture-v1"] = "prepared-heldout-fixture-v1"
    task_id: str = Field(min_length=1, max_length=64)
    seal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _registered_task(task_id: str) -> tuple[HeldoutRegistration, Task]:
    registration = REGISTRATIONS[0]
    verify_catalog(registration)
    tasks = load_tasks(registration.root / "tasks.yaml")
    task = next((task for task in tasks if task.id == task_id), None)
    if task is None:
        raise ValueError("Unknown registered held-out task")
    return registration, task


def _owned_location(directory: Path) -> Path:
    # Refuse Git workspaces, including this checkout; no user repo is an eval fixture.
    parent = directory.parent.resolve(strict=True)
    destination = parent / directory.name
    for ancestor in (parent, *parent.parents):
        if (ancestor / ".git").exists() or (ancestor / ".git").is_symlink():
            raise ValueError("Prepared evaluation directories must be outside Git workspaces")
    for registration in REGISTRATIONS:
        root = registration.root.resolve()
        if destination.is_relative_to(root) or root.is_relative_to(destination):
            raise ValueError("Prepared directory overlaps registered evaluation inputs")
    return destination


def prepare_fixture(task_id: str, directory: Path) -> PreparedFixture:
    registration, task = _registered_task(task_id)
    destination = _owned_location(directory)
    destination.mkdir(mode=0o700, exist_ok=False)
    # Refusal/failure leaves the owned directory intact for inspection, never resets it.
    shutil.copytree(task.fixture, destination / "workspace", symlinks=True)
    verify_fixture(task.fixture, destination / "workspace")
    receipt = PreparedFixture(task_id=task.id, seal_sha256=registration.seal_sha256)
    descriptor = os.open(destination / "prepared.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(receipt.model_dump_json(indent=2) + "\n")
    return receipt


def verify_prepared(directory: Path, *, task_id: str) -> Path:
    destination = _owned_location(directory)
    if destination.is_symlink() or not destination.is_dir():
        raise HeldoutDriftError("Invalid prepared evaluation directory")
    if {path.name for path in islice(destination.iterdir(), 3)} != {"prepared.json", "workspace"}:
        raise HeldoutDriftError("Prepared evaluation inventory changed")
    raw, mode = _read_regular(destination / "prepared.json", 4096)
    receipt = PreparedFixture.model_validate_json(raw)
    registration, task = _registered_task(task_id)
    if (
        receipt.task_id != task.id
        or receipt.seal_sha256 != registration.seal_sha256
        or mode != 0o600
    ):
        raise HeldoutDriftError("Prepared receipt does not match reviewed task identity")
    verify_fixture(task.fixture, destination / "workspace")
    return destination / "workspace"


@contextmanager
def admit_prepared(directory: Path, *, task_id: str) -> Iterator[DisposableWorkspace]:
    """Single-use evaluation admission; retain all artifacts on every exit path."""
    root = verify_prepared(directory, task_id=task_id)
    baseline = _snapshot(root)
    marker = root.parent / "admitted.json"
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(
            PreparedFixture(
                task_id=task_id, seal_sha256=REGISTRATIONS[0].seal_sha256
            ).model_dump_json()
            + "\n"
        )
    _, task = _registered_task(task_id)
    verify_fixture(task.fixture, root)
    yield DisposableWorkspace(workspace=Workspace(root=root), baseline=baseline)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_prepared(args.directory, task_id=args.task)
    else:
        prepare_fixture(args.task, args.directory)
    print("Prepared fixture verified; no model execution or resume authorization.")


if __name__ == "__main__":
    main()
