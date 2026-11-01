"""Preserve verified no-forward captures before fresh exact-path fixture staging."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import islice
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from evals.heldout_command_environment import load_environment_pin
from evals.heldout_comparison import PLAN_PATH, validate_plan
from evals.heldout_guard import REGISTRATIONS, _read_regular, verify_fixture
from evals.heldout_inputs import _plan_digest, _save
from evals.heldout_settings import load_settings_pin
from evals.heldout_workspaces import (
    PreparedFixture,
    _owned_location,
    prepare_fixture,
    verify_prepared,
)
from evals.real_run import MODES, Mode, _plan_cohort, load_tasks


class CaptureCell(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    ordinal: int
    task_id: str
    mode: Mode
    attempt: int
    files: dict[str, str]
    runtime_config: dict[str, Any]
    repeat_bytes_equal: bool | None


class RuntimePin(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty: Literal[False]


class CaptureIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    format: Literal["heldout-first-inputs-v1"]
    evidence_kind: Literal["mock-production-adapter"]
    inference_ran: Literal[False]
    forwarded_requests: Literal[0]
    runtime: RuntimePin
    plan_sha256: str
    seal_sha256: str
    settings_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    command_environment_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    runtime_inventory_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    cells: list[CaptureCell] = Field(min_length=16, max_length=16)
    planned_cells: Literal[16]


def verify_capture(
    directory: Path, *, expected_index_sha256: str, original_directory: Path | None = None
) -> CaptureIndex:
    source = _owned_location(directory)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Invalid capture directory")
    raw, mode = _read_regular(source / "index.json", 2_000_000)
    if mode != 0o600 or hashlib.sha256(raw).hexdigest() != expected_index_sha256:
        raise ValueError("Capture index differs from reviewed hash")
    index = CaptureIndex.model_validate_json(raw)
    plan = validate_plan()
    if index.plan_sha256 != _plan_digest(PLAN_PATH) or index.seal_sha256 != plan.seal_sha256:
        raise ValueError("Capture plan/seal changed")
    tasks = load_tasks(REGISTRATIONS[0].root / "tasks.yaml")
    _, schedule = _plan_cohort(tasks, MODES, plan.repeats)
    inventory = {"index.json"}
    if index.settings_sha256 is not None:
        load_settings_pin(source / "settings.json", expected_sha256=index.settings_sha256)
        inventory.add("settings.json")
    if index.command_environment_sha256 is not None:
        load_environment_pin(source / "command-environment.json", index.command_environment_sha256)
        inventory.add("command-environment.json")
    if index.runtime_inventory_sha256 is not None:
        from evals.heldout_runtime_guard import RuntimeInventory

        raw_runtime, runtime_mode = _read_regular(source / "runtime-inventory.json", 2_000_000)
        if (
            runtime_mode != 0o600
            or hashlib.sha256(raw_runtime).hexdigest() != index.runtime_inventory_sha256
        ):
            raise ValueError("Archived runtime inventory bytes/mode changed")
        RuntimeInventory.model_validate_json(raw_runtime)
        inventory.add("runtime-inventory.json")
    for ordinal, (cell, (task, mode_name, attempt)) in enumerate(
        zip(index.cells, schedule, strict=True), 1
    ):
        if (cell.ordinal, cell.task_id, cell.mode, cell.attempt) != (
            ordinal,
            task.id,
            mode_name,
            attempt,
        ):
            raise ValueError("Capture cell order/identity changed")
        names = {f"{ordinal:02d}-normalized.json", f"{ordinal:02d}-wire.json"}
        if set(cell.files) != names:
            raise ValueError("Capture file map changed")
        inventory.update(names)
        bodies = {}
        for name in names:
            content, file_mode = _read_regular(source / name, 2_000_000)
            if file_mode != 0o600 or hashlib.sha256(content).hexdigest() != cell.files[name]:
                raise ValueError("Captured input bytes/mode changed")
            bodies[name] = json.loads(content)
        container = source / f"cell-{ordinal:02d}"
        inventory.add(container.name)
        if container.is_symlink() or not container.is_dir():
            raise ValueError("Invalid captured workspace container")
        if {p.name for p in islice(container.iterdir(), 4)} != {
            "prepared.json",
            "admitted.json",
            "workspace",
        }:
            raise ValueError("Captured container inventory changed")
        expected = PreparedFixture(task_id=task.id, seal_sha256=plan.seal_sha256)
        for receipt_name in ("prepared.json", "admitted.json"):
            receipt, receipt_mode = _read_regular(container / receipt_name, 4096)
            if receipt_mode != 0o600 or PreparedFixture.model_validate_json(receipt) != expected:
                raise ValueError("Captured receipt identity changed")
        root = container / "workspace"
        expected_root = (original_directory or source) / container.name / "workspace"
        marker = f"Workspace root: {expected_root}"
        for body in bodies.values():
            if marker not in body["messages"][0]["content"].splitlines():
                raise ValueError("Captured system message has a different workspace root")
        if cell.runtime_config.get("prepared_workspace_root") != str(expected_root):
            raise ValueError("Capture root differs from original named path")
        verify_fixture(task.fixture, root)
    if {p.name for p in islice(source.iterdir(), len(inventory) + 1)} != inventory:
        raise ValueError("Capture bundle inventory changed")
    return index


def stage_capture(directory: Path, archive: Path, *, expected_index_sha256: str) -> None:
    source = _owned_location(directory)
    target = _owned_location(archive)
    if source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Capture and archive paths overlap")
    index = verify_capture(source, expected_index_sha256=expected_index_sha256)
    target.mkdir(mode=0o700, exist_ok=False)
    # The exclusive archive container reserves an absent child destination.
    source.rename(target / "captured")
    verify_capture(
        target / "captured", expected_index_sha256=expected_index_sha256, original_directory=source
    )
    source.mkdir(mode=0o700, exist_ok=False)
    for cell in index.cells:
        container = source / f"cell-{cell.ordinal:02d}"
        prepare_fixture(cell.task_id, container)
        verify_prepared(container, task_id=cell.task_id)
    _save(
        source / "staged.json",
        json.dumps(
            {
                "format": "staged-heldout-capture-v1",
                "capture_index_sha256": expected_index_sha256,
                "archive": str(target / "captured"),
                "inference_authorized": False,
            },
            indent=2,
        ).encode(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    args = parser.parse_args()
    stage_capture(args.directory, args.archive, expected_index_sha256=args.index_sha256)
    print("No-forward evidence archived and fresh fixtures staged; no inference authorized.")


if __name__ == "__main__":
    main()
