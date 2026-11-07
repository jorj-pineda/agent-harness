"""Offline public-development pilot validation; no resources or inference authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from evals.heldout_comparison import PLAN_PATH as BUDGET_SOURCE
from evals.heldout_comparison import ProtocolModel, Runtime
from evals.heldout_guard import _read_regular
from evals.real_run import TASKS_ROOT, TaskCategory, load_tasks
from workspace.changes import snapshot

PLAN_PATH = Path(__file__).parent / "public_envelope/plan.json"
PLAN_SHA256 = "9e12feed16ea9a6ab63bf7802f99be89295daf23f58b529c5460a04a4a18939d"
TASK_IDS = ("divide_zero", "paginate_items", "extract_label_normalizer")


class PublicTaskPin(ProtocolModel):
    id: str
    category: TaskCategory
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    acceptance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PublicCell(ProtocolModel):
    ordinal: int = Field(ge=1, le=6)
    task_id: str
    mode: Literal["harness"]
    repeat: int = Field(ge=1, le=2)


class PublicEnvelopePlan(ProtocolModel):
    format: Literal["public-task-envelope-v1"]
    evidence_label: Literal["public-development-task-envelope"]
    inference_authorized: Literal[False]
    model: Literal["gemma4:12b"]
    model_digest: Literal["4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"]
    expected_ollama_version: Literal["0.33.2"]
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    coding_prompt_profile: Literal["direct"]
    coding_toolset: Literal["whole_file_check"]
    keep_alive_seconds: Literal[300]
    tasks: list[PublicTaskPin] = Field(min_length=3, max_length=3)
    cells: list[PublicCell] = Field(min_length=6, max_length=6)


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tree_digest(directory: Path) -> str:
    if directory.is_symlink():
        raise ValueError("Pilot tree root cannot be a symlink")
    modes: dict[str, int] = {}
    files = snapshot(
        directory, ignore=(), max_files=100, max_bytes=1_000_000, strict=True, file_modes=modes
    )
    if not files or any(value.startswith("symlink:") for value in files.values()):
        raise ValueError("Pilot trees must contain bounded regular files")
    return _digest(json.dumps({"files": files, "modes": modes}, sort_keys=True).encode())


def task_pins() -> list[PublicTaskPin]:
    tasks = {task.id: task for task in load_tasks(TASKS_ROOT / "tasks.yaml")}
    return [
        PublicTaskPin(
            id=task.id,
            category=task.category,
            request_sha256=_digest(task.request.encode()),
            fixture_sha256=_tree_digest(task.fixture),
            acceptance_sha256=_tree_digest(task.acceptance),
            reference_sha256=_tree_digest(task.reference),
        )
        for task in (tasks[task_id] for task_id in TASK_IDS)
    ]


def expected_cells() -> list[PublicCell]:
    return [
        PublicCell(
            ordinal=repeat_index * 3 + task_index + 1,
            task_id=task_id,
            mode="harness",
            repeat=repeat_index + 1,
        )
        for repeat_index in range(2)
        for task_index, task_id in enumerate(TASK_IDS)
    ]


def validate_public_plan(path: Path = PLAN_PATH) -> PublicEnvelopePlan:
    raw, _ = _read_regular(path, 64_000)
    if _digest(raw) != PLAN_SHA256:
        raise ValueError("Pilot plan differs from reviewed bytes")
    plan = PublicEnvelopePlan.model_validate_json(raw)
    catalog, _ = _read_regular(TASKS_ROOT / "tasks.yaml", 1_000_000)
    budget, _ = _read_regular(BUDGET_SOURCE, 64_000)
    # Existing budget provenance only; never inherit the old held-out task schedule.
    Runtime.model_validate(json.loads(budget)["runtime"])
    if plan.catalog_sha256 != _digest(catalog) or plan.budget_source_sha256 != _digest(budget):
        raise ValueError("Pilot catalog or original budget bytes changed")
    if plan.tasks != task_pins() or plan.cells != expected_cells():
        raise ValueError("Pilot task pins or fixed six-cell order changed")
    return plan
