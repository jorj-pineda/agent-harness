"""Offline fixed development schedule; no provider, workspace admission or approval."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field

from evals.heldout_comparison import PLAN_PATH, ProtocolModel, validate_plan
from evals.heldout_guard import _read_regular

RESIDENCY_PLAN_PATH = Path(__file__).parent / "heldout_comparison/residency-plan.json"


class ResidencyCell(ProtocolModel):
    ordinal: int = Field(ge=1, le=16)
    task_id: str
    mode: Literal["harness"]
    repeat: int = Field(ge=1, le=2)
    keep_alive_seconds: Literal[0, 300]


class ResidencyPlan(ProtocolModel):
    format: Literal["residency-development-plan-v1"]
    evidence_label: Literal["development-on-retired-v1"]
    inference_authorized: Literal[False]
    base_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cells: list[ResidencyCell] = Field(min_length=16, max_length=16)


def validate_residency_plan(path: Path = RESIDENCY_PLAN_PATH) -> ResidencyPlan:
    """Verify all cells and original criteria/budgets before any execution resource."""
    raw, _ = _read_regular(path, 64_000)
    plan = ResidencyPlan.model_validate_json(raw)
    base_raw, _ = _read_regular(PLAN_PATH, 64_000)
    base = validate_plan()
    if hashlib.sha256(base_raw).hexdigest() != plan.base_plan_sha256:
        raise ValueError("Residency base plan differs from selected bytes")
    expected = []
    for repeat, arms in ((1, (0, 300)), (2, (300, 0))):
        for task in base.tasks:
            for seconds in arms:
                expected.append(
                    ResidencyCell(
                        ordinal=len(expected) + 1,
                        task_id=task.id,
                        mode="harness",
                        repeat=repeat,
                        keep_alive_seconds=seconds,
                    )
                )
    if plan.cells != expected:
        raise ValueError("Residency cells differ from fixed task/repeat/arm order")
    return plan
