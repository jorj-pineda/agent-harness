"""Offline validation of the public visible-check comparison; no launch authority."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from evals.heldout_comparison import ProtocolModel
from evals.heldout_guard import _read_regular
from evals.public_envelope import PublicTaskPin, _digest, validate_public_plan
from evals.real_run import TASKS_ROOT, load_tasks

ROOT = Path(__file__).parent / "public_feedback"
PLAN_PATH = ROOT / "plan.json"
PLAN_SHA256 = "9ba6c0b94c1ec9865d7fda15626d5361426ce368adfe209857b0c74b21fef852"
FIXTURE_PATH = ROOT / "fixture.json"
FIXTURE_SHA256 = "8ad0511f2e5c97e106e8e8af17401897ddf839342ffd9e78a2f045294d966f92"
TREATMENT_PATH = ROOT / "fixtures/test_visible.py"
Arm = Literal["control", "bool_check"]


class FeedbackCell(ProtocolModel):
    ordinal: int = Field(ge=1, le=8)
    task_id: Literal["paginate_items"]
    mode: Literal["harness"]
    arm: Arm
    repeat: int = Field(ge=1, le=4)


class Treatment(ProtocolModel):
    changed_path: Literal["test_visible.py"]
    retain_original_assertions: Literal[True]
    additional_cases: Literal[8]
    inputs: list[bool]
    arguments: list[Literal["page", "page_size"]]
    empty_inputs: list[bool]
    expected_exception: Literal["ValueError"]
    changes_to_implementation: Literal[False]
    changes_to_request: Literal[False]
    changes_to_acceptance: Literal[False]


class FeedbackPlan(ProtocolModel):
    format: Literal["public-visible-check-comparison-v1"]
    evidence_label: Literal["public-development-visible-check-comparison"]
    inference_authorized: Literal[False]
    provider: Literal["ollama"]
    model: Literal["gemma4:12b"]
    model_digest: Literal["4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"]
    expected_ollama_version: Literal["0.33.2"]
    source_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task: PublicTaskPin
    coding_prompt_profile: Literal["direct"]
    coding_toolset: Literal["whole_file_check"]
    keep_alive_seconds: Literal[300]
    arms: list[Arm] = Field(min_length=2, max_length=2)
    repeats: Literal[4]
    cells: list[FeedbackCell] = Field(min_length=8, max_length=8)
    treatment: Treatment


class FixturePin(ProtocolModel):
    format: Literal["public-visible-check-fixture-v1"]
    plan_sha256: Literal["9ba6c0b94c1ec9865d7fda15626d5361426ce368adfe209857b0c74b21fef852"]
    path: Literal["test_visible.py"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: Literal[420]


def _treatment_bytes() -> bytes:
    raw, _ = _read_regular(FIXTURE_PATH, 4096)
    if _digest(raw) != FIXTURE_SHA256:
        raise ValueError("Feedback fixture selection differs from reviewed bytes")
    pin = FixturePin.model_validate_json(raw)
    content, mode = _read_regular(TREATMENT_PATH, 16_000)
    if _digest(content) != pin.sha256 or mode != pin.mode:
        raise ValueError("Feedback visible test differs from selected bytes/mode")
    return content


def validate_feedback_plan(path: Path = PLAN_PATH) -> FeedbackPlan:
    raw, _ = _read_regular(path, 64_000)
    if _digest(raw) != PLAN_SHA256:
        raise ValueError("Feedback plan differs from reviewed bytes")
    plan = FeedbackPlan.model_validate_json(raw)
    original = validate_public_plan()
    from evals.public_envelope import PLAN_SHA256 as SOURCE_PLAN_SHA256

    if (
        plan.source_plan_sha256 != SOURCE_PLAN_SHA256
        or plan.catalog_sha256 != original.catalog_sha256
        or plan.budget_source_sha256 != original.budget_source_sha256
        or plan.task != next(task for task in original.tasks if task.id == "paginate_items")
    ):
        raise ValueError("Feedback original task or budget selection changed")
    _treatment_bytes()
    return plan


def visible_check_bytes(arm: Arm) -> bytes:
    """Return only selected test bytes; do not prepare fixtures or grant execution."""
    if arm not in ("control", "bool_check"):
        raise ValueError("Select control or bool_check")
    validate_feedback_plan()
    if arm == "bool_check":
        content = _treatment_bytes()
    else:
        task = next(
            task for task in load_tasks(TASKS_ROOT / "tasks.yaml") if task.id == "paginate_items"
        )
        content, _ = _read_regular(task.fixture / "test_visible.py", 16_000)
    validate_feedback_plan()
    return content
