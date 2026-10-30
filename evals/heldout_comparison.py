"""Offline protocol validation only; no provider construction or launch path."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evals.heldout_guard import REGISTRATIONS, verify_catalog
from evals.real_run import MODES, TaskCategory, _plan_cohort, load_tasks
from harness.config import Settings

PLAN_PATH = Path(__file__).parent / "heldout_comparison/plan.json"


class ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Runtime(ProtocolModel):
    coding_prompt_profile: Literal["compact"]
    coding_toolset: Literal["full"]
    load_root_agents: Literal[False]
    read_file_default_lines: Literal[0]
    max_turn_wall_seconds: Literal[150]
    max_total_tokens_per_turn: Literal[32000]
    max_completion_tokens_per_turn: Literal[4096]
    max_request_output_tokens: Literal[2048]
    min_request_output_tokens: Literal[256]
    max_tool_calls_per_turn: Literal[12]
    max_tool_iterations: Literal[6]
    max_context_tokens: Literal[16384]
    ollama_num_ctx: Literal[16384]
    ollama_think: Literal[False]
    prompt_estimate_bytes_per_token: Literal[3]
    request_timeout_seconds: Literal[180]
    require_verification_before_finish: Literal[True]
    require_plan_before_edit: Literal[False]
    max_completion_retries: Literal[1]
    max_identical_tool_calls: Literal[2]
    max_files_touched_per_turn: Literal[2]
    project_check_argv: list[str]


class TaskPin(ProtocolModel):
    id: str
    category: TaskCategory
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ComparisonPlan(ProtocolModel):
    format: Literal["heldout-comparison-protocol-v1"]
    inference_authorized: Literal[False]
    provider: Literal["ollama"]
    model: Literal["gemma4:12b"]
    model_digest: Literal["4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"]
    expected_ollama_version: Literal["0.33.2"]
    catalog_id: Literal["lower-tier-heldout-v1"]
    seal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repeats: Literal[2]
    modes: list[Literal["harness", "minimal"]]
    tasks: list[TaskPin] = Field(min_length=4, max_length=4)
    runtime: Runtime


def validate_plan(path: Path = PLAN_PATH) -> ComparisonPlan:
    with path.open("rb") as stream:
        raw = stream.read(64_001)
    if len(raw) > 64_000:
        raise ValueError("Comparison plan exceeds byte limit")
    plan = ComparisonPlan.model_validate_json(raw)
    registration = next(r for r in REGISTRATIONS if r.root.name == "v1")
    seal = verify_catalog(registration)
    tasks = load_tasks(registration.root / "tasks.yaml")
    expected = [
        TaskPin(
            id=t.id,
            category=t.category,
            request_sha256=hashlib.sha256(t.request.encode()).hexdigest(),
        )
        for t in tasks
    ]
    if (
        plan.seal_sha256 != registration.seal_sha256
        or plan.catalog_id != seal.catalog_id
        or plan.tasks != expected
        or plan.modes != list(MODES)
    ):
        raise ValueError("Comparison inputs differ from reviewed held-out catalog/order")
    if plan.runtime.project_check_argv != ["python", "-m", "pytest", "-q", "test_visible.py"]:
        raise ValueError("Comparison check differs from fixed visible check")
    # Isolated settings validate cross-field constraints; ambient .env is never loaded.
    Settings(_env_file=None, ollama_model=plan.model, **plan.runtime.model_dump())
    ledger, _ = _plan_cohort(tasks, MODES, plan.repeats)
    if len(ledger.cells) != 16:
        raise ValueError("Comparison must retain all sixteen cells")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    args = parser.parse_args()
    validate_plan(args.plan)
    print("Protocol valid: 16 planned cells; no provider constructed or inference authorized.")


if __name__ == "__main__":
    main()
