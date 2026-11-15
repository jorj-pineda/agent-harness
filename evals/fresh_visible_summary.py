"""Descriptive arm/pair counts from owned report rows; no quality or regrade claim."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evals.fresh_visible_runner import FreshLedger
from evals.fresh_visible_workspaces import SELECTION_SHA256

if TYPE_CHECKING:
    from evals.real_run import RealTaskResult


def fresh_summary(
    ledger: FreshLedger, results: list[RealTaskResult], *, scripted: bool
) -> dict[str, Any]:
    """Summarize an executor snapshot without reading inputs or running any checks.

    A started cell may have a durable result before cleanup. Keep it separate from
    recorded cells. Missing results and unknown usage never become zero totals.
    These recorded acceptance outcomes still require independent regrades/review.
    """
    ledger = FreshLedger.model_validate(ledger.model_dump())
    cells = ledger.cells
    # Fixed v1 schedule in memory: stopped reports must not reread source inputs.
    task_ids = ("fv_lookup_option", "fv_merge_windows", "fv_header_pair")
    expected = [
        (task_ids[t], arm, repeat)
        for repeat, t, arms in (
            (1, 0, ("weak", "stronger")),
            (1, 1, ("stronger", "weak")),
            (1, 2, ("weak", "stronger")),
            (2, 0, ("stronger", "weak")),
            (2, 1, ("weak", "stronger")),
            (2, 2, ("stronger", "weak")),
        )
        for arm in arms
    ]
    if [
        (state.cell.ordinal, state.cell.task_id, state.cell.arm, state.cell.repeat)
        for state in cells
    ] != [(i, *identity) for i, identity in enumerate(expected, 1)]:
        raise ValueError("Fresh report requires the fixed twelve-cell v1 schedule")
    statuses = [state.status for state in cells]
    recorded = statuses.count("recorded")
    started = statuses.count("started")
    if (
        started > 1
        or statuses
        != ["recorded"] * recorded + ["started"] * started + ["unrun"] * (12 - recorded - started)
        or not recorded <= len(results) <= recorded + started
        or (ledger.status == "finished" and recorded != 12)
        or (ledger.status == "planned" and (recorded or started))
    ):
        raise ValueError("Fresh result prefix differs from ledger progress")
    for state, result in zip(cells, results, strict=False):
        if (result.task_id, result.mode, result.attempt) != (
            state.cell.task_id,
            "harness",
            state.cell.repeat,
        ):
            raise ValueError("Fresh result identity differs from its ordinal/repeat")
        preservation = result.runtime_config.get("fresh_visible_preservation", {})
        if (
            preservation.get("format") != "fresh-visible-preservation-v1"
            or preservation.get("selection_sha256") != SELECTION_SHA256
            or type(preservation.get("ordinal")) is not int
            or preservation.get("ordinal") != state.cell.ordinal
        ):
            raise ValueError("Fresh result preservation differs from its cell ordinal")

    rows = []
    for index, state in enumerate(cells):
        result = results[index] if index < len(results) else None
        rows.append(
            {
                **state.cell.model_dump(),
                "ledger_status": state.status,
                "result_present": result is not None,
                "accepted": result.passed if result is not None else None,
                "completed_and_accepted": (
                    result.passed and result.completion_status == "completed"
                    if result is not None
                    else None
                ),
                "completion_status": result.completion_status if result is not None else None,
                "verification_status": result.verification_status if result is not None else None,
                "termination": result.termination if result is not None else None,
                "prompt_tokens": result.prompt_tokens if result is not None else None,
                "completion_tokens": result.completion_tokens if result is not None else None,
                "recorded_latency_ms": result.latency_ms if result is not None else None,
                "visible_check_exit_codes": (
                    [
                        (tool.get("result") or {}).get("exit_code")
                        for tool in result.tool_trace
                        if tool.get("name") == "run_project_check"
                    ]
                    if result is not None
                    else []
                ),
            }
        )

    def counts(selected_rows):
        arms = {}
        for arm in ("weak", "stronger"):
            selected = [row for row in selected_rows if row["arm"] == arm]
            observed = [row for row in selected if row["result_present"]]
            usage_known = len(observed) == len(selected) and all(
                row["prompt_tokens"] is not None and row["completion_tokens"] is not None
                for row in observed
            )
            arms[arm] = {
                "planned": len(selected),
                "results_present": len(observed),
                "recorded_cells": sum(row["ledger_status"] == "recorded" for row in selected),
                "missing_results": len(selected) - len(observed),
                "accepted": sum(row["accepted"] for row in observed),
                "completed_and_accepted": sum(row["completed_and_accepted"] for row in observed),
                "unknown_usage_results": sum(
                    row["prompt_tokens"] is None or row["completion_tokens"] is None
                    for row in observed
                ),
                "prompt_tokens": sum(row["prompt_tokens"] for row in observed)
                if usage_known
                else None,
                "completion_tokens": (
                    sum(row["completion_tokens"] for row in observed) if usage_known else None
                ),
                "known_result_prompt_tokens": sum(row["prompt_tokens"] or 0 for row in observed),
                "known_result_completion_tokens": sum(
                    row["completion_tokens"] or 0 for row in observed
                ),
            }
        return arms

    return {
        "format": "fresh-visible-descriptive-summary-v1",
        "evidence_label": "scripted-real-tools-rehearsal"
        if scripted
        else "real-tools-trusted-fixtures",
        "cohort_status": ledger.status,
        "stop_reason": ledger.stop_reason,
        "arms": counts(rows),
        "tasks": {
            task: counts([row for row in rows if row["task_id"] == task]) for task in task_ids
        },
        "pairs": [
            {
                "task_id": task_id,
                "repeat": repeat,
                **{
                    arm: next(
                        row
                        for row in rows
                        if row["task_id"] == task_id
                        and row["repeat"] == repeat
                        and row["arm"] == arm
                    )
                    for arm in ("weak", "stronger")
                },
            }
            for task_id in task_ids
            for repeat in (1, 2)
        ],
        "benefit_assessment": "not_assessed",
        "independent_regrades": "not_supplied",
        "manual_rescues": None,
        "new_regressions": None,
        "limits": [
            "Recorded artifact acceptance is separate from visible checks and completion.",
            "Started results do not certify settlement; missing cells are not failed attempts.",
            "Unknown or incomplete usage totals remain null; known result totals are partial.",
            "Row latency excludes parts of setup/storage/cleanup and is not end-to-end timing.",
            "Benefit claims require independent regrades and intervention/regression review.",
            "Scripted counts do not measure model quality; no default or release decision follows.",
        ],
    }
