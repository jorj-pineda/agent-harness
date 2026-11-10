"""Descriptive arm/pair counts from owned report rows; no quality or regrade claim."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evals.public_feedback_runner import FeedbackLedger

if TYPE_CHECKING:
    from evals.real_run import RealTaskResult


def feedback_summary(
    ledger: FeedbackLedger, results: list[RealTaskResult], *, scripted: bool
) -> dict[str, Any]:
    """Summarize an executor snapshot without reading inputs or running any checks.

    A started cell may have a durable result before cleanup. Keep it separate from
    recorded cells. Missing results and unknown usage never become zero totals.
    These recorded acceptance outcomes still require independent regrades/review.
    """
    ledger = FeedbackLedger.model_validate(ledger.model_dump())
    cells = ledger.cells
    if [state.cell.ordinal for state in cells] != list(range(1, 9)):
        raise ValueError("Feedback report requires eight ordered distinct ordinals")
    for repeat in range(1, 5):
        pair = cells[(repeat - 1) * 2 : repeat * 2]
        if any(state.cell.repeat != repeat for state in pair) or {
            state.cell.arm for state in pair
        } != {"control", "bool_check"}:
            raise ValueError("Feedback report requires both arms in every paired repeat")
    statuses = [state.status for state in cells]
    recorded = statuses.count("recorded")
    started = statuses.count("started")
    if (
        started > 1
        or statuses
        != ["recorded"] * recorded + ["started"] * started + ["unrun"] * (8 - recorded - started)
        or not recorded <= len(results) <= recorded + started
        or (ledger.status == "finished" and recorded != 8)
        or (ledger.status == "planned" and (recorded or started))
    ):
        raise ValueError("Feedback result prefix differs from ledger progress")
    for state, result in zip(cells, results, strict=False):
        if (result.task_id, result.mode, result.attempt) != (
            state.cell.task_id,
            state.cell.mode,
            state.cell.repeat,
        ):
            raise ValueError("Feedback result identity differs from its ordinal/repeat")

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
    arms = {}
    for arm in ("control", "bool_check"):
        selected = [row for row in rows if row["arm"] == arm]
        observed = [row for row in selected if row["result_present"]]
        usage_known = len(observed) == 4 and all(
            row["prompt_tokens"] is not None and row["completion_tokens"] is not None
            for row in observed
        )
        arms[arm] = {
            "planned": 4,
            "results_present": len(observed),
            "recorded_cells": sum(row["ledger_status"] == "recorded" for row in selected),
            "missing_results": 4 - len(observed),
            "accepted": sum(row["accepted"] for row in observed),
            "completed_and_accepted": sum(row["completed_and_accepted"] for row in observed),
            "unknown_usage_results": sum(
                row["prompt_tokens"] is None or row["completion_tokens"] is None for row in observed
            ),
            "prompt_tokens": sum(row["prompt_tokens"] for row in observed) if usage_known else None,
            "completion_tokens": (
                sum(row["completion_tokens"] for row in observed) if usage_known else None
            ),
            "known_result_prompt_tokens": sum(row["prompt_tokens"] or 0 for row in observed),
            "known_result_completion_tokens": sum(
                row["completion_tokens"] or 0 for row in observed
            ),
        }
    return {
        "format": "public-feedback-descriptive-summary-v1",
        "evidence_label": "scripted-real-tools-rehearsal"
        if scripted
        else "real-tools-trusted-fixtures",
        "cohort_status": ledger.status,
        "stop_reason": ledger.stop_reason,
        "arms": arms,
        "pairs": [
            {
                "repeat": repeat,
                **{
                    arm: next(row for row in rows if row["repeat"] == repeat and row["arm"] == arm)
                    for arm in ("control", "bool_check")
                },
            }
            for repeat in range(1, 5)
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
