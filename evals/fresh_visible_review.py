"""Pure pinned fresh review join; caller observations are not authenticated."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field, TypeAdapter

from evals.fresh_visible_regrade import FreshRegrades
from evals.fresh_visible_runner import FreshLedger
from evals.fresh_visible_summary import fresh_summary
from evals.fresh_visible_workspaces import FreshCell
from evals.heldout_execution_manifest import Digest, FrozenModel
from evals.public_feedback_review import ReviewEvidence, _selected
from evals.real_run import RealTaskResult


class CellReview(FrozenModel):
    cell: FreshCell
    artifact_sha256: Digest | None
    manual_rescue: Literal["absent", "present", "unknown"]
    test_weakening: Literal["absent", "present", "unknown"]
    tool_misuse: Literal["absent", "present", "unknown"]
    other_regression: Literal["absent", "present", "unknown"]
    settlement: Literal["clean_unforced", "failed_or_forced", "unknown"]
    coverage: Literal["complete", "incomplete"]
    evidence: list[ReviewEvidence] = Field(max_length=100)
    notes: str = Field(min_length=1, max_length=8000)


class FreshReview(FrozenModel):
    format: Literal["fresh-visible-review-v1"] = "fresh-visible-review-v1"
    manifest_sha256: Digest
    report_sha256: Digest
    regrades_sha256: Digest
    reviewer: str = Field(min_length=1, max_length=200)
    cells: list[CellReview] = Field(min_length=12, max_length=12)


def review_fresh_run(
    *,
    report: bytes,
    report_sha256: str,
    regrades: bytes,
    regrades_sha256: str,
    review: bytes,
    review_sha256: str,
) -> dict[str, Any]:
    """Join selected bounded bytes without IO, execution, regrading or approval.

    Evidence references and inspection coverage are caller assertions. This checks
    consistency, not their truth or external reference contents. Benefit remains
    not_assessed even when the limited public-follow-up rule is eligible for review.
    """
    raw = json.loads(_selected(report, report_sha256, 2_000_000))
    grades = FreshRegrades.model_validate_json(_selected(regrades, regrades_sha256, 200_000))
    inspected = FreshReview.model_validate_json(_selected(review, review_sha256, 200_000))
    mode = raw["mode"]
    if mode not in {"scripted-real-tools-rehearsal", "real-tools-trusted-fixtures"}:
        raise ValueError("Unknown fresh evidence mode")
    if (
        grades.report_sha256 != report_sha256
        or inspected.report_sha256 != report_sha256
        or inspected.regrades_sha256 != regrades_sha256
        or inspected.manifest_sha256 != grades.manifest_sha256
        or grades.source_report_mode != mode
    ):
        raise ValueError("Fresh review input bindings differ")
    ledger = FreshLedger.model_validate(raw["cohort"])
    results = [
        TypeAdapter(RealTaskResult).validate_json(json.dumps(row), strict=True)
        for row in raw["results"]
    ]
    summary = fresh_summary(ledger, results, scripted=mode == "scripted-real-tools-rehearsal")
    # fresh_summary enforces the complete fixed v1 schedule in memory.
    if set(grades.criteria_sha256) != set(summary["tasks"]):
        raise ValueError("Fresh criterion task bindings differ")
    for result in results:
        identity = result.acceptance_identity or {}
        criterion = grades.criteria_sha256[result.task_id]
        if (
            result.acceptance_policy != "owned-pytest-v1"
            or identity.get("status") != "matching"
            or identity.get("before_sha256") != criterion
            or identity.get("after_sha256") != criterion
        ):
            raise ValueError("Fresh original criterion binding differs from its task")
    clean = True
    disagreements = []
    unknown = []
    for index, (state, grade, observation) in enumerate(
        zip(ledger.cells, grades.cells, inspected.cells, strict=True)
    ):
        if (
            grade.cell != state.cell
            or observation.cell != state.cell
            or grade.ledger_status != state.status
            or observation.artifact_sha256 != grade.artifact_sha256
        ):
            raise ValueError("Fresh review cell identity/artifact differs")
        result = results[index] if index < len(results) else None
        if result is None:
            if grade.status != "no_result" or any(
                value is not None
                for value in (
                    grade.recorded_passed,
                    grade.recorded_completion,
                    grade.recorded_acceptance_exit_code,
                    grade.regrade_exit_code,
                    grade.artifact_sha256,
                    grade.matches_recorded_exit_code,
                    grade.scoped_source_unchanged,
                    grade.acceptance_output,
                    grade.acceptance_output_truncated,
                )
            ):
                raise ValueError("Fresh missing result has regrade outcome")
            if (
                observation.coverage == "complete"
                or observation.manual_rescue != "unknown"
                or observation.test_weakening != "unknown"
                or observation.tool_misuse != "unknown"
                or observation.other_regression != "unknown"
                or observation.settlement != "unknown"
            ):
                raise ValueError("Missing fresh execution cannot have a clean review")
        else:
            if (
                grade.status != "graded"
                or grade.artifact_sha256 is None
                or grade.regrade_exit_code is None
                or grade.scoped_source_unchanged is not True
                or grade.recorded_passed != result.passed
                or grade.recorded_completion != result.completion_status
                or grade.recorded_acceptance_exit_code != result.acceptance_exit_code
                or grade.matches_recorded_exit_code
                is not (grade.regrade_exit_code == result.acceptance_exit_code)
            ):
                raise ValueError("Fresh regrade original outcome differs")
            if (
                grade.regrade_exit_code != result.acceptance_exit_code
                or (grade.regrade_exit_code == 0) != result.passed
            ):
                disagreements.append(state.cell.ordinal)
        known = (
            result is not None
            and state.status == "recorded"
            and observation.coverage == "complete"
            and bool(observation.evidence)
            and observation.manual_rescue != "unknown"
            and observation.test_weakening != "unknown"
            and observation.tool_misuse != "unknown"
            and observation.other_regression != "unknown"
            and observation.settlement != "unknown"
            and result.prompt_tokens is not None
            and result.completion_tokens is not None
        )
        if not known:
            unknown.append(state.cell.ordinal)
        clean = (
            clean
            and known
            and (
                observation.manual_rescue == "absent"
                and observation.test_weakening == "absent"
                and observation.tool_misuse == "absent"
                and observation.other_regression == "absent"
                and observation.settlement == "clean_unforced"
            )
        )
    pairs = []
    for task_id in summary["tasks"]:
        for repeat in (1, 2):
            weak = next(
                g
                for g in grades.cells
                if g.cell.task_id == task_id and g.cell.repeat == repeat and g.cell.arm == "weak"
            )
            treatment = next(
                g
                for g in grades.cells
                if g.cell.task_id == task_id
                and g.cell.repeat == repeat
                and g.cell.arm == "stronger"
            )
            known_pair = (
                weak.regrade_exit_code is not None and treatment.regrade_exit_code is not None
            )
            pairs.append(
                {
                    "task_id": task_id,
                    "repeat": repeat,
                    "weak_ordinal": weak.cell.ordinal,
                    "stronger_ordinal": treatment.cell.ordinal,
                    "weak_regrade_exit_code": weak.regrade_exit_code,
                    "stronger_regrade_exit_code": treatment.regrade_exit_code,
                    "acceptance_regression": (
                        weak.regrade_exit_code == 0 and treatment.regrade_exit_code != 0
                        if known_pair
                        else None
                    ),
                }
            )
    eligible = (
        mode == "real-tools-trusted-fixtures"
        and ledger.status == "finished"
        and clean
        and not disagreements
        and all(
            task["stronger"]["completed_and_accepted"] >= task["weak"]["completed_and_accepted"]
            for task in summary["tasks"].values()
        )
        and all(pair["acceptance_regression"] is False for pair in pairs)
        and summary["arms"]["stronger"]["completed_and_accepted"]
        > summary["arms"]["weak"]["completed_and_accepted"]
    )
    return {
        "format": "fresh-visible-review-summary-v1",
        "report_sha256": report_sha256,
        "regrades_sha256": regrades_sha256,
        "review_sha256": review_sha256,
        "manifest_sha256": grades.manifest_sha256,
        "evidence_label": mode,
        "recorded_outcomes": summary,
        "pairs": pairs,
        "unknown_review_ordinals": unknown,
        "regrade_disagreement_ordinals": disagreements,
        "public_followup_rule_eligible_for_human_review": eligible,
        "benefit_assessment": "not_assessed",
        "reviewer_assertions_authenticated": False,
        "external_evidence_references_verified": False,
        "inference_ran": False,
        "release_decision": "no_go",
    }
