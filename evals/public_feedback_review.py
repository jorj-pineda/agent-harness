"""Pure pinned feedback review join; caller observations are not authenticated."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import Field, TypeAdapter

from evals.heldout_execution_manifest import Digest, FrozenModel
from evals.public_feedback import FeedbackCell
from evals.public_feedback_regrade import FeedbackRegrades
from evals.public_feedback_runner import FeedbackLedger
from evals.public_feedback_summary import feedback_summary
from evals.real_run import RealTaskResult


class ReviewEvidence(FrozenModel):
    reference: str = Field(min_length=1, max_length=1000)
    sha256: Digest


class CellReview(FrozenModel):
    cell: FeedbackCell
    artifact_sha256: Digest | None
    manual_rescue: Literal["absent", "present", "unknown"]
    test_weakening: Literal["absent", "present", "unknown"]
    tool_misuse: Literal["absent", "present", "unknown"]
    other_regression: Literal["absent", "present", "unknown"]
    settlement: Literal["clean_unforced", "failed_or_forced", "unknown"]
    coverage: Literal["complete", "incomplete"]
    evidence: list[ReviewEvidence] = Field(max_length=100)
    notes: str = Field(min_length=1, max_length=8000)


class FeedbackReview(FrozenModel):
    format: Literal["public-feedback-review-v1"] = "public-feedback-review-v1"
    manifest_sha256: Digest
    report_sha256: Digest
    regrades_sha256: Digest
    reviewer: str = Field(min_length=1, max_length=200)
    cells: list[CellReview] = Field(min_length=8, max_length=8)


def _selected(raw: bytes, digest: str, limit: int) -> bytes:
    TypeAdapter(Digest).validate_python(digest, strict=True)
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Feedback review selected bytes differ or exceed limit")
    return raw


def review_feedback_run(
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
    grades = FeedbackRegrades.model_validate_json(_selected(regrades, regrades_sha256, 200_000))
    inspected = FeedbackReview.model_validate_json(_selected(review, review_sha256, 200_000))
    mode = raw["mode"]
    if mode not in {"scripted-real-tools-rehearsal", "real-tools-trusted-fixtures"}:
        raise ValueError("Unknown feedback evidence mode")
    if (
        grades.report_sha256 != report_sha256
        or inspected.report_sha256 != report_sha256
        or inspected.regrades_sha256 != regrades_sha256
        or inspected.manifest_sha256 != grades.manifest_sha256
        or grades.source_report_mode != mode
    ):
        raise ValueError("Feedback review input bindings differ")
    ledger = FeedbackLedger.model_validate(raw["cohort"])
    results = [
        TypeAdapter(RealTaskResult).validate_json(json.dumps(row), strict=True)
        for row in raw["results"]
    ]
    summary = feedback_summary(ledger, results, scripted=mode == "scripted-real-tools-rehearsal")
    # This v1 join reviews the fixed protocol, independent of mutable repo files.
    expected_arms = (
        "control",
        "bool_check",
        "bool_check",
        "control",
        "control",
        "bool_check",
        "bool_check",
        "control",
    )
    if any(
        state.cell.ordinal != index + 1
        or state.cell.repeat != index // 2 + 1
        or state.cell.arm != expected_arms[index]
        for index, state in enumerate(ledger.cells)
    ):
        raise ValueError("Feedback review schedule differs")
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
            raise ValueError("Feedback review cell identity/artifact differs")
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
                )
            ):
                raise ValueError("Feedback missing result has regrade outcome")
            if (
                observation.coverage == "complete"
                or observation.manual_rescue != "unknown"
                or observation.test_weakening != "unknown"
                or observation.tool_misuse != "unknown"
                or observation.other_regression != "unknown"
                or observation.settlement != "unknown"
            ):
                raise ValueError("Missing feedback execution cannot have a clean review")
        else:
            if (
                grade.status != "graded"
                or grade.artifact_sha256 is None
                or grade.regrade_exit_code is None
                or grade.scoped_source_unchanged is not True
                or grade.recorded_passed != result.passed
                or grade.recorded_completion != result.completion_status
                or grade.recorded_acceptance_exit_code != result.acceptance_exit_code
            ):
                raise ValueError("Feedback regrade original outcome differs")
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
    for repeat in range(1, 5):
        control = next(
            g for g in grades.cells if g.cell.repeat == repeat and g.cell.arm == "control"
        )
        treatment = next(
            g for g in grades.cells if g.cell.repeat == repeat and g.cell.arm == "bool_check"
        )
        known_pair = (
            control.regrade_exit_code is not None and treatment.regrade_exit_code is not None
        )
        pairs.append(
            {
                "repeat": repeat,
                "control_ordinal": control.cell.ordinal,
                "bool_check_ordinal": treatment.cell.ordinal,
                "control_regrade_exit_code": control.regrade_exit_code,
                "bool_check_regrade_exit_code": treatment.regrade_exit_code,
                "acceptance_regression": (
                    control.regrade_exit_code == 0 and treatment.regrade_exit_code != 0
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
        and all(pair["acceptance_regression"] is False for pair in pairs)
        and summary["arms"]["bool_check"]["completed_and_accepted"]
        > summary["arms"]["control"]["completed_and_accepted"]
    )
    return {
        "format": "public-feedback-review-summary-v1",
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
