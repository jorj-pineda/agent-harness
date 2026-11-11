"""Synthetic review joins; no reviewer authentication or model-quality evidence."""

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from evals.public_feedback_regrade import FeedbackRegrades, RegradedCell
from evals.public_feedback_review import review_feedback_run
from tests.evals.test_public_feedback_summary import ledger, results
from tests.evals.test_public_feedback_summary import row as row


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def inputs(row, *, scripted=False, count=8):
    selected = ledger() if count == 8 else ledger(count, False, "stopped")
    rows = [
        replace(r, acceptance_exit_code=0 if r.passed else 1) for r in results(row, selected, count)
    ]
    mode = "scripted-real-tools-rehearsal" if scripted else "real-tools-trusted-fixtures"
    report = {"mode": mode, "cohort": selected.model_dump(), "results": [asdict(r) for r in rows]}
    grades = FeedbackRegrades(
        manifest_sha256="a" * 64,
        report_sha256=digest(json.dumps(report).encode()),
        criterion_sha256="b" * 64,
        source_report_mode=mode,
        grader_revision={},
        cells=[
            RegradedCell(
                cell=s.cell,
                ledger_status=s.status,
                status="graded" if i < count else "no_result",
                recorded_passed=rows[i].passed if i < count else None,
                recorded_completion=rows[i].completion_status if i < count else None,
                recorded_acceptance_exit_code=rows[i].acceptance_exit_code if i < count else None,
                regrade_exit_code=rows[i].acceptance_exit_code if i < count else None,
                artifact_sha256="c" * 64 if i < count else None,
                scoped_source_unchanged=True if i < count else None,
            )
            for i, s in enumerate(selected.cells)
        ],
    ).model_dump(mode="json")
    review = {
        "manifest_sha256": "a" * 64,
        "report_sha256": grades["report_sha256"],
        "regrades_sha256": digest(json.dumps(grades).encode()),
        "reviewer": "synthetic fixture",
        "cells": [
            {
                "cell": s.cell.model_dump(),
                "artifact_sha256": "c" * 64 if i < count else None,
                "manual_rescue": "absent" if i < count else "unknown",
                "test_weakening": "absent" if i < count else "unknown",
                "tool_misuse": "absent" if i < count else "unknown",
                "other_regression": "absent" if i < count else "unknown",
                "settlement": "clean_unforced" if i < count else "unknown",
                "coverage": "complete" if i < count else "incomplete",
                "evidence": [{"reference": "synthetic-only", "sha256": "d" * 64}],
                "notes": "Synthetic coverage assertion; not actual review.",
            }
            for i, s in enumerate(selected.cells)
        ],
    }
    return report, grades, review


def join(report, grades, review, *, bind=True):
    report_bytes = json.dumps(report).encode()
    if bind:
        grades["report_sha256"] = digest(report_bytes)
    grade_bytes = json.dumps(grades).encode()
    if bind:
        review["report_sha256"] = digest(report_bytes)
        review["regrades_sha256"] = digest(grade_bytes)
    review_bytes = json.dumps(review).encode()
    return review_feedback_run(
        report=report_bytes,
        report_sha256=digest(report_bytes),
        regrades=grade_bytes,
        regrades_sha256=digest(grade_bytes),
        review=review_bytes,
        review_sha256=digest(review_bytes),
    )


async def test_complete_live_rule_only_eligible_for_human_review(row):
    data = inputs(row)
    before = json.dumps(data)
    summary = join(*data)
    assert summary["public_followup_rule_eligible_for_human_review"] is True
    assert summary["benefit_assessment"] == "not_assessed"
    assert summary["external_evidence_references_verified"] is False
    assert summary["release_decision"] == "no_go"
    assert json.dumps(data) == before


async def test_scripted_advantage_never_live_benefit(row):
    assert not join(*inputs(row, scripted=True))["public_followup_rule_eligible_for_human_review"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("manual_rescue", "present"),
        ("manual_rescue", "unknown"),
        ("test_weakening", "present"),
        ("tool_misuse", "present"),
        ("other_regression", "present"),
        ("settlement", "failed_or_forced"),
        ("coverage", "incomplete"),
        ("evidence", []),
    ],
)
async def test_unclean_or_unknown_review_prevents_eligibility(row, field, value):
    report, grades, review = inputs(row)
    review["cells"][0][field] = value
    assert not join(report, grades, review)["public_followup_rule_eligible_for_human_review"]


async def test_stopped_missing_and_started_stay_unknown(row):
    report, grades, review = inputs(row, count=2)
    summary = join(report, grades, review)
    assert summary["unknown_review_ordinals"] == [3, 4, 5, 6, 7, 8]
    assert summary["pairs"][1]["acceptance_regression"] is None
    assert not summary["public_followup_rule_eligible_for_human_review"]
    report["cohort"]["cells"][1]["status"] = "started"
    grades["cells"][1]["ledger_status"] = "started"
    assert 2 in join(report, grades, review)["unknown_review_ordinals"]


async def test_regression_and_regrade_disagreement_remain_visible(row):
    report, grades, review = inputs(row)
    grades["cells"][0]["regrade_exit_code"] = 0
    grades["cells"][1]["regrade_exit_code"] = 1
    summary = join(report, grades, review)
    assert summary["pairs"][0]["acceptance_regression"] is True
    assert summary["regrade_disagreement_ordinals"] == [1, 2]
    assert not summary["public_followup_rule_eligible_for_human_review"]


@pytest.mark.parametrize(
    "change", ["binding", "ordinal", "artifact", "original", "mode", "missing"]
)
async def test_mismatched_selected_evidence_refuses(row, change):
    report, grades, review = inputs(row)
    if change == "binding":
        review["manifest_sha256"] = "e" * 64
    elif change == "ordinal":
        review["cells"][0]["cell"]["ordinal"] = 2
    elif change == "artifact":
        review["cells"][0]["artifact_sha256"] = "e" * 64
    elif change == "original":
        grades["cells"][0]["recorded_passed"] = True
    elif change == "mode":
        grades["source_report_mode"] = "scripted-real-tools-rehearsal"
    else:
        grades["cells"][0]["status"] = "no_result"
    with pytest.raises(ValueError):
        join(report, grades, review)


async def test_hash_mismatch_refuses_before_parsing():
    with pytest.raises(ValueError, match="selected bytes"):
        review_feedback_run(
            report=b"bad",
            report_sha256="0" * 64,
            regrades=b"",
            regrades_sha256="0" * 64,
            review=b"",
            review_sha256="0" * 64,
        )


async def test_tie_and_unknown_usage_never_eligible(row):
    report, grades, review = inputs(row)
    report["results"][0]["prompt_tokens"] = None
    summary = join(report, grades, review)
    assert 1 in summary["unknown_review_ordinals"]
    assert not summary["public_followup_rule_eligible_for_human_review"]
    report, grades, review = inputs(row)
    for i in range(8):
        report["results"][i]["passed"] = True
        report["results"][i]["acceptance_exit_code"] = 0
        grades["cells"][i]["recorded_passed"] = True
        grades["cells"][i]["recorded_acceptance_exit_code"] = 0
        grades["cells"][i]["regrade_exit_code"] = 0
    assert not join(report, grades, review)["public_followup_rule_eligible_for_human_review"]


async def test_stale_regrade_bytes_binding_refuses(row):
    report, grades, review = inputs(row)
    review["regrades_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="bindings"):
        join(report, grades, review, bind=False)


async def test_join_never_reads_filesystem(row, monkeypatch):
    from pathlib import Path

    data = inputs(row)

    def forbidden(*args, **kwargs):
        raise AssertionError("Pure join attempted filesystem access")

    with monkeypatch.context() as guard:
        guard.setattr("builtins.open", forbidden)
        guard.setattr("io.open", forbidden)
        guard.setattr(Path, "read_bytes", forbidden)
        guard.setattr(Path, "read_text", forbidden)
        assert join(*data)["public_followup_rule_eligible_for_human_review"]


async def test_join_refuses_consistently_reordered_arms(row):
    report, grades, review = inputs(row)
    for index in (0, 1):
        arm = "bool_check" if index == 0 else "control"
        report["cohort"]["cells"][index]["cell"]["arm"] = arm
        grades["cells"][index]["cell"]["arm"] = arm
        review["cells"][index]["cell"]["arm"] = arm
    with pytest.raises(ValueError, match="schedule differs"):
        join(report, grades, review)
