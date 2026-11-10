"""Synthetic aggregation controls; no model-performance or regrade observations."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from evals.public_feedback import PLAN_SHA256, validate_feedback_plan
from evals.public_feedback_runner import CellState, FeedbackLedger
from evals.public_feedback_summary import feedback_summary
from evals.real_run import load_tasks, run_task, write_report
from harness.config import Settings
from tests.api.conftest import ScriptedProvider, make_response


@pytest.fixture
async def row():
    provider = ScriptedProvider()
    provider.script(make_response(content="No patch", prompt_tokens=10, completion_tokens=5))
    return await run_task(
        next(task for task in load_tasks() if task.id == "paginate_items"),
        provider=provider,
        settings=Settings(_env_file=None),
        mode="harness",
    )


def ledger(recorded=8, started=False, status="finished"):
    return FeedbackLedger(
        status=status,
        cells=[
            CellState(
                cell=cell,
                status=(
                    "recorded"
                    if i < recorded
                    else "started"
                    if i == recorded and started
                    else "unrun"
                ),
            )
            for i, cell in enumerate(validate_feedback_plan().cells)
        ],
    )


def results(row, selected, count=8):
    return [
        replace(
            row,
            task_id=state.cell.task_id,
            mode=state.cell.mode,
            attempt=state.cell.repeat,
            passed=state.cell.arm == "bool_check",
            completion_status="completed",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=123.0,
            runtime_config={
                **row.runtime_config,
                "public_feedback_preservation": {
                    "format": "public-feedback-preservation-v1",
                    "plan_sha256": PLAN_SHA256,
                    "ordinal": state.cell.ordinal,
                },
            },
            tool_trace=[{"name": "run_project_check", "result": {"exit_code": 0}}],
        )
        for state in selected.cells[:count]
    ]


async def test_full_pairs_keep_arm_counts_labels_and_review_unknown(row):
    selected = ledger()
    rows = results(row, selected)
    before = selected.model_dump(), [asdict(result) for result in rows]
    summary = feedback_summary(selected, rows, scripted=True)
    assert summary["evidence_label"] == "scripted-real-tools-rehearsal"
    assert summary["arms"]["control"]["accepted"] == 0
    assert summary["arms"]["bool_check"]["completed_and_accepted"] == 4
    for arm in summary["arms"].values():
        assert arm["planned"] == arm["results_present"] == arm["recorded_cells"] == 4
        assert arm["prompt_tokens"] == 40 and arm["completion_tokens"] == 20
    assert [pair["repeat"] for pair in summary["pairs"]] == [1, 2, 3, 4]
    assert [pair["control"]["ordinal"] for pair in summary["pairs"]] == [1, 4, 5, 8]
    assert [pair["bool_check"]["ordinal"] for pair in summary["pairs"]] == [2, 3, 6, 7]
    assert summary["benefit_assessment"] == "not_assessed"
    assert summary["independent_regrades"] == "not_supplied"
    assert summary["manual_rescues"] is None and summary["new_regressions"] is None
    assert before == (selected.model_dump(), [asdict(result) for result in rows])


async def test_visible_pass_and_verified_state_do_not_replace_artifact_or_completion(row):
    selected = ledger()
    rows = [
        replace(result, verification_status="verified", completion_status="incomplete")
        for result in results(row, selected)
    ]
    summary = feedback_summary(selected, rows, scripted=False)
    assert summary["evidence_label"] == "real-tools-trusted-fixtures"
    assert summary["arms"]["bool_check"]["accepted"] == 4
    assert summary["arms"]["bool_check"]["completed_and_accepted"] == 0
    control = summary["pairs"][0]["control"]
    assert control["visible_check_exit_codes"] == [0]
    assert control["verification_status"] == "verified" and control["accepted"] is False


async def test_stopped_started_result_stays_distinct_from_recorded_and_missing(row):
    selected = ledger(1, True, "stopped").model_copy(update={"stop_reason": "transport"})
    rows = results(row, selected, 2)
    rows[1] = replace(rows[1], prompt_tokens=None, completion_tokens=None)
    summary = feedback_summary(selected, rows, scripted=False)
    assert summary["cohort_status"] == "stopped" and summary["stop_reason"] == "transport"
    first, second = summary["pairs"][0]["control"], summary["pairs"][0]["bool_check"]
    assert first["ledger_status"] == "recorded" and second["ledger_status"] == "started"
    assert second["result_present"] and second["prompt_tokens"] is None
    later = summary["pairs"][1]["bool_check"]
    assert later["ledger_status"] == "unrun" and not later["result_present"]
    assert later["accepted"] is None and later["completed_and_accepted"] is None
    assert summary["arms"]["control"]["known_result_prompt_tokens"] == 10
    assert summary["arms"]["bool_check"]["unknown_usage_results"] == 1
    assert all(arm["prompt_tokens"] is None for arm in summary["arms"].values())


async def test_full_cohort_unknown_usage_is_not_a_zero_total(row):
    selected = ledger()
    rows = results(row, selected)
    rows[7] = replace(rows[7], prompt_tokens=None, completion_tokens=None)
    control = feedback_summary(selected, rows, scripted=False)["arms"]["control"]
    assert control["results_present"] == 4 and control["missing_results"] == 0
    assert control["prompt_tokens"] is None and control["completion_tokens"] is None
    assert control["unknown_usage_results"] == 1 and control["known_result_prompt_tokens"] == 30


def test_planned_report_does_not_invent_failed_attempts_or_known_usage(monkeypatch):
    selected = ledger(0, False, "planned")
    monkeypatch.setattr(Path, "read_bytes", lambda *a: pytest.fail("summary read an input"))
    summary = feedback_summary(selected, [], scripted=True)
    for arm in summary["arms"].values():
        assert arm["results_present"] == 0 and arm["missing_results"] == 4
        assert arm["accepted"] == 0 and arm["prompt_tokens"] is None


@pytest.mark.parametrize(
    "fault",
    ["ordinal", "arm", "repeat", "gap", "finished", "row_identity", "missing_row", "swapped_arms"],
)
async def test_mismatched_snapshots_refuse_instead_of_misassigning_pairs(row, fault):
    selected = ledger()
    rows = results(row, selected)
    if fault in {"ordinal", "arm", "repeat"}:
        field, value = {
            "ordinal": ("ordinal", 2),
            "arm": ("arm", "bool_check"),
            "repeat": ("repeat", 4),
        }[fault]
        selected.cells[0] = selected.cells[0].model_copy(
            update={"cell": selected.cells[0].cell.model_copy(update={field: value})}
        )
    elif fault == "gap":
        selected.cells[0] = selected.cells[0].model_copy(update={"status": "unrun"})
    elif fault == "finished":
        selected = ledger(1, False, "finished")
        rows = rows[:1]
    elif fault == "row_identity":
        rows[0] = replace(rows[0], attempt=4)
    elif fault == "swapped_arms":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows.pop()
    with pytest.raises(ValueError):
        feedback_summary(selected, rows, scripted=True)


async def test_report_adds_feedback_summary_only_to_feedback_and_keeps_previous_on_failure(
    row, tmp_path
):
    selected = ledger()
    rows = results(row, selected)
    path = tmp_path / "report.json"
    write_report(path, "synthetic", rows, cohort=selected, scripted=True)
    report = json.loads(path.read_bytes())
    assert report["feedback_comparison"] == feedback_summary(selected, rows, scripted=True)
    assert report["results"] == json.loads(json.dumps([asdict(result) for result in rows]))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        write_report(path, "synthetic", rows[:1], cohort=selected, scripted=True)
    assert path.read_bytes() == before
    write_report(path, "synthetic", rows)
    assert "feedback_comparison" not in json.loads(path.read_bytes())
