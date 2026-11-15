"""Synthetic report controls; no measured model performance."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from evals.fresh_visible_runner import CellState, FreshLedger
from evals.fresh_visible_summary import fresh_summary
from evals.fresh_visible_workspaces import SELECTION_SHA256, selected_cell
from evals.real_run import write_report
from tests.evals import test_public_feedback_summary as summary_tests

row = summary_tests.row


def ledger(recorded=12, started=False, status="finished"):
    return FreshLedger(
        status=status,
        cells=[
            CellState(
                cell=selected_cell(i)[0],
                status=(
                    "recorded"
                    if i <= recorded
                    else "started"
                    if i == recorded + 1 and started
                    else "unrun"
                ),
            )
            for i in range(1, 13)
        ],
    )


def results(base, selected, count=12):
    return [
        replace(
            base,
            task_id=s.cell.task_id,
            mode="harness",
            attempt=s.cell.repeat,
            passed=s.cell.arm == "stronger",
            completion_status="completed",
            prompt_tokens=10,
            completion_tokens=5,
            runtime_config={
                "fresh_visible_preservation": {
                    "format": "fresh-visible-preservation-v1",
                    "selection_sha256": SELECTION_SHA256,
                    "ordinal": s.cell.ordinal,
                }
            },
            tool_trace=[{"name": "run_project_check", "result": {"exit_code": 0}}],
        )
        for s in selected.cells[:count]
    ]


async def test_six_pairs_and_per_task_counts_preserve_inputs_without_io(row, monkeypatch):
    selected = ledger()
    rows = results(row, selected)
    before = selected.model_dump(), [asdict(r) for r in rows]
    monkeypatch.setattr(Path, "read_bytes", lambda *a: pytest.fail("summary read input"))
    summary = fresh_summary(selected, rows, scripted=True)
    assert len(summary["pairs"]) == 6 and len(summary["tasks"]) == 3
    assert summary["arms"]["stronger"]["completed_and_accepted"] == 6
    assert summary["arms"]["weak"]["accepted"] == 0
    for counts in summary["tasks"].values():
        assert counts["stronger"]["accepted"] == 2 and counts["stronger"]["prompt_tokens"] == 20
        assert counts["weak"]["planned"] == 2
    assert [p["stronger"]["ordinal"] for p in summary["pairs"]] == [2, 7, 3, 10, 6, 11]
    assert summary["benefit_assessment"] == "not_assessed"
    assert summary["independent_regrades"] == "not_supplied"
    assert summary["evidence_label"] == "scripted-real-tools-rehearsal"
    assert before == (selected.model_dump(), [asdict(r) for r in rows])


async def test_stopped_result_unknown_usage_and_missing_stay_separate(row):
    selected = ledger(1, True, "stopped")
    rows = results(row, selected, 2)
    rows[1] = replace(rows[1], prompt_tokens=None, completion_tokens=None)
    summary = fresh_summary(selected, rows, scripted=False)
    assert summary["pairs"][0]["stronger"]["ledger_status"] == "started"
    assert summary["pairs"][1]["stronger"]["accepted"] is None
    assert summary["arms"]["stronger"]["unknown_usage_results"] == 1
    assert summary["arms"]["weak"]["known_result_prompt_tokens"] == 10
    assert all(arm["prompt_tokens"] is None for arm in summary["arms"].values())


async def test_completion_and_visible_checks_do_not_replace_acceptance(row):
    selected = ledger()
    rows = [
        replace(r, completion_status="incomplete", verification_status="verified")
        for r in results(row, selected)
    ]
    summary = fresh_summary(selected, rows, scripted=False)
    assert summary["arms"]["stronger"]["accepted"] == 6
    assert summary["arms"]["stronger"]["completed_and_accepted"] == 0
    assert summary["pairs"][0]["weak"]["visible_check_exit_codes"] == [0]
    assert summary["pairs"][0]["weak"]["accepted"] is False


@pytest.mark.parametrize(
    "fault",
    ["ordinal", "task", "arm", "repeat", "swap", "gap", "missing", "pin", "row", "finished"],
)
async def test_invalid_binding_refuses(row, fault):
    selected = ledger()
    rows = results(row, selected)
    if fault in {"ordinal", "task", "arm", "repeat"}:
        field, value = {
            "ordinal": ("ordinal", 2),
            "task": ("task_id", "fv_merge_windows"),
            "arm": ("arm", "stronger"),
            "repeat": ("repeat", 2),
        }[fault]
        selected.cells[0] = selected.cells[0].model_copy(
            update={"cell": selected.cells[0].cell.model_copy(update={field: value})}
        )
    elif fault == "swap":
        rows[0], rows[1] = rows[1], rows[0]
    elif fault == "gap":
        selected.cells[0] = selected.cells[0].model_copy(update={"status": "unrun"})
    elif fault == "missing":
        rows.pop()
    elif fault == "pin":
        rows[0] = replace(rows[0], runtime_config={})
    elif fault == "row":
        rows[0] = replace(rows[0], attempt=2)
    else:
        selected = ledger(1, False, "finished")
        rows = rows[:1]
    with pytest.raises(ValueError):
        fresh_summary(selected, rows, scripted=True)


async def test_atomic_report_keeps_raw_rows_and_previous_file_on_invalid_snapshot(row, tmp_path):
    selected = ledger()
    rows = results(row, selected)
    path = tmp_path / "report.json"
    write_report(path, "synthetic", rows, cohort=selected, scripted=True)
    report = json.loads(path.read_bytes())
    assert report["fresh_comparison"] == fresh_summary(selected, rows, scripted=True)
    assert report["results"] == json.loads(json.dumps([asdict(r) for r in rows]))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        write_report(path, "synthetic", rows[:1], cohort=selected, scripted=True)
    assert path.read_bytes() == before
    write_report(path, "synthetic", rows)
    assert "fresh_comparison" not in json.loads(path.read_bytes())


def test_planned_snapshot_preserves_missing_without_io(monkeypatch):
    selected = ledger(0, False, "planned")
    monkeypatch.setattr(Path, "read_bytes", lambda *a: pytest.fail("input read"))
    summary = fresh_summary(selected, [], scripted=True)
    assert len(summary["pairs"]) == 6
    for arm in summary["arms"].values():
        assert arm["accepted"] == 0 and arm["missing_results"] == 6
        assert arm["prompt_tokens"] is None


async def test_complete_unknown_usage_does_not_poison_other_task_totals(row):
    selected = ledger()
    rows = results(row, selected)
    rows[1] = replace(rows[1], prompt_tokens=None)
    summary = fresh_summary(selected, rows, scripted=False)
    assert summary["arms"]["stronger"]["prompt_tokens"] is None
    assert summary["tasks"]["fv_lookup_option"]["stronger"]["prompt_tokens"] is None
    assert summary["tasks"]["fv_merge_windows"]["stronger"]["prompt_tokens"] == 20
