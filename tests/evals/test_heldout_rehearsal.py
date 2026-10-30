import json
from dataclasses import replace

import pytest

from evals import heldout_rehearsal as rehearsal
from evals.heldout_comparison import validate_plan
from harness import providers


async def test_all_sixteen_canned_cells_use_real_tools_and_grading_without_live_provider(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        providers, "build_configured_provider", lambda *a: pytest.fail("live provider")
    )
    monkeypatch.setenv("MAX_TOOL_ITERATIONS", "1")
    report = tmp_path / "rehearsal.json"
    ledger = await rehearsal.rehearse(report)
    data = json.loads(report.read_text())
    assert ledger.status == "finished" and len(ledger.cells) == 16
    assert data["mode"] == "scripted-real-tools-rehearsal"
    assert len(data["results"]) == 16
    for row in data["results"]:
        assert row["passed"] and row["completion_status"] == "completed"
        assert row["heldout_preservation"]["status"] == "checks_passed"
        assert row["runtime_config"]["max_tool_iterations"] == 6
        if row["category"] == "repository_question":
            assert not row["modified"] and row["verification_status"] == "not_run"
            assert row["runtime_config"]["inspection_only"]
        else:
            assert row["modified"] and row["verification_status"] == "passed"
    for mode in ("harness", "minimal"):
        assert data["summary"][mode]["attempts"] == 8


@pytest.mark.parametrize("fault", ["missing_usage", "wrong_model", "transport"])
async def test_boundary_fault_stops_before_tool_dispatch_and_next_cell(
    monkeypatch, tmp_path, fault
):
    made = []
    original = rehearsal.CannedProvider

    def make(*args):
        provider = original(*args)
        made.append(provider)
        return provider

    monkeypatch.setattr(rehearsal, "CannedProvider", make)
    report = tmp_path / "rehearsal.json"
    with pytest.raises(rehearsal.ProtocolStop):
        await rehearsal.rehearse(report, fault=fault)
    data = json.loads(report.read_text())
    assert len(made) == 1 and made[0].calls == 1 and made[0].closed
    assert len(data["results"]) == 1 and data["cohort"]["stop_reason"] == fault
    row = data["results"][0]
    assert not row["passed"] and not row["tool_trace"] and not row["modified"]
    assert row["prompt_tokens"] is None and row["completion_tokens"] is None
    assert [c["status"] for c in data["cohort"]["cells"]] == ["recorded"] + ["unrun"] * 15


async def test_result_policy_accepts_artifact_rejection_but_refuses_missing_evidence(tmp_path):
    report = tmp_path / "rehearsal.json"
    await rehearsal.rehearse(report)
    from evals.real_run import RealTaskResult

    row = RealTaskResult(**json.loads(report.read_text())["results"][0])
    plan = validate_plan()
    assert (
        rehearsal.stop_reason(replace(row, passed=False, termination="acceptance_failed"), plan)
        is None
    )
    for changed, reason in [
        (replace(row, prompt_tokens=None), "missing_usage"),
        (replace(row, model="other"), "model_identity"),
        (replace(row, acceptance_identity=None), "criterion_identity"),
        (replace(row, heldout_preservation=None), "preservation"),
        (replace(row, termination="error:TimeoutError"), "runtime_or_provider_error"),
    ]:
        assert rehearsal.stop_reason(changed, plan) == reason


async def test_checked_provider_latches_failure_before_any_second_dispatch():
    plan = validate_plan()
    from evals.heldout_guard import REGISTRATIONS

    task = rehearsal.load_tasks(REGISTRATIONS[0].root / "tasks.yaml")[0]
    raw = rehearsal.CannedProvider(task, plan, "missing_usage")
    checked = rehearsal.CheckedProvider(raw, plan)
    for _ in range(2):
        with pytest.raises(rehearsal.ProtocolStop, match="missing_usage"):
            await checked.chat([])
    assert raw.calls == 1
    await checked.aclose()
    assert raw.closed
