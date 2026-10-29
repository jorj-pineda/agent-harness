"""Scripted cohort bookkeeping; these checks measure no model quality."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from evals import real_run
from harness.config import Settings
from tests.api.conftest import ScriptedProvider, make_response


def setup_cli(monkeypatch, tmp_path):
    provider = ScriptedProvider()
    provider.script(*[make_response(content="No patch") for _ in range(4)])
    provider.aclose = AsyncMock()
    monkeypatch.setattr(real_run, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(real_run, "build_configured_provider", lambda *args: provider)
    report = tmp_path / "report.json"
    args = ["--task", "divide_zero", "--mode", "both", "--repeats", "2", "--report", str(report)]
    return provider, report, args


def test_cli_persists_full_order_before_provider_construction_and_retains_final_links(
    monkeypatch, tmp_path
):
    provider, report, args = setup_cli(monkeypatch, tmp_path)
    prepared = []

    def construct(*args):
        data = json.loads(report.read_text())
        assert data["results"] == [] and data["summary"] == {}
        ledger = data["cohort"]
        assert ledger["format"] == "real-task-cohort-v1" and ledger["status"] == "prepared"
        assert [c["status"] for c in ledger["cells"]] == ["planned"] * 4
        assert [(c["mode"], c["attempt"]) for c in ledger["cells"]] == [
            ("harness", 1),
            ("minimal", 1),
            ("minimal", 2),
            ("harness", 2),
        ]
        assert all(c["task_set"]["partition"] == "development" for c in ledger["cells"])
        prepared.append(True)
        return provider

    monkeypatch.setattr(real_run, "build_configured_provider", construct)
    assert real_run.main(args) == 0
    data = json.loads(report.read_text())
    assert prepared == [True] and len(data["results"]) == 4
    assert data["cohort"]["status"] == "finished"
    for index, cell in enumerate(data["cohort"]["cells"]):
        assert cell["ordinal"] == index + 1
        assert cell["status"] == "recorded" and cell["result_index"] == index
        result = data["results"][cell["result_index"]]
        assert (cell["task_id"], cell["mode"], cell["attempt"], cell["task_set"]) == (
            result["task_id"],
            result["mode"],
            result["attempt"],
            result["task_set"],
        )
    provider.aclose.assert_awaited_once()


def test_exception_after_prefix_retains_no_result_and_unrun_cells(monkeypatch, tmp_path):
    provider, report, args = setup_cli(monkeypatch, tmp_path)
    original = real_run.run_task
    calls = []

    async def fail_second(*args, **kwargs):
        calls.append(kwargs["mode"])
        if len(calls) == 2:
            raise RuntimeError("private exception message must not enter ledger")
        return await original(*args, **kwargs)

    monkeypatch.setattr(real_run, "run_task", fail_second)
    with pytest.raises(RuntimeError):
        real_run.main(args)
    data = json.loads(report.read_text())
    ledger = data["cohort"]
    assert ledger["status"] == "stopped" and ledger["stop_reason"] == "RuntimeError"
    assert [c["status"] for c in ledger["cells"]] == ["recorded", "no_result", "unrun", "unrun"]
    assert [c["result_index"] for c in ledger["cells"]] == [0, None, None, None]
    assert len(data["results"]) == 1 and data["summary"]["harness"]["attempts"] == 1
    assert "minimal" not in data["summary"]
    assert "private exception message" not in json.dumps(ledger)
    assert calls == ["harness", "minimal"]
    provider.aclose.assert_awaited_once()


def test_cancelled_request_retains_plan_without_inventing_results_or_usage(monkeypatch, tmp_path):
    provider, report, args = setup_cli(monkeypatch, tmp_path)
    provider.chat = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        real_run.main(args)
    data = json.loads(report.read_text())
    assert data["results"] == [] and data["summary"] == {}
    assert data["cohort"]["stop_reason"] == "CancelledError"
    assert [c["status"] for c in data["cohort"]["cells"]] == [
        "no_result",
        "unrun",
        "unrun",
        "unrun",
    ]
    assert all(c["result_index"] is None for c in data["cohort"]["cells"])
    provider.chat.assert_awaited_once()
    provider.aclose.assert_awaited_once()


def test_provider_construction_failure_retains_all_cells_unrun(monkeypatch, tmp_path):
    _, report, args = setup_cli(monkeypatch, tmp_path)

    def fail(*args):
        raise ValueError("provider setup")

    monkeypatch.setattr(real_run, "build_configured_provider", fail)
    with pytest.raises(ValueError, match="provider setup"):
        real_run.main(args)
    data = json.loads(report.read_text())
    assert data["results"] == [] and data["cohort"]["status"] == "stopped"
    assert [c["status"] for c in data["cohort"]["cells"]] == ["unrun"] * 4


@pytest.mark.parametrize("initial", [True, False])
def test_report_failure_prevents_further_work_and_settles_owned_provider(
    monkeypatch, tmp_path, initial
):
    provider, report, args = setup_cli(monkeypatch, tmp_path)
    original = real_run.write_report
    writes = []
    builds = []

    def persist(*args, **kwargs):
        writes.append(True)
        if len(writes) == (1 if initial else 3):
            raise OSError("report persistence failed")
        return original(*args, **kwargs)

    def construct(*args):
        builds.append(True)
        return provider

    monkeypatch.setattr(real_run, "write_report", persist)
    monkeypatch.setattr(real_run, "build_configured_provider", construct)
    with pytest.raises(OSError):
        real_run.main(args)
    data = json.loads(report.read_text())
    assert data["cohort"]["status"] == "stopped"
    assert data["cohort"]["stop_reason"] == "OSError"
    assert len(provider.calls) == (0 if initial else 1)
    assert len(data["results"]) == (0 if initial else 1)
    assert builds == ([] if initial else [True])
    assert provider.aclose.await_count == (0 if initial else 1)


@pytest.mark.parametrize("primary", [True, False])
def test_cleanup_error_is_recorded_without_hiding_primary_failure(monkeypatch, tmp_path, primary):
    provider, report, args = setup_cli(monkeypatch, tmp_path)
    provider.aclose = AsyncMock(side_effect=LookupError("cleanup"))
    if primary:

        async def fail(*args, **kwargs):
            raise RuntimeError("primary")

        monkeypatch.setattr(real_run, "run_task", fail)
    with pytest.raises(RuntimeError if primary else LookupError):
        real_run.main(args)
    data = json.loads(report.read_text())
    assert data["cohort"]["status"] == "stopped"
    assert data["cohort"]["cleanup_error"] == "LookupError"
    assert data["cohort"]["stop_reason"] == ("RuntimeError" if primary else "LookupError")
    assert len(data["results"]) == (0 if primary else 4)
    provider.aclose.assert_awaited_once()


def test_oversized_plan_refuses_before_controls_or_provider_allocation(monkeypatch, tmp_path):
    _, report, args = setup_cli(monkeypatch, tmp_path)
    args[args.index("--repeats") + 1] = str(real_run.MAX_COHORT_CELLS)
    monkeypatch.setattr(real_run, "validate_task", lambda *args: pytest.fail("allocated controls"))
    monkeypatch.setattr(
        real_run, "build_configured_provider", lambda *args: pytest.fail("provider")
    )
    with pytest.raises(ValueError, match="Cohort must contain"):
        real_run.main(args)
    assert not report.exists()


def test_plan_preserves_multi_task_alternation_and_catalog_identity():
    tasks = real_run.load_tasks()[:2]
    ledger, schedule = real_run._plan_cohort(tasks, real_run.MODES, 2)
    assert [(t.id, mode, attempt) for t, mode, attempt in schedule] == [
        (tasks[0].id, "harness", 1),
        (tasks[0].id, "minimal", 1),
        (tasks[1].id, "minimal", 1),
        (tasks[1].id, "harness", 1),
        (tasks[0].id, "minimal", 2),
        (tasks[0].id, "harness", 2),
        (tasks[1].id, "harness", 2),
        (tasks[1].id, "minimal", 2),
    ]
    assert len(ledger.cells) == 8
    assert ledger.cells[-1].task_set == tasks[1].task_set


def test_criterion_drift_retains_invalidated_row_and_stops_remaining_cells(monkeypatch, tmp_path):
    import shutil
    from dataclasses import replace

    provider, report, args = setup_cli(monkeypatch, tmp_path)
    original_task = real_run.load_tasks()[0]
    criteria = tmp_path / "criteria"
    shutil.copytree(original_task.acceptance, criteria)
    task = replace(original_task, acceptance=criteria)
    monkeypatch.setattr(real_run, "load_tasks", lambda *args: [task])
    grader = real_run.acceptance_check

    async def drifting_grade(*args, **kwargs):
        result = await grader(*args, **kwargs)
        (criteria / "extra.py").write_text("# changed after grading\n")
        return result

    monkeypatch.setattr(real_run, "acceptance_check", drifting_grade)
    # Authoring controls already have separate coverage; inject drift only at execution.
    monkeypatch.setattr(real_run, "validate_task", AsyncMock())
    with pytest.raises(real_run.AcceptanceDriftError):
        real_run.main(args)
    data = json.loads(report.read_text())
    assert len(data["results"]) == 1 and not data["results"][0]["passed"]
    assert data["results"][0]["termination"] == "acceptance_invalidated"
    assert data["cohort"]["stop_reason"] == "AcceptanceDriftError"
    assert [c["status"] for c in data["cohort"]["cells"]] == ["recorded", "unrun", "unrun", "unrun"]
    assert len(provider.calls) == 1
    provider.aclose.assert_awaited_once()
