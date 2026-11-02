import asyncio
import json
from pathlib import Path

import pytest

from evals import heldout_execution_cohort as cohort
from tests.evals.test_execution_cohort import setup as setup
from tests.evals.test_execution_preflight import prepared as prepared


def report(admission):
    return json.loads((admission.directory / "report.json").read_bytes())


def test_control_refusal_retains_all_unrun_and_consumes_start_marker(setup, monkeypatch):
    admission, events, _state, transport = setup

    async def refuse(*args, **kwargs):
        raise ValueError("scripted control refusal")

    monkeypatch.setattr(cohort, "validate_task", refuse)
    with pytest.raises(ValueError, match="control"):
        cohort.run_admitted_cohort(admission, transport_factory=transport, scripted=True)
    data = report(admission)
    assert data["results"] == []
    assert [c["status"] for c in data["cohort"]["cells"]] == ["unrun"] * 16
    assert not events
    original = (admission.directory / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        cohort.run_admitted_cohort(admission, transport_factory=transport, scripted=True)
    assert (admission.directory / "report.json").read_bytes() == original


@pytest.mark.parametrize("fault", ["provider_close", "post_settlement_drift"])
def test_after_result_failures_stop_before_next_workspace(setup, monkeypatch, fault):
    admission, events, _state, transport = setup

    def factory(ordinal, directory):
        inner = transport(ordinal, directory)
        close = inner.aclose

        async def altered_close():
            await close()
            if fault == "provider_close":
                raise OSError("scripted provider close")
            (admission.directory / "manifest.json").write_bytes(b"changed")

        inner.aclose = altered_close
        return inner

    with pytest.raises((OSError, ValueError)):
        cohort.run_admitted_cohort(admission, transport_factory=factory, scripted=True)
    data = report(admission)
    assert len(data["results"]) == 1
    assert [c["status"] for c in data["cohort"]["cells"]] == ["recorded"] + ["unrun"] * 15
    assert events == ["open", "settle"]
    assert not Path(admission.manifest.staged_directory, "cell-02/admitted.json").exists()
    if fault == "provider_close":
        assert data["cohort"]["cleanup_error"] == "OSError"


def test_cancellation_after_submission_is_not_a_retry_slot(setup):
    admission, events, _state, transport = setup

    def factory(ordinal, directory):
        inner = transport(ordinal, directory)

        async def cancel(request):
            raise asyncio.CancelledError("scripted uncertain submission")

        inner.handle_async_request = cancel
        return inner

    with pytest.raises(asyncio.CancelledError):
        cohort.run_admitted_cohort(admission, transport_factory=factory, scripted=True)
    data = report(admission)
    assert data["cohort"]["status"] == "stopped"
    assert [c["status"] for c in data["cohort"]["cells"]] == ["no_result"] + ["unrun"] * 15
    failure = json.loads(
        (admission.directory / "cell-01/native/request-01.failure.json").read_bytes()
    )
    assert failure["delegated"] and failure["error_type"] == "CancelledError"
    assert events == ["open", "settle"]
    assert not Path(admission.manifest.staged_directory, "cell-02/admitted.json").exists()


@pytest.mark.parametrize("stage", ["initial", "row", "finished"])
def test_report_failure_stops_and_preserves_primary_error(setup, monkeypatch, stage):
    admission, events, _state, transport = setup
    original = cohort.write_report
    tripped = []

    def fail(path, model, results, **kwargs):
        ledger = kwargs["cohort"]
        hit = (
            stage == "initial"
            or (stage == "row" and results)
            or (stage == "finished" and ledger.status == "finished")
        )
        if hit:
            tripped.append(True)
            raise OSError("scripted primary report failure")
        if tripped:
            raise RuntimeError("scripted secondary report failure")
        return original(path, model, results, **kwargs)

    monkeypatch.setattr(cohort, "write_report", fail)
    with pytest.raises(OSError, match="primary report failure") as caught:
        cohort.run_admitted_cohort(admission, transport_factory=transport, scripted=True)
    assert any("Final report also failed" in note for note in caught.value.__notes__)
    assert (admission.directory / "cohort-started.json").exists()
    assert len(events) == {"initial": 0, "row": 2, "finished": 32}[stage]
    if stage == "row":
        assert not Path(admission.manifest.staged_directory, "cell-02/admitted.json").exists()


def test_provider_construction_failure_closes_transport_before_workspace_admission(
    setup, monkeypatch
):
    admission, events, _state, transport = setup
    created = []

    def factory(ordinal, directory):
        inner = transport(ordinal, directory)
        created.append(inner)
        return inner

    def refuse(*args, **kwargs):
        raise ValueError("scripted provider construction refusal")

    monkeypatch.setattr(cohort, "build_cell_provider", refuse)
    with pytest.raises(ValueError, match="construction"):
        cohort.run_admitted_cohort(admission, transport_factory=factory, scripted=True)
    assert created[0].closed
    assert events == ["open", "settle"]
    assert not Path(admission.manifest.staged_directory, "cell-01/admitted.json").exists()
    assert [c["status"] for c in report(admission)["cohort"]["cells"]] == ["no_result"] + [
        "unrun"
    ] * 15


async def test_active_event_loop_refuses_before_admission_or_resources():
    with pytest.raises(ValueError, match="outside an active event loop"):
        cohort.run_admitted_cohort(
            None, transport_factory=lambda *args: pytest.fail("transport allocated"), scripted=True
        )
