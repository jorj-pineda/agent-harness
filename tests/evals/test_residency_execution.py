import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import residency_execution as executor
from evals.heldout_comparison import validate_plan
from evals.heldout_inputs import _save
from evals.heldout_operator_rehearsal import ScriptedTransport
from evals.heldout_rehearsal import CannedProvider
from evals.heldout_staging import verify_residency_capture
from tests.evals.test_execution_preflight import prepared as prepared


def test_real_execution_requires_provenance_before_transport_or_resources():
    with pytest.raises(ValueError, match="source provenance"):
        executor.run_residency_cohort(
            SimpleNamespace(),
            transport_factory=lambda *a: pytest.fail("transport"),
            scripted=False,
        )


async def test_active_event_loop_refuses_before_admission_access():
    with pytest.raises(ValueError, match="outside an active event loop"):
        executor.run_residency_cohort(
            SimpleNamespace(),
            transport_factory=lambda *a: pytest.fail("transport"),
            scripted=True,
        )


@pytest.fixture
def setup(prepared, monkeypatch):
    output = Path(prepared.output_directory)
    output.mkdir()
    raw = prepared.model_dump_json().encode()
    _save(output / "manifest.json", raw)
    admission = SimpleNamespace(
        manifest=prepared,
        directory=output,
        capture=verify_residency_capture(
            Path(prepared.capture_directory),
            expected_index_sha256=prepared.capture_index_sha256,
            original_directory=Path(prepared.staged_directory),
        ),
        approval=SimpleNamespace(manifest_sha256=hashlib.sha256(raw).hexdigest()),
    )
    events = []
    state = {"cleanup_failure": False}

    def prepare(source, destination, *, expected_manifest_sha256):
        assert source == Path(prepared.image_directory) / "models"
        assert expected_manifest_sha256 == validate_plan().model_digest
        destination.mkdir()

    monkeypatch.setattr(executor, "prepare_image", prepare)

    @contextmanager
    def owned(image, directory, binary, *, keep_alive_seconds):
        assert image == directory.parent / "image"
        assert image.is_dir()
        events.append(("open", keep_alive_seconds))
        try:
            yield SimpleNamespace(check_owned=lambda: None)
        finally:
            events.append(("settle", keep_alive_seconds))
            if state["cleanup_failure"]:
                raise ValueError("settlement failed")

    monkeypatch.setattr(executor, "owned_backend", owned)
    tasks = {
        task.id: task for task in executor.load_tasks(executor.REGISTRATIONS[0].root / "tasks.yaml")
    }

    def transport(ordinal, directory, fault="none"):
        cell = admission.capture.cells[ordinal - 1]
        return ScriptedTransport(
            CannedProvider(tasks[cell.task_id], validate_plan(), fault), directory
        )

    return admission, events, state, transport


@pytest.mark.parametrize("prepared", [True], indirect=True)
def test_sixteen_owned_cells_use_exact_arms_and_settle_before_advance(setup):
    admission, events, _state, transport = setup
    ledger = executor.run_residency_cohort(admission, transport_factory=transport, scripted=True)
    assert ledger.status == "finished"
    arms = [0, 300] * 4 + [300, 0] * 4
    assert events == [event for arm in arms for event in [("open", arm), ("settle", arm)]]
    report = json.loads((admission.directory / "report.json").read_bytes())
    assert len(report["results"]) == 16
    assert all(row["passed"] for row in report["results"])
    assert all(cell["status"] == "recorded" for cell in report["cohort"]["cells"])
    with pytest.raises((ValueError, FileExistsError)):
        executor.run_residency_cohort(admission, transport_factory=transport, scripted=True)
    assert len(events) == 32


@pytest.mark.parametrize("prepared", [True], indirect=True)
@pytest.mark.parametrize("fault", ["transport", "cleanup"])
def test_stopped_cell_keeps_result_before_cleanup_and_leaves_both_arms_unrun(setup, fault):
    admission, events, state, transport = setup
    state["cleanup_failure"] = fault == "cleanup"
    with pytest.raises(ValueError):
        executor.run_residency_cohort(
            admission,
            transport_factory=lambda ordinal, directory: transport(
                ordinal, directory, "transport" if fault == "transport" else "none"
            ),
            scripted=True,
        )
    report = json.loads((admission.directory / "report.json").read_bytes())
    assert len(report["results"]) == 1
    assert report["cohort"]["status"] == "stopped"
    assert [cell["status"] for cell in report["cohort"]["cells"]] == ["started"] + ["unrun"] * 15
    assert events == [("open", 0), ("settle", 0)]
    assert not Path(admission.manifest.staged_directory, "cell-02/admitted.json").exists()


@pytest.mark.parametrize("prepared", [True], indirect=True)
def test_cell_image_failure_stops_before_backend_transport_or_workspace_admission(
    setup, monkeypatch
):
    admission, events, _state, _transport = setup

    def fail(source, destination, **kwargs):
        destination.mkdir()
        raise OSError("clone failed")

    monkeypatch.setattr(executor, "prepare_image", fail)
    with pytest.raises(OSError, match="clone failed"):
        executor.run_residency_cohort(
            admission,
            transport_factory=lambda *a: pytest.fail("transport constructed"),
            scripted=True,
        )
    assert events == []
    assert (admission.directory / "cell-01/image").is_dir()
    assert not Path(admission.manifest.staged_directory, "cell-01/admitted.json").exists()
    report = json.loads((admission.directory / "report.json").read_bytes())
    assert report["results"] == []
    assert report["cohort"]["status"] == "stopped"
    assert [cell["status"] for cell in report["cohort"]["cells"]] == ["started"] + ["unrun"] * 15
