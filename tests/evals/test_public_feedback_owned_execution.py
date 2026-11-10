import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import evals.public_feedback_execution as executor
from evals.heldout_comparison import validate_plan
from evals.heldout_operator_rehearsal import ScriptedTransport
from evals.heldout_rehearsal import CannedProvider
from tests.evals import test_public_feedback_execution_provider as provider_tests

captured = provider_tests.captured
template = provider_tests.template
source = provider_tests.source
prepared = provider_tests.prepared


@pytest.fixture
def setup(prepared, monkeypatch):
    admission, _settings, _messages, _kwargs = provider_tests.setup.__wrapped__(
        prepared, monkeypatch
    )
    monkeypatch.setattr(executor, "_runtime_matches", provider_tests.fixtures.gate._runtime_matches)
    events = []
    state = {"cleanup_failure": False}

    def prepare(source, destination, *, expected_manifest_sha256):
        assert source == Path(prepared.image_directory) / "models"
        assert expected_manifest_sha256 == executor.validate_feedback_plan().model_digest
        destination.mkdir()

    monkeypatch.setattr(executor, "prepare_image", prepare)

    @contextmanager
    def owned(image, directory, binary, *, keep_alive_seconds):
        assert image == directory.parent / "image" and image.is_dir()
        assert keep_alive_seconds == 300
        events.append(("open", directory.parent.name))
        try:
            yield SimpleNamespace(check_owned=lambda: None)
        finally:
            events.append(("settle", directory.parent.name))
            if state["cleanup_failure"]:
                # The result must already be durable when owned cleanup starts.
                report = json.loads((admission.directory / "report.json").read_bytes())
                assert len(report["results"]) == 1
                assert report["cohort"]["cells"][0]["status"] == "started"
                raise ValueError("settlement failed")

    monkeypatch.setattr(executor, "owned_backend", owned)
    tasks = {task.id: task for task in executor.load_tasks()}

    def transport(ordinal, directory, fault="none"):
        cell = admission.capture.cells[ordinal - 1]
        return ScriptedTransport(
            CannedProvider(tasks[cell.cell.task_id], validate_plan(), fault), directory
        )

    return admission, events, state, transport


async def test_active_loop_refuses_before_admission_access():
    with pytest.raises(ValueError, match="outside an active event loop"):
        executor.run_feedback_cohort(
            SimpleNamespace(), transport_factory=lambda *a: pytest.fail("transport"), scripted=True
        )


def test_eight_feedback_cells_settle_before_advance_and_cannot_reenter(setup):
    admission, events, _state, transport = setup
    ledger = executor.run_feedback_cohort(admission, transport_factory=transport, scripted=True)
    assert ledger.status == "finished"
    assert events == [
        event
        for i in range(1, 9)
        for event in [("open", f"cell-{i:02d}"), ("settle", f"cell-{i:02d}")]
    ]
    report = json.loads((admission.directory / "report.json").read_bytes())
    assert len(report["results"]) == 8 and all(row["passed"] for row in report["results"])
    assert report["mode"] == "scripted-real-tools-rehearsal"
    plan = executor.validate_feedback_plan()
    assert [row["cell"] for row in report["cohort"]["cells"]] == [
        cell.model_dump() for cell in plan.cells
    ]
    assert [row["attempt"] for row in report["results"]] == [cell.repeat for cell in plan.cells]
    from evals.public_feedback import visible_check_bytes

    for cell in plan.cells:
        workspace = Path(admission.manifest.staged_directory) / f"cell-{cell.ordinal:02d}"
        assert (workspace / "workspace/test_visible.py").read_bytes() == visible_check_bytes(
            cell.arm
        )
        native = admission.directory / f"cell-{cell.ordinal:02d}/native"
        assert len(list(native.glob("*.request.bin"))) == 3
    assert all(cell["status"] == "recorded" for cell in report["cohort"]["cells"])
    with pytest.raises((ValueError, FileExistsError)):
        executor.run_feedback_cohort(admission, transport_factory=transport, scripted=True)
    assert len(events) == 16


@pytest.mark.parametrize("fault", ["transport", "cleanup"])
def test_stopped_cell_keeps_result_before_cleanup_and_leaves_later_unrun(setup, fault):
    admission, events, _state, transport = setup
    _state["cleanup_failure"] = fault == "cleanup"
    with pytest.raises(ValueError):
        executor.run_feedback_cohort(
            admission,
            scripted=True,
            transport_factory=lambda n, d: transport(
                n, d, "transport" if fault == "transport" else "none"
            ),
        )
    report = json.loads((admission.directory / "report.json").read_bytes())
    assert len(report["results"]) == 1
    assert report["cohort"]["status"] == "stopped"
    assert [cell["status"] for cell in report["cohort"]["cells"]] == ["started"] + ["unrun"] * 7
    assert events == [("open", "cell-01"), ("settle", "cell-01")]
    assert not Path(admission.manifest.staged_directory, "cell-02/admitted.json").exists()


def test_image_failure_precedes_backend_transport_and_cell_admission(setup, monkeypatch):
    admission, events, _state, _transport = setup

    def fail(source, destination, **kwargs):
        destination.mkdir()
        raise OSError("clone failed")

    monkeypatch.setattr(executor, "prepare_image", fail)
    with pytest.raises(OSError, match="clone failed"):
        executor.run_feedback_cohort(
            admission, scripted=True, transport_factory=lambda *a: pytest.fail("transport")
        )
    assert not events and (admission.directory / "cell-01/image").is_dir()
    assert not Path(admission.manifest.staged_directory, "cell-01/admitted.json").exists()
    report = json.loads((admission.directory / "report.json").read_bytes())
    assert report["results"] == [] and report["cohort"]["status"] == "stopped"


def test_withdrawn_admission_refuses_before_start_marker(setup):
    admission, events, _state, transport = setup
    (admission.source_manifest.parent / "WITHDRAWN.json").write_text("withdrawn")
    with pytest.raises(ValueError, match="withdrawn"):
        executor.run_feedback_cohort(admission, transport_factory=transport, scripted=True)
    assert not events and not (admission.directory / "cohort-started.json").exists()


def test_rejected_artifacts_with_known_usage_and_clean_settlement_advance(setup):
    import httpx

    admission, events, _state, _transport = setup

    def transport(ordinal, directory):
        return httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "model": "gemma4:12b",
                    "done": True,
                    "message": {"role": "assistant", "content": "scripted unchanged artifact"},
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                },
            )
        )

    result = executor.run_feedback_cohort(admission, transport_factory=transport, scripted=True)
    assert result.status == "finished" and len(events) == 16
    report = json.loads((admission.directory / "report.json").read_bytes())
    assert len(report["results"]) == 8 and not any(row["passed"] for row in report["results"])
    assert all(cell["status"] == "recorded" for cell in report["cohort"]["cells"])
