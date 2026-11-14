import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import evals.fresh_visible_execution as executor
from evals.heldout_comparison import validate_plan
from evals.heldout_inputs import _save
from evals.heldout_operator_rehearsal import ScriptedTransport
from evals.heldout_rehearsal import CannedProvider
from tests.evals import test_fresh_visible_execution_provider as provider_tests

captured = provider_tests.captured
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
        assert expected_manifest_sha256 == executor.validate_public_plan().model_digest
        destination.mkdir()
        _save(destination / "prepared-image.json", b"{}")

    monkeypatch.setattr(executor, "prepare_image", prepare)

    @contextmanager
    def owned(image, directory, binary, *, keep_alive_seconds):
        assert image == directory.parent / "image" and image.is_dir()
        assert keep_alive_seconds == 300
        directory.mkdir()
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
            cleanup = {
                "passed": True,
                "settled": True,
                "forced": False,
                "returncode": 0,
                "errors": {},
            }
            _save(directory / "cleanup.json", json.dumps(cleanup).encode())
            _save(
                directory / "session.json",
                json.dumps(
                    {
                        "format": "heldout-owned-backend-session-v1",
                        "inference_authorized": False,
                        "cleanup": cleanup,
                    }
                ).encode(),
            )

    monkeypatch.setattr(executor, "owned_backend", owned)
    tasks = {task.id: task for task in executor.load_tasks(executor.ROOT / "tasks.yaml")}

    def transport(ordinal, directory, fault="none"):
        cell = admission.capture.cells[ordinal - 1]
        return ScriptedTransport(
            CannedProvider(tasks[cell.cell.task_id], validate_plan(), fault), directory
        )

    return admission, events, state, transport


async def test_active_loop_refuses_before_admission_access():
    with pytest.raises(ValueError, match="outside an active event loop"):
        executor.run_fresh_cohort(
            SimpleNamespace(), transport_factory=lambda *a: pytest.fail("transport"), scripted=True
        )


def test_twelve_fresh_cells_settle_before_advance_and_cannot_reenter(setup):
    admission, events, _state, transport = setup
    ledger = executor.run_fresh_cohort(admission, transport_factory=transport, scripted=True)
    assert ledger.status == "finished"
    assert events == [
        event
        for i in range(1, 13)
        for event in [("open", f"cell-{i:02d}"), ("settle", f"cell-{i:02d}")]
    ]
    report = json.loads((admission.directory / "report.json").read_bytes())
    assert len(report["results"]) == 12 and all(row["passed"] for row in report["results"])
    assert report["mode"] == "scripted-real-tools-rehearsal"
    cells = [executor.selected_cell(i)[0] for i in range(1, 13)]
    assert [row["cell"] for row in report["cohort"]["cells"]] == [
        cell.model_dump() for cell in cells
    ]
    assert [row["attempt"] for row in report["results"]] == [cell.repeat for cell in cells]

    for cell in cells:
        workspace = Path(admission.manifest.staged_directory) / f"cell-{cell.ordinal:02d}"
        assert (workspace / "workspace/test_visible.py").read_bytes() == executor.selected_cell(
            cell.ordinal
        )[2]
        native = admission.directory / f"cell-{cell.ordinal:02d}/native"
        assert len(list(native.glob("*.request.bin"))) == 3
    assert all(cell["status"] == "recorded" for cell in report["cohort"]["cells"])
    with pytest.raises((ValueError, FileExistsError)):
        executor.run_fresh_cohort(admission, transport_factory=transport, scripted=True)
    assert len(events) == 24


@pytest.mark.parametrize("fault", ["transport", "cleanup"])
def test_stopped_cell_keeps_result_before_cleanup_and_leaves_later_unrun(setup, fault):
    admission, events, _state, transport = setup
    _state["cleanup_failure"] = fault == "cleanup"
    with pytest.raises(ValueError):
        executor.run_fresh_cohort(
            admission,
            scripted=True,
            transport_factory=lambda n, d: transport(
                n, d, "transport" if fault == "transport" else "none"
            ),
        )
    report = json.loads((admission.directory / "report.json").read_bytes())
    assert len(report["results"]) == 1
    assert report["cohort"]["status"] == "stopped"
    assert [cell["status"] for cell in report["cohort"]["cells"]] == ["started"] + ["unrun"] * 11
    assert events == [("open", "cell-01"), ("settle", "cell-01")]
    assert not Path(admission.manifest.staged_directory, "cell-02/admitted.json").exists()


def test_image_failure_precedes_backend_transport_and_cell_admission(setup, monkeypatch):
    admission, events, _state, _transport = setup

    def fail(source, destination, **kwargs):
        destination.mkdir()
        raise OSError("clone failed")

    monkeypatch.setattr(executor, "prepare_image", fail)
    with pytest.raises(OSError, match="clone failed"):
        executor.run_fresh_cohort(
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
        executor.run_fresh_cohort(admission, transport_factory=transport, scripted=True)
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

    result = executor.run_fresh_cohort(admission, transport_factory=transport, scripted=True)
    assert result.status == "finished" and len(events) == 24
    report = json.loads((admission.directory / "report.json").read_bytes())
    assert len(report["results"]) == 12 and not any(row["passed"] for row in report["results"])
    assert all(cell["status"] == "recorded" for cell in report["cohort"]["cells"])


@pytest.mark.parametrize("fault", ["forced", "errors", "agreement", "mode"])
def test_settlement_requires_private_agreeing_unforced_evidence(tmp_path, fault):
    output = tmp_path / "cell"
    (output / "image").mkdir(parents=True)
    (output / "backend").mkdir()
    cleanup = {"passed": True, "settled": True, "forced": False, "returncode": 0, "errors": {}}
    session = {
        "format": "heldout-owned-backend-session-v1",
        "inference_authorized": False,
        "cleanup": dict(cleanup),
    }
    if fault == "forced":
        cleanup["forced"] = True
    elif fault == "errors":
        cleanup["errors"] = {"settlement_gate": "ValueError"}
    elif fault == "agreement":
        session["cleanup"] = {}
    _save(output / "image/prepared-image.json", b"{}")
    _save(output / "backend/cleanup.json", json.dumps(cleanup).encode())
    _save(output / "backend/session.json", json.dumps(session).encode())
    if fault == "mode":
        (output / "backend/session.json").chmod(0o644)
    with pytest.raises(ValueError):
        executor._settlement_files(output)
