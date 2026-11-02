import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import heldout_execution_cohort as cohort
from evals.heldout_comparison import validate_plan
from evals.heldout_guard import REGISTRATIONS
from evals.heldout_inputs import _save
from evals.heldout_operator_rehearsal import ScriptedTransport
from evals.heldout_rehearsal import CannedProvider
from evals.heldout_staging import verify_capture
from evals.real_run import MODES, _plan_cohort, load_tasks
from tests.evals.test_execution_preflight import prepared as prepared


@pytest.fixture
def setup(prepared, monkeypatch):
    manifest = prepared
    output = Path(manifest.output_directory)
    output.mkdir()
    raw = manifest.model_dump_json().encode()
    _save(output / "manifest.json", raw)
    index = verify_capture(
        Path(manifest.capture_directory),
        expected_index_sha256=manifest.capture_index_sha256,
        original_directory=Path(manifest.staged_directory),
    )
    admission = SimpleNamespace(
        directory=output,
        manifest=manifest,
        capture=index,
        approval=SimpleNamespace(manifest_sha256=hashlib.sha256(raw).hexdigest()),
    )
    events = []
    state = {"cleanup_failure": False}

    @contextmanager
    def owned(image, directory, binary):
        events.append("open")
        try:
            yield SimpleNamespace(check_owned=lambda: None)
        finally:
            events.append("settle")
            if state["cleanup_failure"]:
                raise ValueError("scripted settlement failure")

    monkeypatch.setattr(cohort, "owned_backend", owned)
    tasks = load_tasks(REGISTRATIONS[0].root / "tasks.yaml")
    _, schedule = _plan_cohort(tasks, MODES, 2)

    def transport(ordinal, directory, fault="none"):
        return ScriptedTransport(
            CannedProvider(schedule[ordinal - 1][0], validate_plan(), fault), directory
        )

    return admission, events, state, transport


def test_sixteen_rows_settle_serially_and_cannot_restart(setup):
    admission, events, _state, transport = setup
    result = cohort.run_admitted_cohort(admission, transport_factory=transport, scripted=True)
    assert result.status == "finished"
    assert events == ["open", "settle"] * 16
    data = json.loads((admission.directory / "report.json").read_bytes())
    assert len(data["results"]) == 16 and all(c.status == "recorded" for c in result.cells)
    assert all(row["passed"] and row["completion_status"] == "completed" for row in data["results"])
    assert data["mode"] == "scripted-real-tools-rehearsal"
    with pytest.raises((ValueError, FileExistsError)):
        cohort.run_admitted_cohort(admission, transport_factory=transport, scripted=True)
    assert len(events) == 32


@pytest.mark.parametrize("fault", ["transport", "cleanup"])
def test_failure_preserves_first_row_and_fifteen_unrun_cells(setup, fault):
    admission, events, state, transport = setup
    state["cleanup_failure"] = fault == "cleanup"

    def factory(ordinal, directory):
        return transport(ordinal, directory, "transport" if fault == "transport" else "none")

    with pytest.raises(ValueError):
        cohort.run_admitted_cohort(admission, transport_factory=factory, scripted=True)
    data = json.loads((admission.directory / "report.json").read_bytes())
    assert len(data["results"]) == 1
    assert [row["status"] for row in data["cohort"]["cells"]] == ["recorded"] + ["unrun"] * 15
    assert data["cohort"]["status"] == "stopped"
    assert events == ["open", "settle"]
    assert not Path(admission.manifest.staged_directory, "cell-02/admitted.json").exists()
