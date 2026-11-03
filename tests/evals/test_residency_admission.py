from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.heldout_execution_cohort import run_admitted_cohort
from evals.heldout_execution_preflight import preflight
from evals.heldout_input_guard import load_pins
from tests.evals.test_execution_preflight import prepared as prepared


@pytest.mark.parametrize("prepared", [True], indirect=True)
def test_residency_preflight_and_pin_loading_require_explicit_selection(prepared):
    index = preflight(prepared)
    assert [cell.keep_alive_seconds for cell in index.cells] == [0, 300] * 4 + [300, 0] * 4
    kwargs = {
        "original_directory": Path(prepared.staged_directory),
        "expected_index_sha256": prepared.capture_index_sha256,
        "ordinal": 1,
    }
    pins = load_pins(Path(prepared.capture_directory), residency=True, **kwargs)
    assert pins.wire and pins.normalized
    with pytest.raises(ValueError):
        load_pins(Path(prepared.capture_directory), **kwargs)
    assert not Path(prepared.output_directory).exists()


@pytest.mark.parametrize("prepared", [True], indirect=True)
@pytest.mark.parametrize("fault", ["missing", "hash", "override"])
def test_residency_selection_mismatch_refuses_before_output(prepared, fault):
    updates = {
        "missing": {"residency_plan_sha256": None},
        "hash": {"residency_plan_sha256": "0" * 64},
        "override": {"keep_alive_seconds": 300},
    }[fault]
    with pytest.raises(ValueError):
        preflight(prepared.model_copy(update=updates))
    assert not Path(prepared.output_directory).exists()


@pytest.mark.parametrize("prepared", [True], indirect=True)
def test_legacy_cohort_never_runs_a_residency_admission(prepared):
    admission = SimpleNamespace(manifest=prepared)
    with pytest.raises(ValueError, match="dedicated executor"):
        run_admitted_cohort(
            admission,
            transport_factory=lambda *a: pytest.fail("transport constructed"),
            scripted=True,
        )
    assert not Path(prepared.output_directory).exists()
