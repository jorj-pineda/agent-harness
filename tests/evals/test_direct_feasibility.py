import json
from pathlib import Path

import pytest

from evals.heldout_comparison import validate_plan
from evals.heldout_execution_manifest import ExecutionManifest
from evals.heldout_execution_preflight import preflight
from evals.heldout_inputs import capture_inputs
from evals.heldout_settings import protocol_settings
from evals.residency_execution import run_residency_feasibility
from tests.evals.test_execution_manifest import selected as selected
from tests.evals.test_execution_preflight import prepared as prepared
from tests.evals.test_residency_execution import setup as setup


def test_settings_override_changes_only_prompt_selection():
    plan = validate_plan()
    original = protocol_settings(plan).model_dump()
    direct = protocol_settings(plan, prompt_profile="direct").model_dump()
    assert direct == {**original, "coding_prompt_profile": "direct"}


@pytest.mark.parametrize("profile", ["direct", "unknown", "legacy"])
async def test_nondefault_capture_refuses_without_explicit_scope_before_allocation(
    tmp_path, profile
):
    destination = tmp_path / "capture"
    with pytest.raises(ValueError):
        await capture_inputs(destination, prompt_profile=profile)
    assert not destination.exists()


def test_direct_manifest_cannot_select_full_cohort(selected):
    raw = json.loads(selected[0].read_bytes())
    raw["coding_prompt_profile"] = "direct"
    with pytest.raises(ValueError, match="explicit single-cell"):
        ExecutionManifest.model_validate(raw)
    raw.update(residency_plan_sha256="0" * 64, residency_feasibility_ordinal=2)
    assert ExecutionManifest.model_validate(raw).coding_prompt_profile == "direct"


@pytest.mark.parametrize("prepared", ["direct"], indirect=True)
def test_preflight_binds_manifest_profile_to_captured_settings(prepared):
    assert preflight(prepared).settings_sha256 is not None
    with pytest.raises(ValueError, match="differ from capture"):
        preflight(prepared.model_copy(update={"coding_prompt_profile": "compact"}))
    assert not Path(prepared.output_directory).exists()


@pytest.mark.parametrize("prepared", ["direct"], indirect=True)
def test_direct_first_input_and_runtime_are_shared_in_single_cell_execution(setup):
    admission, events, _state, transport = setup
    ledger = run_residency_feasibility(admission, transport_factory=transport, scripted=True)
    assert ledger.status == "finished" and events == [("open", 300), ("settle", 300)]
    report = json.loads((admission.directory / "report.json").read_bytes())
    row = report["results"][0]
    assert row["passed"] and row["prompt_version"] == "coding-v1-direct-v1"
    assert row["runtime_config"]["coding_prompt_profile"] == "direct"
    assert (admission.directory / "cell-02/native/request-01.request.bin").read_bytes() == Path(
        admission.manifest.capture_directory, "02-wire.json"
    ).read_bytes()
