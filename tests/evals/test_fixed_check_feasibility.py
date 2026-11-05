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


def test_toolset_override_changes_only_toolset():
    plan = validate_plan()
    original = protocol_settings(plan, prompt_profile="direct").model_dump()
    assert protocol_settings(
        plan, toolset="no_plan_check", prompt_profile="direct"
    ).model_dump() == {
        **original,
        "coding_toolset": "no_plan_check",
    }


@pytest.mark.parametrize("toolset", ["no_plan_check", "unknown", "whole_file"])
async def test_nondefault_capture_requires_explicit_scope(tmp_path, toolset):
    destination = tmp_path / "capture"
    with pytest.raises(ValueError):
        await capture_inputs(destination, toolset=toolset)
    assert not destination.exists()


def test_fixed_check_manifest_requires_single_cell(selected):
    raw = json.loads(selected[0].read_bytes())
    assert ExecutionManifest.model_validate(raw).coding_toolset == "full"
    raw["coding_toolset"] = "no_plan_check"
    with pytest.raises(ValueError, match="explicit single-cell"):
        ExecutionManifest.model_validate(raw)
    raw.update(
        residency_plan_sha256="0" * 64,
        residency_feasibility_ordinal=2,
        coding_prompt_profile="direct",
    )
    assert ExecutionManifest.model_validate(raw).coding_toolset == "no_plan_check"


@pytest.mark.parametrize("prepared", ["fixed_check"], indirect=True)
def test_toolset_mismatch_refuses_before_allocation(prepared):
    assert preflight(prepared).settings_sha256 is not None
    with pytest.raises(ValueError, match="differ from capture"):
        preflight(prepared.model_copy(update={"coding_toolset": "full"}))
    assert not Path(prepared.output_directory).exists()


@pytest.mark.parametrize("prepared", ["fixed_check"], indirect=True)
def test_fixed_check_shared_first_input_and_single_cell_execution(setup):
    admission, events, _state, transport = setup
    ledger = run_residency_feasibility(admission, transport_factory=transport, scripted=True)
    assert ledger.status == "finished" and events == [("open", 300), ("settle", 300)]
    report = json.loads((admission.directory / "report.json").read_bytes())
    row = report["results"][0]
    assert row["passed"] and row["runtime_config"]["coding_toolset"] == "no_plan_check"
    assert all(spec["name"] != "emit_plan" for spec in row["runtime_config"]["tool_specs"])
    assert all(spec["name"] != "run_command" for spec in row["runtime_config"]["tool_specs"])
    checks = [call for call in row["tool_trace"] if call["name"] == "run_project_check"]
    assert (
        len(checks) == 1 and checks[0]["arguments"] == {} and checks[0]["result"]["success"] is True
    )
    assert row["verification_status"] == "passed"
    assert row["prompt_version"] == "coding-v1-direct-v1-fixed-check-v1"
    assert (admission.directory / "cell-02/native/request-01.request.bin").read_bytes() == Path(
        admission.manifest.capture_directory, "02-wire.json"
    ).read_bytes()


async def test_fixed_check_capture_requires_direct_before_allocation(tmp_path):
    destination = tmp_path / "capture"
    with pytest.raises(ValueError, match="direct"):
        await capture_inputs(
            destination, residency_schedule=True, prepared=True, toolset="no_plan_check"
        )
    assert not destination.exists()
