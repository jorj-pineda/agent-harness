import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from evals import heldout_inputs as inputs
from evals import heldout_operator_rehearsal as operator
from evals.heldout_staging import stage_capture


@pytest.fixture
async def staged(tmp_path, monkeypatch):
    runtime = {"commit": "a" * 40, "dirty": False}
    monkeypatch.setattr(inputs, "_harness_revision", lambda: runtime)
    monkeypatch.setattr(operator, "_harness_revision", lambda: runtime)
    environment = {
        "PATH": os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin"]),
        "LC_ALL": "C",
    }
    source = tmp_path / "inputs"
    index = await inputs.capture_inputs(source, prepared=True, command_env=environment)
    digest = hashlib.sha256((source / "index.json").read_bytes()).hexdigest()
    stage_capture(source, tmp_path / "archive", expected_index_sha256=digest)
    return source, tmp_path / "archive/captured", tmp_path / "operator", digest, environment, index


async def test_selected_environment_matches_capture_through_sixteen_mock_cells(staged, monkeypatch):
    source, capture, output, digest, environment, index = staged
    monkeypatch.setenv("PATH", "/ambient-unavailable")
    monkeypatch.setenv("LANG", "ambient-sentinel")
    ledger = await operator.rehearse(
        source, capture, output, expected_index_sha256=digest, command_env=environment
    )
    assert ledger.status == "finished"
    report = json.loads((output / "report.json").read_bytes())
    assert len(report["results"]) == 16 and all(row["passed"] for row in report["results"])
    assert all(
        row["runtime_config"]["command_environment"]["policy"] == "replacement-v1"
        for row in report["results"]
    )
    evidence = json.loads((output / "operator.json").read_bytes())
    assert evidence["command_environment_sha256"] == index["command_environment_sha256"]
    assert evidence["forwarded_requests"] == 0 and not evidence["inference_ran"]
    for n in range(1, 17):
        assert (output / f"cell-{n:02d}/request-01.json").read_bytes() == (
            capture / f"{n:02d}-wire.json"
        ).read_bytes()


@pytest.mark.parametrize("fault", ["missing", "different", "file"])
async def test_environment_mismatch_refuses_before_admission_or_allocation(
    staged, monkeypatch, fault
):
    source, capture, output, digest, environment, _ = staged
    selected = (
        None
        if fault == "missing"
        else {**environment, "LC_ALL": "POSIX"}
        if fault == "different"
        else environment
    )
    if fault == "file":
        (capture / "command-environment.json").chmod(0o644)
    monkeypatch.setattr(operator, "ScriptedTransport", lambda *args: pytest.fail("allocated"))
    with pytest.raises(ValueError):
        await operator.rehearse(
            source, capture, output, expected_index_sha256=digest, command_env=selected
        )
    assert not output.exists() and not (source / "cell-01/admitted.json").exists()


@pytest.mark.parametrize("fault", ["mode", "hash", "flags", "unknown", "size"])
def test_pin_file_validation_refuses_changed_or_ineffective_values(tmp_path, fault):
    from evals.heldout_command_environment import environment_bytes, load_environment_pin
    from evals.heldout_inputs import _save

    path = tmp_path / "environment.json"
    digest = _save(path, environment_bytes({"PATH": "/usr/bin", "LC_ALL": "C"}))
    if fault == "mode":
        path.chmod(0o644)
    elif fault == "hash":
        path.write_bytes(b"changed")
    elif fault == "size":
        path.write_bytes(b"x" * 64001)
    else:
        data = json.loads(path.read_bytes())
        data["values"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD" if fault == "flags" else "UNKNOWN"] = "0"
        path.write_text(json.dumps(data, sort_keys=True))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        load_environment_pin(path, digest)
