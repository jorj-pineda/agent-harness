import hashlib
import json

import pytest

from evals import heldout_inputs as inputs
from evals import heldout_operator_rehearsal as operator
from evals.heldout_dependency_files import record_files
from evals.heldout_staging import stage_capture, verify_capture


@pytest.fixture
async def staged(tmp_path, monkeypatch):
    runtime = {"commit": "a" * 40, "dirty": False}
    monkeypatch.setattr(inputs, "_harness_revision", lambda: runtime)
    monkeypatch.setattr(operator, "_harness_revision", lambda: runtime)
    source_file = tmp_path / "dependency.py"
    source_file.write_text("value = 1\n")
    report = tmp_path / "dependencies.json"
    record_files([source_file], report)
    selected = hashlib.sha256(report.read_bytes()).hexdigest()
    source = tmp_path / "inputs"
    await inputs.capture_inputs(
        source, prepared=True, dependency_manifest=report, dependency_sha256=selected
    )
    digest = hashlib.sha256((source / "index.json").read_bytes()).hexdigest()
    stage_capture(source, tmp_path / "archive", expected_index_sha256=digest)
    return source, tmp_path / "archive/captured", tmp_path / "output", digest, selected


async def test_dependency_selected_before_all_sixteen_cells_and_preserved(staged):
    source, capture, output, digest, selected = staged
    original = (capture / "dependency-files.json").read_bytes()
    result = await operator.rehearse(
        source, capture, output, expected_index_sha256=digest, dependency_sha256=selected
    )
    assert result.status == "finished"
    data = json.loads((output / "report.json").read_bytes())
    assert len(data["results"]) == 16 and all(row["passed"] for row in data["results"])
    assert (capture / "dependency-files.json").read_bytes() == original
    assert (
        json.loads((output / "operator.json").read_bytes())["dependency_files_sha256"] == selected
    )


@pytest.mark.parametrize("fault", ["missing", "drift", "file"])
async def test_dependency_gate_refuses_before_provider_and_admission(staged, monkeypatch, fault):
    source, capture, output, digest, selected = staged
    if fault == "missing":
        selected = None
    elif fault == "drift":
        source_file = capture.parent.parent / "dependency.py"
        source_file.write_text("value = 2\n")
        # Archival verification must not depend on the current machine runtime.
        verify_capture(capture, expected_index_sha256=digest, original_directory=source)
    else:
        (capture / "dependency-files.json").chmod(0o644)
    monkeypatch.setattr(operator, "ScriptedTransport", lambda *a: pytest.fail("allocated"))
    with pytest.raises(ValueError):
        await operator.rehearse(
            source, capture, output, expected_index_sha256=digest, dependency_sha256=selected
        )
    assert not output.exists() and not (source / "cell-01/admitted.json").exists()


async def test_incomplete_dependency_selection_refuses_before_capture_creation(tmp_path):
    directory = tmp_path / "inputs"
    with pytest.raises(ValueError, match="together"):
        await inputs.capture_inputs(directory, dependency_sha256="0" * 64)
    assert not directory.exists()
