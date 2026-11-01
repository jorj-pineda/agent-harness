import hashlib
import sys
from pathlib import Path

import pytest

from evals import heldout_execution_preflight as gate
from evals import heldout_inputs as inputs
from evals import heldout_operator_rehearsal as operator
from evals.heldout_comparison import PLAN_PATH
from evals.heldout_dependency_files import record_files
from evals.heldout_execution_manifest import ExecutionManifest
from evals.heldout_runtime_inventory import record_runtime
from evals.heldout_staging import stage_capture
from evals.real_run import _task_command_environment


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
async def prepared(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    runtime = {"commit": "a" * 40, "dirty": False}
    monkeypatch.setattr(inputs, "_harness_revision", lambda: runtime)
    monkeypatch.setattr(operator, "_harness_revision", lambda: runtime)
    inventory = root / "runtime.json"
    record_runtime({"python": Path(sys.executable)}, inventory)
    dependency = root / "dependency.py"
    dependency.write_text("value = 1\n")
    dependencies = root / "dependencies.json"
    record_files([dependency], dependencies)
    environment = _task_command_environment({"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
    source = root / "inputs"
    await inputs.capture_inputs(
        source,
        prepared=True,
        command_env=environment,
        runtime_inventory=inventory,
        runtime_sha256=digest(inventory),
        dependency_manifest=dependencies,
        dependency_sha256=digest(dependencies),
    )
    selected = digest(source / "index.json")
    stage_capture(source, root / "archive", expected_index_sha256=selected)
    image = root / "image"
    image.mkdir()
    inputs._save(image / "prepared-image.json", b"scripted-model-image-receipt")
    # Image verification has its own native/fixture tests. This integration uses
    # real capture/staging/runtime/dependency checks and never allocates a model.
    monkeypatch.setattr(gate, "_prepared_models", lambda path: (path / "models", {}))
    binary = root / "ollama"
    binary.write_bytes(b"scripted-executable-never-run")
    binary.chmod(0o700)
    return ExecutionManifest(
        format="heldout-execution-manifest-v1",
        runtime_commit=runtime["commit"],
        protocol_sha256=digest(PLAN_PATH),
        capture_index_sha256=selected,
        runtime_inventory_sha256=digest(inventory),
        dependency_files_sha256=digest(dependencies),
        image_receipt_sha256=digest(image / "prepared-image.json"),
        binary_sha256=digest(binary),
        staged_directory=str(source),
        capture_directory=str(root / "archive/captured"),
        output_directory=str(root / "output"),
        image_directory=str(image),
        ollama_binary=str(binary),
        command_environment=environment,
        inference_authorized=False,
    )


def test_real_capture_pins_and_sixteen_fresh_workspaces_pass_read_only(prepared):
    index = gate.preflight(prepared)
    assert len(index.cells) == 16
    assert not Path(prepared.output_directory).exists()
    assert not list(Path(prepared.staged_directory).glob("*/admitted.json"))


@pytest.mark.parametrize(
    "fault",
    ["protocol", "runtime", "dependency", "environment", "receipt", "binary", "admitted"],
)
def test_semantic_drift_refuses_without_output_or_admission(prepared, fault):
    updates = {}
    if fault in ("protocol", "runtime", "dependency"):
        key = {
            "protocol": "protocol_sha256",
            "runtime": "runtime_inventory_sha256",
            "dependency": "dependency_files_sha256",
        }[fault]
        updates[key] = "0" * 64
    elif fault == "environment":
        updates["command_environment"] = {**prepared.command_environment, "LC_ALL": "POSIX"}
    elif fault == "receipt":
        Path(prepared.image_directory, "prepared-image.json").write_bytes(b"changed")
    elif fault == "binary":
        Path(prepared.ollama_binary).write_bytes(b"changed")
    else:
        Path(prepared.staged_directory, "cell-16/admitted.json").write_bytes(b"occupied")
    with pytest.raises(ValueError):
        gate.preflight(prepared.model_copy(update=updates))
    assert not Path(prepared.output_directory).exists()
    assert not Path(prepared.staged_directory, "cell-01/admitted.json").exists()


@pytest.mark.parametrize("fault", ["occupied", "overlap", "alias", "binary_link"])
def test_path_refusal_precedes_artifact_inspection(prepared, monkeypatch, fault):
    updates = {}
    if fault == "occupied":
        Path(prepared.output_directory).mkdir()
    elif fault == "overlap":
        updates["output_directory"] = prepared.capture_directory
    elif fault == "alias":
        alias = Path(prepared.output_directory).parent / "alias"
        alias.symlink_to(Path(prepared.staged_directory), target_is_directory=True)
        updates["output_directory"] = str(alias / "output")
    else:
        alias = Path(prepared.output_directory).parent / "binary-link"
        alias.symlink_to(prepared.ollama_binary)
        updates["ollama_binary"] = str(alias)
    monkeypatch.setattr(gate, "verify_capture", lambda *a, **k: pytest.fail("inspected artifacts"))
    with pytest.raises(ValueError):
        gate.preflight(prepared.model_copy(update=updates))
    assert not Path(prepared.staged_directory, "cell-01/admitted.json").exists()


def test_runtime_change_during_preflight_refuses(prepared, monkeypatch):
    observations = iter([{"commit": "a" * 40, "dirty": False}, {"commit": "b" * 40}])
    monkeypatch.setattr(operator, "_harness_revision", lambda: next(observations))
    with pytest.raises(ValueError, match="runtime"):
        gate.preflight(prepared)
    assert not Path(prepared.output_directory).exists()
