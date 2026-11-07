import hashlib
import json
import sys
from pathlib import Path

import httpx
import pytest

import evals.public_inputs as capture
from evals.heldout_dependency_files import record_files
from evals.heldout_runtime_inventory import record_runtime


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def selected(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "_harness_revision", lambda: {"commit": "a" * 40, "dirty": False})
    runtime = tmp_path / "runtime.json"
    record_runtime({"python": Path(sys.executable)}, runtime)
    dependency = tmp_path / "dependency.py"
    dependency.write_text("value = 1\n")
    report = tmp_path / "dependencies.json"
    record_files([dependency], report)
    return {
        "command_env": {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "LC_ALL": "C"},
        "runtime_inventory": runtime,
        "runtime_sha256": sha(runtime),
        "dependency_manifest": report,
        "dependency_sha256": sha(report),
    }


async def test_six_production_inputs_are_mocked_private_and_pinned(tmp_path, selected, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("real HTTP transport constructed")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "__init__", forbidden)
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:9191")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:9191")
    directory = tmp_path / "capture"
    index = await capture.capture_public_inputs(directory, **selected)
    assert len(index.cells) == 6 and index.generation_forwarded == 0
    assert index.inference_authorized is False
    assert index.runtime.commit == "a" * 40 and index.runtime.dirty is False
    assert directory.stat().st_mode & 0o777 == 0o700
    for row in index.cells:
        for name, digest in row.files.items():
            assert sha(directory / name) == digest
            assert (directory / name).stat().st_mode & 0o777 == 0o600
        wire = json.loads((directory / f"{row.ordinal:02d}-wire.json").read_bytes())
        normalized = json.loads((directory / f"{row.ordinal:02d}-normalized.json").read_bytes())
        assert wire["model"] == "gemma4:12b" and wire["stream"] is False
        assert wire["options"]["num_ctx"] == 16384 and wire["think"] is False
        names = {spec["name"] for spec in normalized["tools"]}
        assert {"write_file", "run_project_check"} <= names
        assert not {"replace_text", "run_command", "emit_plan"} & names
        assert row.workspace_root == str(directory / f"cell-{row.ordinal:02d}/workspace")
    assert (directory / "index.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        await capture.capture_public_inputs(directory, **selected)


@pytest.mark.parametrize("defect", ["dirty", "dependency", "runtime-mode", "environment"])
async def test_preconditions_refuse_before_allocation(tmp_path, selected, monkeypatch, defect):
    if defect == "dirty":
        monkeypatch.setattr(
            capture, "_harness_revision", lambda: {"commit": "a" * 40, "dirty": True}
        )
    elif defect == "dependency":
        (tmp_path / "dependency.py").write_text("changed\n")
    elif defect == "runtime-mode":
        selected["runtime_inventory"].chmod(0o644)
    else:
        selected["command_env"] = {"PATH": "relative"}
    directory = tmp_path / "capture"
    with pytest.raises(ValueError):
        await capture.capture_public_inputs(directory, **selected)
    assert not directory.exists()


async def test_midrun_drift_keeps_first_input_and_consumed_cell(tmp_path, selected, monkeypatch):
    original = capture.run_task
    drift = False

    async def run(*args, **kwargs):
        nonlocal drift
        result = await original(*args, **kwargs)
        drift = True
        return result

    monkeypatch.setattr(capture, "run_task", run)
    monkeypatch.setattr(
        capture,
        "_harness_revision",
        lambda: {"commit": ("b" if drift else "a") * 40, "dirty": False},
    )
    directory = tmp_path / "capture"
    with pytest.raises(ValueError, match="runtime changed"):
        await capture.capture_public_inputs(directory, **selected)
    assert (directory / "01-wire.json").exists() and (directory / "01-normalized.json").exists()
    assert (directory / "cell-01/admitted.json").exists()
    assert not (directory / "cell-02").exists() and not (directory / "index.json").exists()


@pytest.mark.parametrize("defect", ["settings", "container-mode", "extra-output"])
async def test_capture_drift_preserves_inputs_and_closes_adapter(
    tmp_path, selected, monkeypatch, defect
):
    original_run = capture.run_task
    original_close = capture.OllamaProvider.aclose
    closed = []

    async def close(adapter):
        closed.append(True)
        await original_close(adapter)

    async def run(*args, **kwargs):
        result = await original_run(*args, **kwargs)
        directory = kwargs["public_cell"].directory.parent
        if defect == "settings":
            path = directory / "settings.json"
            path.write_bytes(path.read_bytes() + b" ")
        elif defect == "container-mode":
            directory.chmod(0o755)
        else:
            (directory / "unknown-output").write_text("preserve\n")
        return result

    monkeypatch.setattr(capture, "run_task", run)
    monkeypatch.setattr(capture.OllamaProvider, "aclose", close)
    directory = tmp_path / "capture"
    with pytest.raises(ValueError):
        await capture.capture_public_inputs(directory, **selected)
    assert closed and (directory / "01-wire.json").exists()
    assert (directory / "cell-01/admitted.json").exists()
    assert not (directory / "index.json").exists()
    if defect != "extra-output":
        assert len(closed) == 1 and not (directory / "cell-02").exists()
