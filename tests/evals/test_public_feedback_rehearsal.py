import hashlib
import json
import sys
from pathlib import Path

import httpx
import pytest

import evals.public_feedback_rehearsal as rehearsal
from evals.heldout_dependency_files import record_files
from evals.heldout_runtime_inventory import record_runtime
from providers.base import ChatMessage, ToolSpec
from providers.ollama import _message_to_ollama, _tool_to_ollama


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def selected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rehearsal, "_harness_revision", lambda: {"commit": "a" * 40, "dirty": False}
    )
    runtime = tmp_path / "runtime.json"
    record_runtime({"python": Path(sys.executable)}, runtime)
    dependency = tmp_path / "dependency.py"
    dependency.write_text("value = 1\n")
    dependencies = tmp_path / "dependencies.json"
    record_files([dependency], dependencies)
    return {
        "command_env": {
            "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "runtime_inventory": runtime,
        "runtime_sha256": sha(runtime),
        "dependency_manifest": dependencies,
        "dependency_sha256": sha(dependencies),
    }


async def test_eight_mocked_reference_cells_capture_native_inputs_and_real_check_feedback(
    tmp_path, selected, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("Real HTTP transport constructed")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "__init__", forbidden)
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:9191")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:9191")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret-must-not-appear")
    directory = tmp_path / "rehearsal"
    index = await rehearsal.rehearse_feedback(directory, **selected)
    assert len(index.cells) == 8 and index.generation_forwarded == 0
    assert index.inference_authorized is False
    assert index.evidence_label == "scripted-real-tools-rehearsal"
    assert sha(directory / "report.json") == index.report_sha256
    report = json.loads((directory / "report.json").read_bytes())
    assert report["mode"] == "scripted-real-tools-rehearsal"
    assert all(
        row["passed"] and row["completion_status"] == "completed" for row in report["results"]
    )
    for pin, result in zip(index.cells, report["results"], strict=True):
        prefix = f"{pin.cell.ordinal:02d}"
        for name, digest in pin.files.items():
            assert sha(directory / name) == digest
            assert (directory / name).stat().st_mode & 0o777 == 0o600
        wire = json.loads((directory / f"native-{prefix}/request-01.request.bin").read_bytes())
        normalized = json.loads((directory / f"{prefix}-normalized.json").read_bytes())
        assert wire["model"] == "gemma4:12b" and wire["stream"] is False
        assert wire["options"]["num_ctx"] == 16384 and wire["think"] is False
        assert normalized["messages"][0]["role"] == "system"
        assert wire == {
            "model": "gemma4:12b",
            "stream": False,
            "think": False,
            "messages": [
                _message_to_ollama(ChatMessage.model_validate(m)) for m in normalized["messages"]
            ],
            "tools": [_tool_to_ollama(ToolSpec.model_validate(t)) for t in normalized["tools"]],
            "options": {
                "temperature": normalized["temperature"],
                "num_predict": normalized["max_tokens"],
                "num_ctx": 16384,
            },
        }
        names = {t["name"] for t in normalized["tools"]}
        assert {"write_file", "run_project_check"} <= names
        assert not {"replace_text", "run_command", "emit_plan"} & names
        checks = [t for t in result["tool_trace"] if t["name"] == "run_project_check"]
        assert [c["result"]["exit_code"] for c in checks] == [
            1 if pin.cell.arm == "bool_check" else 0,
            0,
        ]
        assert result["prompt_tokens"] == 400 and result["completion_tokens"] == 40
        assert (directory / f"{prefix}-initial-visible.py").read_bytes() == (
            directory / f"{prefix}-final-visible.py"
        ).read_bytes()
        assert pin.workspace_root == str(directory / f"cell-{prefix}/workspace")
    assert (directory / "index.json").stat().st_mode & 0o777 == 0o600
    assert "ambient-secret-must-not-appear" not in (directory / "settings.json").read_text()
    with pytest.raises(FileExistsError):
        await rehearsal.rehearse_feedback(directory, **selected)


@pytest.mark.parametrize(
    "defect", ["dirty", "unknown", "dependency", "runtime_mode", "environment"]
)
async def test_invalid_preconditions_refuse_before_output(tmp_path, selected, monkeypatch, defect):
    if defect in {"dirty", "unknown"}:
        monkeypatch.setattr(
            rehearsal,
            "_harness_revision",
            lambda: {
                "commit": None if defect == "unknown" else "a" * 40,
                "dirty": defect == "dirty",
            },
        )
    elif defect == "dependency":
        (tmp_path / "dependency.py").write_text("changed\n")
    elif defect == "runtime_mode":
        selected["runtime_inventory"].chmod(0o644)
    else:
        selected["command_env"] = {"PATH": "relative"}
    directory = tmp_path / "rehearsal"
    with pytest.raises(ValueError):
        await rehearsal.rehearse_feedback(directory, **selected)
    assert not directory.exists()


@pytest.mark.parametrize(
    "defect",
    ["runtime", "settings", "extra", "container", "native", "native_inventory", "receipt", "test"],
)
async def test_after_cell_drift_retains_native_inputs_and_stops_later_cells(
    tmp_path, selected, monkeypatch, defect
):
    run = rehearsal.run_task
    changed = False

    async def drift(*args, **kwargs):
        nonlocal changed
        result = await run(*args, **kwargs)
        changed = True
        directory = kwargs["feedback_cell"].directory.parent
        if defect == "settings":
            (directory / "settings.json").write_bytes(b"changed")
        elif defect == "extra":
            (directory / "unexpected").touch()
        elif defect == "container":
            directory.chmod(0o755)
        elif defect == "receipt":
            (directory / "cell-01/admitted.json").write_text("{}")
        elif defect == "test":
            (directory / "cell-01/workspace/test_visible.py").write_text("# changed\n")
        elif defect == "native_inventory":
            (directory / "native-01/request-01.json").rename(directory / "native-01/extra.json")
        elif defect == "native":
            (directory / "native-01/request-01.request.bin").write_bytes(b"changed")
        return result

    monkeypatch.setattr(rehearsal, "run_task", drift)
    if defect == "runtime":
        monkeypatch.setattr(
            rehearsal,
            "_harness_revision",
            lambda: {"commit": ("b" if changed else "a") * 40, "dirty": False},
        )
    directory = tmp_path / "rehearsal"
    with pytest.raises(ValueError):
        await rehearsal.rehearse_feedback(directory, **selected)
    assert (directory / "01-normalized.json").exists()
    assert (directory / "native-01/request-01.request.bin").exists()
    assert (directory / "cell-01/admitted.json").exists()
    assert not (directory / "index.json").exists()
    assert not (directory / "cell-02").exists()


async def test_provider_failure_closes_adapter_and_preserves_failed_cell(
    tmp_path, selected, monkeypatch
):
    original = rehearsal.OllamaProvider.chat
    close = rehearsal.OllamaProvider.aclose
    calls = 0
    closed = []

    async def fail(adapter, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated adapter failure")
        return await original(adapter, *args, **kwargs)

    async def record_close(adapter):
        closed.append(True)
        await close(adapter)

    monkeypatch.setattr(rehearsal.OllamaProvider, "chat", fail)
    monkeypatch.setattr(rehearsal.OllamaProvider, "aclose", record_close)
    directory = tmp_path / "rehearsal"
    with pytest.raises(ValueError, match="did not complete"):
        await rehearsal.rehearse_feedback(directory, **selected)
    assert closed == [True]
    assert (directory / "01-result.json").exists()
    assert not (directory / "cell-02").exists() and not (directory / "index.json").exists()
