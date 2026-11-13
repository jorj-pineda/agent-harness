"""Mandatory mocked generation and real disposable-tool capture, not model evidence."""

import hashlib
import json
import sys
from pathlib import Path

import httpx
import pytest

from evals import fresh_visible_rehearsal as rehearsal
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


async def test_twelve_native_mock_captures_match_normalized_input_and_pass_original_grading(
    tmp_path, monkeypatch, selected
):
    def forbidden(*args, **kwargs):
        pytest.fail("Real HTTP transport constructed")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "__init__", forbidden)
    monkeypatch.setenv("HTTP_PROXY", "http://invalid.example")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret-must-not-appear")
    directory = tmp_path / "capture"
    summary = await rehearsal.rehearse_fresh_inputs(directory, **selected)
    assert summary.mock_generation_requests == 36 and summary.generation_forwarded == 0
    assert not summary.inference_authorized and len(summary.cells) == 12
    assert sha(directory / "report.json") == summary.report_sha256
    report = json.loads((directory / "report.json").read_bytes())
    assert report["mode"] == "scripted-real-tools-rehearsal"
    for cell, result in zip(summary.cells, report["results"], strict=True):
        assert result["passed"] and result["completion_status"] == "completed"
        assert (result["prompt_tokens"], result["completion_tokens"]) == (30, 15)
        for name, digest in cell.files.items():
            assert sha(directory / name) == digest
            assert (directory / name).stat().st_mode & 0o777 == 0o600
        prefix = f"{cell.cell.ordinal:02d}"
        normalized = json.loads((directory / f"{prefix}-normalized.json").read_bytes())
        native = json.loads((directory / f"native-{prefix}/request-01.request.bin").read_bytes())
        assert native == {
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
        assert (directory / f"{prefix}-initial-visible.py").read_bytes() == (
            directory / f"{prefix}-final-visible.py"
        ).read_bytes()
        assert "test_visible.py" not in result["modified"]
        assert (
            result["runtime_config"]["fresh_visible_preservation"]["ordinal"] == cell.cell.ordinal
        )
    assert "ambient-secret-must-not-appear" not in (directory / "settings.json").read_text()
    with pytest.raises(FileExistsError):
        await rehearsal.rehearse_fresh_inputs(directory, **selected)


@pytest.mark.parametrize("input_name", ["runtime_inventory", "dependency_manifest"])
async def test_inventory_drift_refuses_before_allocation(tmp_path, selected, input_name):
    selected[input_name].write_text("{}")
    with pytest.raises(ValueError):
        await rehearsal.rehearse_fresh_inputs(tmp_path / "capture", **selected)
    assert not (tmp_path / "capture").exists()


async def test_native_failure_retains_partial_evidence_closes_and_does_not_publish_success(
    tmp_path, monkeypatch, selected
):
    transports = []
    original = rehearsal._ReferenceTransport

    class Fail(original):
        def __init__(self, *args):
            super().__init__(*args)
            transports.append(self)

        def reply(self, request):
            raise ValueError("Injected native failure")

    monkeypatch.setattr(rehearsal, "_ReferenceTransport", Fail)
    directory = tmp_path / "capture"
    with pytest.raises(ValueError):
        await rehearsal.rehearse_fresh_inputs(directory, **selected)
    assert transports and all(t.closed for t in transports)
    assert (directory / "failure.json").is_file() and not (directory / "index.json").exists()
    assert (directory / "01-result.json").is_file()
    assert (directory / "native-01/request-01.failure.json").is_file()
    assert not (directory / "cell-02").exists()


async def test_preservation_drift_stops_later_dispatch_without_rebinding(
    tmp_path, monkeypatch, selected
):
    original = rehearsal._ReferenceTransport.reply
    directory = tmp_path / "capture"

    def mutate(self, request):
        response = original(self, request)
        (directory / "settings.json").write_text("{}")
        return response

    monkeypatch.setattr(rehearsal._ReferenceTransport, "reply", mutate)
    with pytest.raises(ValueError):
        await rehearsal.rehearse_fresh_inputs(directory, **selected)
    assert (directory / "failure.json").exists() and not (directory / "index.json").exists()
    assert not (directory / "cell-02").exists()


@pytest.mark.parametrize("mutation", ["method", "url", "model"])
def test_mock_transport_has_no_live_or_unexpected_request_path(mutation):
    transport = rehearsal._ReferenceTransport("gemma4:12b", "lookup.py", "reference")
    request = httpx.Request(
        "GET" if mutation == "method" else "POST",
        "http://invalid/api/chat" if mutation == "url" else "http://127.0.0.1:11447/api/chat",
        json={"model": "wrong" if mutation == "model" else "gemma4:12b", "tools": []},
    )
    with pytest.raises(ValueError):
        transport.reply(request)


async def test_cancellation_retains_failure_and_closes_native_transport(
    tmp_path, monkeypatch, selected
):
    import asyncio

    transports = []
    original = rehearsal._ReferenceTransport

    class Cancel(original):
        def __init__(self, *args):
            super().__init__(*args)
            transports.append(self)

        async def handle_async_request(self, request):
            raise asyncio.CancelledError("cancel scripted rehearsal")

    monkeypatch.setattr(rehearsal, "_ReferenceTransport", Cancel)
    directory = tmp_path / "capture"
    with pytest.raises(asyncio.CancelledError):
        await rehearsal.rehearse_fresh_inputs(directory, **selected)
    assert all(t.closed for t in transports) and transports
    assert (directory / "failure.json").exists() and not (directory / "index.json").exists()
    assert (directory / "native-01/request-01.failure.json").exists()
