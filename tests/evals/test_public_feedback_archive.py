import asyncio
import hashlib
import json
import shutil
import sys
from pathlib import Path

import httpx
import pytest

import evals.public_feedback_archive as archive
import evals.public_feedback_rehearsal as rehearsal
from evals.heldout_dependency_files import record_files
from evals.heldout_inputs import normalized_input
from evals.heldout_runtime_inventory import record_runtime
from providers.base import ChatMessage, ToolSpec
from providers.ollama import OllamaProvider


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    root = tmp_path_factory.mktemp("feedback-archive")
    runtime = root / "runtime.json"
    record_runtime({"python": Path(sys.executable)}, runtime)
    dependency = root / "dependency.py"
    dependency.write_text("value = 1\n")
    dependencies = root / "dependencies.json"
    record_files([dependency], dependencies)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rehearsal, "_harness_revision", lambda: {"commit": "a" * 40, "dirty": False})
        asyncio.run(
            rehearsal.rehearse_feedback(
                root / "capture",
                command_env={
                    "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                runtime_inventory=runtime,
                runtime_sha256=sha(runtime),
                dependency_manifest=dependencies,
                dependency_sha256=sha(dependencies),
            )
        )
    return root / "capture"


@pytest.fixture
def copied(tmp_path, captured):
    target = tmp_path / "archive"
    shutil.copytree(captured, target)
    return target


def verify(directory, original):
    return archive.verify_feedback_capture(
        directory, expected_index_sha256=sha(directory / "index.json"), original_directory=original
    )


def rewrite(directory, name, value):
    (directory / name).write_text(json.dumps(value))
    index = json.loads((directory / "index.json").read_bytes())
    index["cells"][0]["files"][name] = sha(directory / name)
    (directory / "index.json").write_text(json.dumps(index))


def test_capture_verification_is_read_only_and_constructs_no_provider(captured, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Verification constructed inference machinery")

    monkeypatch.setattr(OllamaProvider, "__init__", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "__init__", forbidden)
    before = {str(p.relative_to(captured)): sha(p) for p in captured.rglob("*") if p.is_file()}
    selected = archive.verify_feedback_capture(
        captured, expected_index_sha256=sha(captured / "index.json")
    )
    assert len(selected.cells) == 8 and selected.inference_authorized is False
    assert before == {
        str(p.relative_to(captured)): sha(p) for p in captured.rglob("*") if p.is_file()
    }


def test_archival_verification_does_not_reobserve_historical_dependencies(copied, captured):
    dependency = captured.parent / "dependency.py"
    original = dependency.read_bytes()
    try:
        dependency.write_text("historical dependency changed\n")
        assert len(verify(copied, captured).cells) == 8
    finally:
        dependency.write_bytes(original)


@pytest.mark.parametrize(
    "name",
    [
        "01-normalized.json",
        "native-01/request-01.request.bin",
        "cell-01/prepared.json",
        "settings.json",
        "report.json",
    ],
)
def test_selected_file_hash_drift_refuses(copied, captured, name):
    path = copied / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        verify(copied, captured)


@pytest.mark.parametrize("change", ["mode", "symlink"])
def test_private_file_boundary_refuses(copied, captured, change):
    path = copied / "01-normalized.json"
    if change == "mode":
        path.chmod(0o644)
    else:
        path.unlink()
        path.symlink_to(captured / "01-normalized.json")
    with pytest.raises((ValueError, OSError)):
        verify(copied, captured)


@pytest.mark.parametrize("change", ["order", "filemap", "workspace", "authority"])
def test_rebound_index_semantic_drift_refuses(copied, captured, change):
    index = json.loads((copied / "index.json").read_bytes())
    if change == "order":
        index["cells"].reverse()
    elif change == "filemap":
        files = index["cells"][0]["files"]
        files["../../escape"] = files.pop("01-normalized.json")
    elif change == "workspace":
        index["cells"][0]["workspace_root"] += "-other"
    else:
        index["inference_authorized"] = True
    (copied / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        verify(copied, captured)


def test_rebound_native_input_must_match_normalized_input(copied, captured):
    name = "native-01/request-01.request.bin"
    wire = json.loads((copied / name).read_bytes())
    wire["options"]["temperature"] = 1
    rewrite(copied, name, wire)
    with pytest.raises(ValueError, match="native first request"):
        verify(copied, captured)


def test_normalized_task_identity_cannot_be_rebound(copied, captured):
    name = "01-normalized.json"
    normalized = json.loads((copied / name).read_bytes())
    normalized["messages"][1]["content"] += " changed"
    raw = normalized_input(
        [ChatMessage.model_validate(m) for m in normalized["messages"]],
        [ToolSpec.model_validate(t) for t in normalized["tools"]],
        normalized["temperature"],
        normalized["max_tokens"],
    )
    rewrite(copied, name, json.loads(raw))
    with pytest.raises(ValueError, match="first request task"):
        verify(copied, captured)


@pytest.mark.parametrize("change", ["config", "receipt", "visible", "native_meta"])
def test_rebound_selected_file_semantic_drift_refuses(copied, captured, change):
    if change == "config":
        name = "01-runtime-config.json"
        data = json.loads((copied / name).read_bytes())
        data["coding_toolset"] = "full"
    elif change == "receipt":
        name = "cell-01/admitted.json"
        data = json.loads((copied / name).read_bytes())
        data["cell"]["arm"] = "bool_check"
    elif change == "visible":
        name = "01-initial-visible.py"
        data = "changed"
    else:
        name = "native-01/request-01.json"
        data = json.loads((copied / name).read_bytes())
        data["status_code"] = 500
    rewrite(copied, name, data)
    with pytest.raises(ValueError):
        verify(copied, captured)


@pytest.mark.parametrize("change", ["result", "usage"])
def test_rebound_report_and_cell_result_must_agree_with_recorded_outcomes(copied, captured, change):
    report = json.loads((copied / "report.json").read_bytes())
    if change == "result":
        report["results"][0]["passed"] = False
    else:
        report["results"][0]["prompt_tokens"] += 1
    rewrite(copied, "01-result.json", report["results"][0])
    (copied / "report.json").write_text(json.dumps(report))
    index = json.loads((copied / "index.json").read_bytes())
    index["report_sha256"] = sha(copied / "report.json")
    (copied / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        verify(copied, captured)


@pytest.mark.parametrize("where", ["outer", "cell", "native", "implementation"])
def test_extra_inventory_and_final_artifact_changes_refuse(copied, captured, where):
    relative = {
        "outer": "extra",
        "cell": "cell-01/extra",
        "native": "native-01/extra",
        "implementation": "cell-01/workspace/pagination.py",
    }[where]
    (copied / relative).write_text("changed\n")
    with pytest.raises(ValueError):
        verify(copied, captured)


def test_wrong_selected_index_hash_refuses(captured):
    with pytest.raises(ValueError):
        archive.verify_feedback_capture(captured, expected_index_sha256="0" * 64)
