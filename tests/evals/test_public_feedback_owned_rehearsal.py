"""Scripted seams test orchestration; native ownership is separate retained proof."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import evals.public_feedback_execution as executor
import evals.public_feedback_owned_rehearsal as rehearsal
from evals.heldout_inputs import _save
from evals.public_feedback_execution_manifest import FeedbackExecutionManifest
from tests.evals import test_public_feedback_execution as execution_tests
from tests.evals import test_public_feedback_preparation as preparation_tests

configured = preparation_tests.configured


@pytest.fixture
def owned(configured, monkeypatch):
    directory, kwargs, _source = configured
    events = []

    def image(source, destination, **options):
        destination.mkdir(mode=0o700)

    monkeypatch.setattr(executor, "prepare_image", image)

    @contextmanager
    def backend(image, path, binary, *, keep_alive_seconds):
        assert image.is_dir() and keep_alive_seconds == 300
        path.mkdir(mode=0o700)
        events.append(("open", path.parent.name))
        try:
            yield SimpleNamespace(check_owned=lambda: None)
        finally:
            report = json.loads((directory / "run/report.json").read_bytes())
            assert report["cohort"]["cells"][len(report["results"]) - 1]["status"] == "started"
            cleanup = {"passed": True, "settled": True, "forced": False, "errors": {}}
            _save(path / "cleanup.json", json.dumps(cleanup).encode())
            _save(
                path / "session.json",
                json.dumps(
                    {
                        "format": "heldout-owned-backend-session-v1",
                        "cleanup": cleanup,
                        "inference_authorized": False,
                    }
                ).encode(),
            )
            events.append(("settle", path.parent.name))

    monkeypatch.setattr(executor, "owned_backend", backend)
    return directory, kwargs, events


def test_fresh_owned_mock_bundle_has_exact_evidence_and_cannot_consume_existing_scope(owned):
    directory, kwargs, events = owned
    result = rehearsal.rehearse_feedback_execution(directory, **kwargs)
    assert result.mock_generation_requests == 24 and result.generation_forwarded == 0
    assert not result.inference_ran and not result.parent_approval_issued
    assert result.evidence_label == "scripted-real-tools-rehearsal"
    assert len(result.cells) == 8 and all(len(cell.native_files) == 9 for cell in result.cells)
    assert events == [
        event
        for i in range(1, 9)
        for event in [("open", f"cell-{i:02d}"), ("settle", f"cell-{i:02d}")]
    ]
    assert (directory / "owned-rehearsal.json").stat().st_mode & 0o777 == 0o600
    fixture = json.loads((directory / "mock-parent-fixture.json").read_bytes())
    assert fixture["freeze_reviewed_head"] == fixture["freeze_merge_commit"] == "0" * 40
    assert "NOT-actual-approval" in fixture["approval_id"]
    report = json.loads((directory / "run/report.json").read_bytes())
    assert report["feedback_comparison"]["benefit_assessment"] == "not_assessed"
    with pytest.raises(FileExistsError):
        rehearsal.rehearse_feedback_execution(directory, **kwargs)
    assert len(events) == 16 and not (directory / "WITHDRAWN.json").exists()


async def test_active_loop_refuses_before_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rehearsal, "prepare_feedback_candidate", lambda *a, **k: pytest.fail("prepare")
    )
    with pytest.raises(ValueError, match="outside an active"):
        rehearsal.rehearse_feedback_execution(
            tmp_path / "new",
            expected_python=Path("/unused/python"),
            expected_commit="a" * 40,
            executables={},
            dependency_files=[],
            source_models=tmp_path,
            command_env={},
        )
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("fault", ["runtime", "false_finish", "recording"])
def test_post_preparation_failure_retains_partial_evidence_and_withdraws(
    tmp_path, monkeypatch, fault
):
    directory = tmp_path.resolve() / "new"
    manifest = FeedbackExecutionManifest.model_validate(execution_tests.values(directory))

    def prepare(*args, **kwargs):
        directory.mkdir(mode=0o700)
        _save(directory / "manifest-candidate.json", manifest.model_dump_json().encode())
        return manifest

    monkeypatch.setattr(rehearsal, "prepare_feedback_candidate", prepare)

    def admit(*args, **kwargs):
        fixture = rehearsal.FeedbackParentApproval.model_validate_json(
            (directory / "mock-parent-fixture.json").read_bytes()
        )
        kwargs["verify_parent"](manifest, fixture)
        return SimpleNamespace()

    monkeypatch.setattr(rehearsal, "admit_feedback_execution", admit)

    def run(*args, **kwargs):
        assert kwargs["scripted"] is True
        if fault == "false_finish":
            return SimpleNamespace(status="finished")
        raise ValueError("scripted runtime failure")

    monkeypatch.setattr(rehearsal, "run_feedback_cohort", run)
    original = rehearsal._save
    if fault == "recording":

        def save(path, raw):
            if path.name == "WITHDRAWN.json":
                raise OSError("recording failure")
            return original(path, raw)

        monkeypatch.setattr(rehearsal, "_save", save)
    with pytest.raises(ValueError) as caught:
        rehearsal.rehearse_feedback_execution(
            directory,
            expected_python=Path("/unused/python"),
            expected_commit="a" * 40,
            executables={},
            dependency_files=[],
            source_models=tmp_path,
            command_env={},
        )
    assert (directory / "manifest-candidate.json").exists()
    assert (directory / "mock-parent-fixture.json").exists()
    assert not (directory / "owned-rehearsal.json").exists()
    if fault == "recording":
        assert "runtime failure" in str(caught.value)
        assert any("OSError" in note for note in caught.value.__notes__)
    else:
        assert (directory / "WITHDRAWN.json").exists()
        assert (
            json.loads((directory / "owned-rehearsal-failure.json").read_bytes())["status"]
            == "failed"
        )


async def test_reference_transport_has_only_three_in_memory_responses_and_closes():
    transport = rehearsal._ReferenceTransport("gemma4:12b", "selected reference")
    payload = {
        "tools": [{"function": {"name": name}} for name in ("write_file", "run_project_check")],
        "messages": [{"content": json.dumps({"exit_code": 0})}],
    }

    def request(url=rehearsal.ENDPOINT):
        return httpx.Request("POST", url, json=payload)

    with pytest.raises(ValueError):
        transport.reply(request("http://127.0.0.1:11434/api/chat"))
    assert transport.requests == 0
    first = transport.reply(request()).json()
    assert (
        first["message"]["tool_calls"][0]["function"]["arguments"]["content"]
        == "selected reference"
    )
    assert (
        transport.reply(request()).json()["message"]["tool_calls"][0]["function"]["name"]
        == "run_project_check"
    )
    assert transport.reply(request()).json()["message"]["tool_calls"] == []
    with pytest.raises(ValueError):
        transport.reply(request())
    await transport.aclose()
    assert transport.closed
    with pytest.raises(ValueError):
        transport.reply(request())
