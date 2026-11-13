"""Owned orchestration with explicit fake image/backend seams; native proof is separate."""

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from evals import fresh_visible_owned_rehearsal as owned
from evals import fresh_visible_rehearsal as capture_module
from evals.heldout_inputs import _save


@pytest.fixture
def configured(tmp_path, monkeypatch):
    revision = {"commit": "b" * 40, "dirty": False}
    monkeypatch.setattr(owned, "_harness_revision", lambda: revision)
    monkeypatch.setattr(capture_module, "_harness_revision", lambda: revision)
    monkeypatch.setattr(owned, "check_existing_interpreter", lambda expected: None)
    source = tmp_path / "models"
    source.mkdir()
    binary = tmp_path / "ollama"
    binary.write_bytes(b"fake binary")
    binary.chmod(0o755)
    dependency = tmp_path / "dependency.py"
    dependency.write_text("value = 1\n")
    events = []

    def image(source, destination, **kwargs):
        destination.mkdir(mode=0o700)
        _save(destination / "prepared-image.json", b'{"fixture":"fake-image-only"}')

    monkeypatch.setattr(owned, "prepare_image", image)

    @contextmanager
    def backend(image, directory, executable, *, keep_alive_seconds):
        assert keep_alive_seconds == 300 and executable == binary
        directory.mkdir(mode=0o700)
        ordinal = int(directory.parent.name[-2:])
        events.append(("open", ordinal))

        def check():
            events.append(("check", ordinal))

        try:
            yield SimpleNamespace(check_owned=check)
        finally:
            result_path = tmp_path / f"owned/capture/{ordinal:02d}-result.json"
            # A successful callback must retain its row before settlement.
            if result_path.exists():
                events.append(("result_before_settle", ordinal))
            cleanup = {
                "passed": True,
                "settled": True,
                "forced": False,
                "returncode": 0,
                "errors": {},
            }
            _save(directory / "cleanup.json", json.dumps(cleanup).encode())
            _save(
                directory / "session.json",
                json.dumps(
                    {
                        "format": "heldout-owned-backend-session-v1",
                        "inference_authorized": False,
                        "cleanup": cleanup,
                    }
                ).encode(),
            )
            events.append(("settle", ordinal))

    monkeypatch.setattr(owned, "owned_backend", backend)
    return (
        tmp_path / "owned",
        {
            "expected_python": Path(sys.executable),
            "expected_commit": "b" * 40,
            "executables": {"python": Path(sys.executable), "ollama": binary},
            "dependency_files": [dependency],
            "source_models": source,
            "command_env": {
                "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
                "LC_ALL": "C",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        },
        events,
    )


def test_twelve_owned_cells_settle_serially_and_keep_generation_mocked(configured, monkeypatch):
    directory, kwargs, events = configured

    def forbidden(*a, **k):
        pytest.fail("HTTP construction in fake owned fixture")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "__init__", forbidden)
    result = owned.rehearse_fresh_owned(directory, **kwargs)
    assert len(result.cells) == 12 and result.mock_generation_requests == 36
    assert result.generation_forwarded == 0 and not result.inference_ran
    assert not result.parent_approval_issued
    lifecycle = [event for event in events if event[0] != "check"]
    assert lifecycle == [
        event
        for i in range(1, 13)
        for event in [("open", i), ("result_before_settle", i), ("settle", i)]
    ]
    assert all(sum(e == ("check", i) for e in events) >= 6 for i in range(1, 13))
    assert (directory / "capture/index.json").exists() and (
        directory / "owned-rehearsal.json"
    ).exists()
    report = json.loads((directory / "capture/report.json").read_bytes())
    assert report["mode"] == "scripted-real-tools-rehearsal"
    assert all(
        row["passed"] and row["completion_status"] == "completed" for row in report["results"]
    )
    with pytest.raises(FileExistsError):
        owned.rehearse_fresh_owned(directory, **kwargs)
    assert len([e for e in events if e[0] == "open"]) == 12


@pytest.mark.parametrize("failure", ["forced", "failed", "unknown", "corrupt_prior"])
def test_bad_settlement_or_retained_evidence_prevents_success(configured, monkeypatch, failure):
    directory, kwargs, events = configured
    original = owned.owned_backend

    @contextmanager
    def bad(*args, **options):
        with original(*args, **options) as session:
            yield session
        path = args[1]
        if failure == "corrupt_prior":
            if path.parent.name == "cell-02":
                (path.parent.parent / "cell-01/backend/session.json").write_text("{}")
        else:
            cleanup = json.loads((path / "cleanup.json").read_bytes())
            cleanup["forced" if failure == "forced" else "passed"] = failure == "forced"
            if failure == "unknown":
                cleanup.pop("settled")
            (path / "cleanup.json").write_text(json.dumps(cleanup))

    monkeypatch.setattr(owned, "owned_backend", bad)
    with pytest.raises(ValueError):
        owned.rehearse_fresh_owned(directory, **kwargs)
    assert (directory / "owned-failure.json").exists() and not (
        directory / "owned-rehearsal.json"
    ).exists()
    assert not (directory / "capture/index.json").exists()
    if failure != "corrupt_prior":
        assert [e for e in events if e[0] == "open"] == [("open", 1)]
    else:
        assert [e for e in events if e[0] == "open"] == [("open", 1), ("open", 2)]


async def test_active_event_loop_refuses_before_allocation(configured):
    directory, kwargs, _ = configured
    with pytest.raises(ValueError, match="outside an active"):
        owned.rehearse_fresh_owned(directory, **kwargs)
    assert not directory.exists()


def test_input_overlap_refuses_without_allocation(configured):
    directory, kwargs, _ = configured
    with pytest.raises(ValueError, match="overlaps"):
        owned.rehearse_fresh_owned(kwargs["source_models"] / "child", **kwargs)
    assert not directory.exists()


def test_failed_owned_guard_stops_mock_dispatch_and_settles(configured, monkeypatch):
    directory, kwargs, events = configured
    original = owned.owned_backend

    @contextmanager
    def bad(*args, **options):
        with original(*args, **options):
            yield SimpleNamespace(
                check_owned=lambda: (_ for _ in ()).throw(ValueError("ownership"))
            )

    monkeypatch.setattr(owned, "owned_backend", bad)
    with pytest.raises(ValueError):
        owned.rehearse_fresh_owned(directory, **kwargs)
    assert ("settle", 1) in events and ("open", 2) not in events
    assert not (directory / "capture/native-01").exists()
    assert (directory / "owned-failure.json").exists()


def test_image_preparation_failure_retains_primary_error_and_no_backend(configured, monkeypatch):
    directory, kwargs, events = configured

    def fail(*args, **options):
        raise OSError("selected image copy failed")

    monkeypatch.setattr(owned, "prepare_image", fail)
    with pytest.raises(OSError, match="image copy"):
        owned.rehearse_fresh_owned(directory, **kwargs)
    assert not events and (directory / "owned-failure.json").is_file()
    assert not (directory / "capture/index.json").exists()
