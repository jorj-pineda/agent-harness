import hashlib
import json
from types import SimpleNamespace

import pytest

from evals import heldout_runtime_guard as guard
from evals import heldout_runtime_inventory as inventory


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    binary = tmp_path / "python"
    binary.write_bytes(b"scripted executable")
    binary.chmod(0o700)
    monkeypatch.setattr(inventory.sys, "executable", str(binary))
    monkeypatch.setattr(
        inventory.importlib.metadata,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": "pkg"}, version="1")],
    )
    report = tmp_path / "inventory.json"
    inventory.record_runtime({"python": binary}, report)
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    return binary, report, digest


def test_matching_runtime_is_checked_read_only(recorded):
    binary, report, digest = recorded
    before = (binary.read_bytes(), report.read_bytes(), binary.stat(), report.stat())
    result = guard.verify_runtime(report, expected_sha256=digest)
    assert not result.dependency_bytes_verified and not result.inference_authorized
    assert before == (binary.read_bytes(), report.read_bytes(), binary.stat(), report.stat())


@pytest.mark.parametrize(
    "fault", ["hash", "mode", "schema", "bytes", "binary-mode", "metadata", "link-target"]
)
def test_changed_or_unknown_runtime_refuses_without_rewriting(
    recorded, monkeypatch, tmp_path, fault
):
    binary, report, digest = recorded
    if fault == "hash":
        digest = "0" * 64
    elif fault == "mode":
        report.chmod(0o644)
    elif fault == "schema":
        data = json.loads(report.read_bytes())
        data["inference_authorized"] = True
        report.write_text(json.dumps(data))
        digest = hashlib.sha256(report.read_bytes()).hexdigest()
    elif fault == "bytes":
        binary.write_bytes(b"modified executable")
    elif fault == "binary-mode":
        binary.chmod(0o500)
    elif fault == "metadata":
        monkeypatch.setattr(
            inventory.importlib.metadata,
            "distributions",
            lambda: [SimpleNamespace(metadata={"Name": "pkg"}, version="2")],
        )
    else:
        target = tmp_path / "other"
        binary.rename(target)
        binary.symlink_to(target)
    before = report.read_bytes()
    with pytest.raises((ValueError, OSError)):
        guard.verify_runtime(report, expected_sha256=digest)
    assert report.read_bytes() == before


def test_wrong_report_hash_refuses_before_runtime_observation(recorded, monkeypatch):
    _, report, _ = recorded
    monkeypatch.setattr(guard, "observe_runtime", lambda *a: pytest.fail("runtime observed"))
    with pytest.raises(ValueError, match="selected hash"):
        guard.verify_runtime(report, expected_sha256="0" * 64)
