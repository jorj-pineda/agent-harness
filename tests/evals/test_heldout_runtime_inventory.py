import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import heldout_runtime_inventory as inventory


@pytest.fixture
def selected(tmp_path, monkeypatch):
    executable = tmp_path / "python"
    executable.write_bytes(b"scripted executable")
    executable.chmod(0o700)
    monkeypatch.setattr(inventory.sys, "executable", str(executable))
    monkeypatch.setattr(
        inventory.importlib.metadata,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": "Some_Package"}, version="1.2.3")],
    )
    return {"python": executable}


def test_private_report_hashes_explicit_binary_without_changing_it(selected, tmp_path):
    binary = selected["python"]
    before = (binary.read_bytes(), binary.stat())
    report = tmp_path / "inventory.json"
    result = inventory.record_runtime(selected, report)
    assert result["executables"]["python"]["size"] == len(before[0])
    assert result["distributions"] == {"some-package": "1.2.3"}
    assert result["dependency_bytes_verified"] is False
    assert result["command_environment_verified"] is False
    assert result["inference_authorized"] is False
    assert report.stat().st_mode & 0o777 == 0o600
    assert binary.read_bytes() == before[0] and binary.stat() == before[1]
    assert json.loads(report.read_bytes()) == result
    with pytest.raises(FileExistsError):
        inventory.record_runtime(selected, report)


@pytest.mark.parametrize(
    "fault", ["python", "relative", "label", "permission", "size", "duplicate", "count", "version"]
)
def test_invalid_inventory_preserves_absent_report(selected, tmp_path, monkeypatch, fault):
    if fault == "python":
        selected = {}
    elif fault == "relative":
        selected["other"] = Path("relative")
    elif fault == "label":
        selected["bad label"] = selected["python"]
    elif fault == "permission":
        selected["python"].chmod(0o600)
    elif fault == "size":
        monkeypatch.setattr(inventory, "MAX_EXECUTABLE_BYTES", 1)
    else:
        row = SimpleNamespace(
            metadata={"Name": "pkg"}, version="1" if fault != "version" else "bad\nversion"
        )
        monkeypatch.setattr(
            inventory.importlib.metadata,
            "distributions",
            lambda: iter([row] * (2001 if fault == "count" else 2 if fault == "duplicate" else 1)),
        )
    report = tmp_path / "report.json"
    with pytest.raises(ValueError):
        inventory.record_runtime(selected, report)
    assert not report.exists()


def test_symlinked_binary_is_observed_but_not_replaced(selected, tmp_path):
    binary = selected["python"]
    link = tmp_path / "python-link"
    link.symlink_to(binary)
    result = inventory.observe_runtime({"python": link})
    assert result["executables"]["python"]["resolved_path"] == str(binary)
    assert link.is_symlink()


def test_actual_interpreter_identity_is_required(selected, tmp_path):
    other = tmp_path / "other"
    other.write_bytes(b"other")
    with pytest.raises(ValueError, match="running interpreter"):
        inventory.observe_runtime({"python": other})
