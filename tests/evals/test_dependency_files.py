import hashlib
import os

import pytest

from evals import heldout_dependency_files as files


@pytest.fixture
def recorded(tmp_path):
    source = tmp_path / "module.py"
    source.write_bytes(b"value = 1\n")
    report = tmp_path / "pins.json"
    files.record_files([source], report)
    return source, report, hashlib.sha256(report.read_bytes()).hexdigest()


def test_selected_file_bytes_recheck_without_import_or_modification(recorded):
    source, report, digest = recorded
    before = (source.read_bytes(), report.read_bytes(), source.stat(), report.stat())
    result = files.verify_files(report, expected_sha256=digest)
    assert not result.import_closure_verified and not result.inference_authorized
    assert before == (source.read_bytes(), report.read_bytes(), source.stat(), report.stat())


@pytest.mark.parametrize("fault", ["bytes", "mode", "missing", "link", "report", "report-mode"])
def test_drift_refuses_without_rebinding(recorded, tmp_path, fault):
    source, report, digest = recorded
    if fault == "bytes":
        source.write_bytes(b"value = 2\n")
    elif fault == "mode":
        source.chmod(0o400)
    elif fault == "missing":
        source.unlink()
    elif fault == "link":
        other = tmp_path / "other.py"
        source.rename(other)
        source.symlink_to(other)
    elif fault == "report":
        report.write_bytes(b"{}")
    else:
        report.chmod(0o644)
    before = report.read_bytes()
    with pytest.raises((ValueError, OSError)):
        files.verify_files(report, expected_sha256=digest)
    assert report.read_bytes() == before


@pytest.mark.parametrize("fault", ["configuration", "duplicate", "size", "total", "fifo"])
def test_invalid_selection_refuses_before_report(recorded, tmp_path, monkeypatch, fault):
    source, _, _ = recorded
    selected = [source]
    if fault == "configuration":
        source = tmp_path / ".env.py"
        source.write_bytes(b"not read")
        selected = [source]
        monkeypatch.setattr(files, "_hash_blob", lambda *a: pytest.fail("read config"))
    elif fault == "duplicate":
        selected = [source, source]
    elif fault == "size":
        monkeypatch.setattr(files, "MAX_FILE_BYTES", 1)
    elif fault == "total":
        monkeypatch.setattr(files, "MAX_TOTAL_BYTES", 1)
    else:
        source.unlink()
        os.mkfifo(source)
    report = tmp_path / "refused.json"
    with pytest.raises(ValueError):
        files.record_files(selected, report)
    assert not report.exists()
