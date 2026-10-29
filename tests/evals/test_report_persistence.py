"""Real-file report replacement with injected local I/O failures."""

import json
import os
import stat
from pathlib import Path

import pytest

from evals import real_run


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["write", "flush", "fsync", "replace"])
def test_failed_report_update_preserves_previous_bytes_and_removes_temporary(
    monkeypatch, tmp_path, existing, failure
):
    report = tmp_path / "report.json"
    previous = b'{"previous": true}\n'
    if existing:
        report.write_bytes(previous)
    original = real_run.tempfile.NamedTemporaryFile

    def fail(*args, **kwargs):
        raise OSError("injected report I/O failure")

    if failure in ("write", "flush"):

        def temporary(*args, **kwargs):
            stream = original(*args, **kwargs)
            if failure == "write":

                def partial(content):
                    stream.file.write(content[:8])
                    raise OSError("partial write")

                stream.write = partial
            else:
                stream.flush = fail
            return stream

        monkeypatch.setattr(real_run.tempfile, "NamedTemporaryFile", temporary)
    else:
        monkeypatch.setattr(real_run.os, failure, fail)
    with pytest.raises(OSError):
        real_run.write_report(report, "scripted failure", [])
    assert report.exists() is existing
    if existing:
        assert report.read_bytes() == previous
    assert list(tmp_path.glob(".*.tmp")) == []


def test_successful_replacement_publishes_complete_private_json(monkeypatch, tmp_path):
    report = tmp_path / "nested" / "report.json"
    report.parent.mkdir()
    report.write_text("previous")
    report.chmod(0o644)
    replace = os.replace
    observed = []

    def publish(source, destination):
        assert Path(source).parent == report.parent
        assert report.read_text() == "previous"
        data = json.loads(Path(source).read_text())
        assert data["started_at"] == "replacement" and data["results"] == []
        assert stat.S_IMODE(Path(source).stat().st_mode) == 0o600
        observed.append(True)
        replace(source, destination)

    monkeypatch.setattr(real_run.os, "replace", publish)
    real_run.write_report(report, "replacement", [])
    assert observed == [True]
    assert json.loads(report.read_text())["cohort"] is None
    assert stat.S_IMODE(report.stat().st_mode) == 0o600
    assert list(report.parent.glob(".*.tmp")) == []


def test_report_symlink_is_replaced_without_writing_target(tmp_path):
    target = tmp_path / "original.json"
    target.write_text("preserve target")
    report = tmp_path / "report.json"
    report.symlink_to(target)
    real_run.write_report(report, "replace link", [])
    assert not report.is_symlink()
    assert target.read_text() == "preserve target"
    assert json.loads(report.read_text())["started_at"] == "replace link"
