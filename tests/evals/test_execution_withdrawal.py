import os

import pytest

from evals import heldout_execution_admission as admission
from tests.evals.test_execution_admission import admit as admit
from tests.evals.test_execution_manifest import selected as selected


@pytest.mark.parametrize("kind", ["file", "directory", "broken_link", "fifo"])
def test_any_marker_refuses_before_parent_verification_or_preflight(
    admit, selected, monkeypatch, kind
):
    marker = selected[0].parent / "WITHDRAWN.json"
    if kind == "file":
        marker.write_bytes(b"not even valid JSON; never read")
    elif kind == "directory":
        marker.mkdir()
    elif kind == "broken_link":
        marker.symlink_to(marker.parent / "missing")
    else:
        os.mkfifo(marker)
    monkeypatch.setattr(admission, "preflight", lambda *a: pytest.fail("preflight called"))
    with pytest.raises(ValueError, match="withdrawn"):
        admit(lambda *a: pytest.fail("parent verification called"))
    assert not (marker.parent / "output_directory").exists()
    assert marker.exists() or marker.is_symlink()


def test_withdrawal_during_preflight_refuses_before_output(admit, selected, monkeypatch):
    def withdraw(manifest):
        (selected[0].parent / "WITHDRAWN.json").write_bytes(b"withdrawn")
        return "scripted-capture"

    monkeypatch.setattr(admission, "preflight", withdraw)
    with pytest.raises(ValueError, match="withdrawn"):
        admit()
    assert not (selected[0].parent / "output_directory").exists()


def test_withdrawal_during_preservation_retains_partial_without_success(
    admit, selected, monkeypatch
):
    save = admission._save

    def withdraw(path, content):
        result = save(path, content)
        if path.name == "parent-approval.json":
            (selected[0].parent / "WITHDRAWN.json").write_bytes(b"withdrawn")
        return result

    monkeypatch.setattr(admission, "_save", withdraw)
    with pytest.raises(ValueError, match="withdrawn"):
        admit()
    output = selected[0].parent / "output_directory"
    assert (output / "manifest.json").read_bytes() == selected[0].read_bytes()
    assert (output / "parent-approval.json").read_bytes() == selected[2].read_bytes()
    assert (output / "admission-failure.json").exists()
    assert not (output / "admission.json").exists()


def test_parent_observed_withdrawal_refuses_before_preflight(admit, selected, monkeypatch):
    def withdraw(manifest, approval):
        (selected[0].parent / "WITHDRAWN.json").write_bytes(b"withdrawn")

    monkeypatch.setattr(admission, "preflight", lambda *a: pytest.fail("preflight called"))
    with pytest.raises(ValueError, match="withdrawn"):
        admit(withdraw)
    assert not (selected[0].parent / "output_directory").exists()
