import os
import stat

import pytest
from pydantic import ValidationError

from harness.checkpoints import capture_checkpoint
from harness.reverts import FileRevertInput, revert_file


def pair(root):
    (root / "a.py").write_bytes(b"user edit\n")
    (root / "a.py").chmod(0o755)
    (root / "other.py").write_bytes(b"other user work")
    before = capture_checkpoint(root, "session")
    (root / "a.py").write_bytes(b"user edit\nagent change\n")
    (root / "a.py").chmod(0o640)
    after = capture_checkpoint(root, "session")
    return before, after


def test_single_file_revert_restores_preexisting_edits_and_modes(tmp_path):
    before, after = pair(tmp_path)
    (tmp_path / "other.py").write_bytes(b"later unrelated user edit")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/index").write_bytes(b"index-marker")
    result = revert_file(before.record, after.record, "a.py", before.contents["a.py"])
    assert result.status == "applied"
    assert (tmp_path / "a.py").read_bytes() == b"user edit\n"
    assert stat.S_IMODE((tmp_path / "a.py").stat().st_mode) == 0o755
    assert (tmp_path / "other.py").read_bytes() == b"later unrelated user edit"
    assert (tmp_path / ".git/index").read_bytes() == b"index-marker"


@pytest.mark.parametrize("change", ["content", "mode"])
def test_later_target_edits_are_preserved(tmp_path, change):
    before, after = pair(tmp_path)
    if change == "content":
        (tmp_path / "a.py").write_bytes(b"later user edit")
    else:
        (tmp_path / "a.py").chmod(0o600)
    raw = (tmp_path / "a.py").read_bytes()
    mode = (tmp_path / "a.py").stat().st_mode
    result = revert_file(before.record, after.record, "a.py", before.contents["a.py"])
    assert result.status == "rejected"
    assert (tmp_path / "a.py").read_bytes() == raw
    assert (tmp_path / "a.py").stat().st_mode == mode


@pytest.mark.parametrize("kind", ["added", "deleted", "empty", "binary"])
def test_eligible_file_shapes_revert_without_whole_workspace_reset(tmp_path, kind):
    target = tmp_path / "a"
    if kind != "added":
        target.write_bytes(
            b"" if kind == "empty" else b"\x00\xff" if kind == "binary" else b"original"
        )
    before = capture_checkpoint(tmp_path, "session")
    if kind == "deleted":
        target.unlink()
    else:
        target.write_bytes(b"changed")
    after = capture_checkpoint(tmp_path, "session")
    result = revert_file(before.record, after.record, "a", before.contents.get("a"))
    assert result.status == "applied"
    if kind == "added":
        assert not target.exists()
    else:
        assert target.read_bytes() == before.contents["a"]


def test_change_during_preparation_rejects_without_overwriting_user(tmp_path, monkeypatch):
    before, after = pair(tmp_path)
    real_chmod = os.fchmod

    def change_during_stage(descriptor, mode):
        (tmp_path / "a.py").write_bytes(b"concurrent user edit")
        real_chmod(descriptor, mode)

    monkeypatch.setattr("harness.reverts.os.fchmod", change_during_stage)
    result = revert_file(before.record, after.record, "a.py", before.contents["a.py"])
    assert result.status == "rejected"
    assert (tmp_path / "a.py").read_bytes() == b"concurrent user edit"
    assert not tuple(tmp_path.glob(".harness-revert-*"))


@pytest.mark.parametrize("path", ["/absolute", "../escape", ".", "a//b", "a/./b", "bad\x00path"])
def test_revert_paths_are_validated(path):
    with pytest.raises(ValidationError):
        FileRevertInput(user_id="dev", run_id="r", request_id="key", path=path)


def test_replaced_root_rejects_and_preserves_new_root_content(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    before, after = pair(root)
    root.rename(tmp_path / "old-repo")
    root.mkdir()
    (root / "a.py").write_bytes(b"new root user work")
    outcome = revert_file(before.record, after.record, "a.py", before.contents["a.py"])
    assert outcome.status == "rejected"
    assert (root / "a.py").read_bytes() == b"new root user work"


def test_parent_symlink_never_leads_revert_into_external_directory(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "sub").mkdir()
    (root / "sub/a").write_bytes(b"original")
    before = capture_checkpoint(root, "session")
    (root / "sub/a").write_bytes(b"after")
    after = capture_checkpoint(root, "session")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a").write_bytes(b"outside user work")
    (root / "sub").rename(root / "old-sub")
    (root / "sub").symlink_to(outside, target_is_directory=True)
    outcome = revert_file(before.record, after.record, "sub/a", before.contents["sub/a"])
    assert outcome.status != "applied"
    assert (outside / "a").read_bytes() == b"outside user work"
