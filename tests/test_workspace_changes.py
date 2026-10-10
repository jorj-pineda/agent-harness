from __future__ import annotations

from pathlib import Path

import pytest

from workspace import SnapshotLimitError, compare, snapshot
from workspace.core import DEFAULT_IGNORE_GLOBS


def test_snapshot_compare_reports_content_changes_only(tmp_path: Path) -> None:
    (tmp_path / "same.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "edit.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "gone.py").write_text("x = 1\n", encoding="utf-8")
    before = snapshot(tmp_path, ignore=DEFAULT_IGNORE_GLOBS)
    (tmp_path / "same.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "edit.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "gone.py").unlink()
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "new.py").write_text("", encoding="utf-8")
    changes = compare(before, snapshot(tmp_path, ignore=DEFAULT_IGNORE_GLOBS))
    assert changes.added == ("pkg/new.py",)
    assert changes.modified == ("edit.py",)
    assert changes.deleted == ("gone.py",)


def test_snapshot_skips_ignored_directories(tmp_path: Path) -> None:
    for ignored in (".git", "__pycache__", "node_modules"):
        (tmp_path / ignored).mkdir()
        (tmp_path / ignored / "file").write_text("x", encoding="utf-8")
    (tmp_path / "kept.txt").write_text("x", encoding="utf-8")
    assert list(snapshot(tmp_path, ignore=DEFAULT_IGNORE_GLOBS)) == ["kept.txt"]


def test_snapshot_records_symlinks_without_following_them(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("x", encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "linked_dir").symlink_to(outside)
    (root / "linked_file").symlink_to(outside / "secret.txt")
    before = snapshot(root, ignore=DEFAULT_IGNORE_GLOBS)
    assert set(before) == {"linked_dir", "linked_file"}
    (root / "linked_file").unlink()
    (root / "linked_file").symlink_to(outside)
    assert compare(before, snapshot(root, ignore=DEFAULT_IGNORE_GLOBS)).modified == ("linked_file",)


def test_snapshot_limits_fail_instead_of_reporting_partial_changes(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"f{index}.txt").write_text("12345", encoding="utf-8")
    with pytest.raises(SnapshotLimitError, match="files"):
        snapshot(tmp_path, ignore=(), max_files=2)
    with pytest.raises(SnapshotLimitError, match="bytes"):
        snapshot(tmp_path, ignore=(), max_bytes=12)
    assert len(snapshot(tmp_path, ignore=(), max_files=3, max_bytes=15)) == 3


def test_strict_snapshot_rejects_missing_root_and_read_failures(tmp_path: Path, monkeypatch):
    import workspace.changes as changes

    with pytest.raises(OSError):
        snapshot(tmp_path / "missing", ignore=(), strict=True)
    (tmp_path / "a").write_text("value")

    def fail(*args, **kwargs):
        raise PermissionError("fixture denied")

    monkeypatch.setattr(changes, "_hash_file", fail)
    assert snapshot(tmp_path, ignore=())["a"].startswith("unreadable:")
    with pytest.raises(PermissionError):
        snapshot(tmp_path, ignore=(), strict=True)


def test_strict_snapshot_rejects_walk_errors(tmp_path: Path, monkeypatch):
    import workspace.changes as changes

    def fail_walk(root, *, followlinks, onerror):
        onerror(PermissionError("fixture traversal denied"))
        return iter(())

    monkeypatch.setattr(changes.os, "walk", fail_walk)
    assert snapshot(tmp_path, ignore=()) == {}
    with pytest.raises(PermissionError):
        snapshot(tmp_path, ignore=(), strict=True)


def test_strict_snapshot_stops_reading_a_file_that_grows(tmp_path: Path, monkeypatch):
    import workspace.changes as changes

    path = tmp_path / "growing"
    path.write_text("small")
    original = changes._hash_file

    def grow(path, capture_bytes, **kwargs):
        path.write_text("now much larger")
        return original(path, capture_bytes, **kwargs)

    monkeypatch.setattr(changes, "_hash_file", grow)
    with pytest.raises(SnapshotLimitError, match="grew"):
        snapshot(tmp_path, ignore=(), max_bytes=5, strict=True)


def test_optional_modes_use_opened_regular_file_and_do_not_follow_links(tmp_path: Path):
    file = tmp_path / "script"
    file.write_text("echo hi\n")
    file.chmod(0o750)
    (tmp_path / "link").symlink_to(file)
    modes = {}
    fingerprints = snapshot(tmp_path, ignore=(), strict=True, file_modes=modes)
    assert modes == {"script": 0o750}
    file.chmod(0o640)
    assert snapshot(tmp_path, ignore=(), strict=True) == fingerprints
