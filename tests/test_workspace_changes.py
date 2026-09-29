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
