"""Ignored-entry metadata is bounded evidence, not a subtree checksum."""

from pathlib import Path

import pytest

from harness.workspace_review import (
    WorkspaceReviewBaseline,
    compare_ignored_entries,
    compare_workspace,
)
from workspace import snapshot
from workspace.core import DEFAULT_IGNORE_GLOBS


def capture(root: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    snapshot(root, ignore=DEFAULT_IGNORE_GLOBS, strict=True, ignored_entries=entries)
    return entries


def test_capture_does_not_read_contents_or_follow_ignored_symlinks(tmp_path, monkeypatch):
    import workspace.changes as changes

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / ".git" / "index").write_text("user index; never hashed")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private").write_text("not inspected")
    (repo / ".venv").symlink_to(outside)
    (repo / ".env.local").mkdir()
    (repo / ".env.local" / "secret").write_text("fixture secret")
    original_walk = changes.os.walk
    visited = []

    def walk(*args, **kwargs):
        for entry in original_walk(*args, **kwargs):
            visited.append(Path(entry[0]))
            yield entry

    def no_reads(*args, **kwargs):
        pytest.fail("Ignored contents must not be opened")

    monkeypatch.setattr(changes.os, "walk", walk)
    monkeypatch.setattr(changes, "_hash_file", no_reads)
    entries = {}
    assert (
        snapshot(
            repo, ignore=(*DEFAULT_IGNORE_GLOBS, ".env.*"), strict=True, ignored_entries=entries
        )
        == {}
    )
    assert set(entries) == {".git", ".venv"}
    assert all(len(value) == 64 for value in entries.values())
    assert visited == [repo]
    assert "fixture secret" not in str(entries)
    assert str(outside) not in str(entries)
    (outside / "private").write_text("referent content changed")
    again = {}
    snapshot(repo, ignore=(*DEFAULT_IGNORE_GLOBS, ".env.*"), strict=True, ignored_entries=again)
    assert again == entries


def test_direct_entry_change_is_detected_but_existing_child_rewrite_is_not(tmp_path):
    directory = tmp_path / ".git"
    directory.mkdir()
    child = directory / "index"
    child.write_text("old index")
    baseline = capture(tmp_path)
    child.write_text("rewritten child")
    assert capture(tmp_path) == baseline  # Intentional documented blind spot.
    (directory / "new-entry").write_text("new")
    result = compare_ignored_entries(baseline, capture(tmp_path))
    assert result.status == "changed"
    assert result.modified == [".git"]
    assert "rewriting existing children" in result.scope_note


def test_metadata_reports_create_remove_replace_with_capped_paths(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".venv").mkdir()
    baseline = capture(tmp_path)
    (tmp_path / ".git").rename(tmp_path / "old-git")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".venv").rmdir()
    (tmp_path / "node_modules").mkdir()
    result = compare_ignored_entries(baseline, capture(tmp_path), max_paths=1)
    assert result.status == "changed"
    assert result.added == ["node_modules"]
    assert result.modified == [] and result.deleted == []
    assert result.added_count == result.modified_count == result.deleted_count == 1
    assert result.paths_truncated
    assert compare_ignored_entries(None, capture(tmp_path)).status == "not_recorded"


def test_capture_uses_only_fixed_ignored_root_names_and_active_exclusions(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "arbitrary").mkdir()
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / ".venv").mkdir()
    entries = {}
    snapshot(tmp_path, ignore=("arbitrary", ".venv"), strict=True, ignored_entries=entries)
    assert entries == {}  # .git isn't excluded; nested .venv and custom name aren't captured.


def test_missing_legacy_metadata_stays_unknown_and_failures_skip_comparison(tmp_path, monkeypatch):
    repo = tmp_path.resolve()
    (repo / ".git").mkdir()
    info = repo.stat()
    baseline = WorkspaceReviewBaseline(
        status="recorded", root=str(repo), root_identity=(info.st_dev, info.st_ino), fingerprints={}
    )
    review = compare_workspace(baseline, max_files=10, max_bytes=100)
    assert review.status == "unchanged"
    assert review.ignored_entries.status == "not_recorded"
    baseline.ignored_entry_fingerprints = capture(repo)
    original = Path.lstat

    def denied(path, *args, **kwargs):
        if path == repo / ".git":
            raise PermissionError("do not expose this private failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", denied)
    review = compare_workspace(baseline, max_files=10, max_bytes=100)
    assert review.status == "unavailable"
    assert review.ignored_entries.status == "not_checked"
    assert "PermissionError" in review.reason
    assert "private failure" not in review.reason


def test_workspace_path_cap_is_shared_with_metadata_lists(tmp_path):
    from harness.workspace_review import review_fingerprints

    repo = tmp_path.resolve()
    (repo / "a.py").write_text("before")
    (repo / ".git").mkdir()
    info = repo.stat()
    baseline = WorkspaceReviewBaseline(
        status="recorded",
        root=str(repo),
        root_identity=(info.st_dev, info.st_ino),
        fingerprints=review_fingerprints(snapshot(repo, ignore=DEFAULT_IGNORE_GLOBS, strict=True)),
        ignored_entry_fingerprints=capture(repo),
    )
    (repo / "a.py").write_text("after")
    (repo / ".git" / "new-entry").write_text("changed")
    review = compare_workspace(baseline, max_files=10, max_bytes=100, max_paths=1)
    assert review.modified == ["a.py"]
    assert review.modified_count == 1
    assert review.ignored_entries.modified == []
    assert review.ignored_entries.modified_count == 1
    assert review.ignored_entries.status == "changed"
    assert review.ignored_entries.paths_truncated is True
