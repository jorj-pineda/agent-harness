from __future__ import annotations

from pathlib import Path

import pytest

from workspace import compare, snapshot
from workspace.changes import TextCapture, review_diffs
from workspace.core import DEFAULT_IGNORE_GLOBS


def capture(root: Path, *, file_bytes: int = 128_000, total_bytes: int = 2_000_000):
    text = TextCapture(file_bytes, total_bytes)
    hashes = snapshot(root, ignore=DEFAULT_IGNORE_GLOBS, text_capture=text)
    return hashes, text


def test_diff_reviews_add_delete_dirty_edit_and_missing_newlines(tmp_path: Path) -> None:
    (tmp_path / "edit").write_text("user change\nold")
    (tmp_path / "delete").write_text("removed\n")
    (tmp_path / "unchanged").write_text("pre-existing user edit\n")
    before, old = capture(tmp_path)
    (tmp_path / "edit").write_text("user change\nnew")
    (tmp_path / "delete").unlink()
    (tmp_path / "add").write_text("added\n")
    after, new = capture(tmp_path)
    diffs = {d.path: d for d in review_diffs(compare(before, after), old, new, max_bytes=1000)}
    assert set(diffs) == {"add", "delete", "edit"}
    assert diffs["add"].diff == "--- /dev/null\n+++ b/add\n@@ -0,0 +1 @@\n+added\n"
    assert diffs["delete"].diff == "--- a/delete\n+++ /dev/null\n@@ -1 +0,0 @@\n-removed\n"
    assert diffs["edit"].diff == (
        "--- a/edit\n+++ b/edit\n@@ -1,2 +1,2 @@\n user change\n"
        "-old\n\\ No newline at end of file\n+new\n\\ No newline at end of file\n"
    )


def test_diff_omits_sensitive_binary_and_symlink_content(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside secret")
    (root / ".env").write_text("secret before")
    (root / ".env.local").write_text("another secret before")
    (root / "binary").write_bytes(b"\xff")
    (root / "nul").write_bytes(b"a\0b")
    (root / "link").symlink_to(outside)
    before, old = capture(root)
    (root / ".env").write_text("secret after")
    (root / ".env.local").unlink()
    (root / "binary").write_bytes(b"\xfe")
    (root / "nul").write_text("now text")
    (root / "link").unlink()
    (root / "link").write_text("replaced link")
    after, new = capture(root)
    diffs = review_diffs(compare(before, after), old, new, max_bytes=1000)
    assert len(diffs) == 5
    assert all(not d.diff and d.reason for d in diffs)
    assert ".env" not in old.texts and ".env.local" not in old.texts
    assert "link" not in old.texts
    assert "secret" not in repr(diffs)
    assert "outside secret" not in repr(old)


@pytest.mark.parametrize("file_bytes,total_bytes", [(4, 100), (100, 4)])
def test_capture_limits_do_not_hide_changed_paths(
    tmp_path: Path,
    file_bytes: int,
    total_bytes: int,
) -> None:
    (tmp_path / "a").write_text("1234")
    (tmp_path / "b").write_text("12345")
    before, old = capture(tmp_path, file_bytes=file_bytes, total_bytes=total_bytes)
    (tmp_path / "b").write_text("67890")
    after, new = capture(tmp_path, file_bytes=file_bytes, total_bytes=total_bytes)
    changes = compare(before, after)
    assert changes.modified == ("b",)
    assert old.bytes_used <= total_bytes and new.bytes_used <= total_bytes
    diff = review_diffs(changes, old, new, max_bytes=1000)[0]
    assert diff.diff == ""
    assert diff.reason == "Text capture byte limit exceeded"


def test_output_budget_counts_utf8_and_keeps_later_small_diff(tmp_path: Path) -> None:
    before, old = capture(tmp_path)
    (tmp_path / "a").write_text("é" * 100)
    (tmp_path / "b").write_text("x\n")
    after, new = capture(tmp_path)
    diffs = review_diffs(compare(before, after), old, new, max_bytes=80)
    assert diffs[0].diff == ""
    assert diffs[0].reason == "Diff output byte limit exceeded"
    assert "+x\n" in diffs[1].diff
    assert sum(len(d.diff.encode("utf-8")) for d in diffs) <= 80
    exact = len(diffs[1].diff.encode("utf-8"))
    only_small = compare({}, {"b": after["b"]})
    assert review_diffs(only_small, old, new, max_bytes=exact)[0].diff == diffs[1].diff
    assert review_diffs(only_small, old, new, max_bytes=exact - 1)[0].reason


def test_empty_files_and_control_characters_in_paths(tmp_path: Path) -> None:
    (tmp_path / "deleted_empty").write_text("")
    before, old = capture(tmp_path)
    (tmp_path / "added_empty").write_text("")
    (tmp_path / "deleted_empty").unlink()
    (tmp_path / "name\nwith\tcontrols").write_text("hello\n")
    after, new = capture(tmp_path)
    diffs = review_diffs(compare(before, after), old, new, max_bytes=1000)
    assert diffs[0].reason == diffs[1].reason == "Empty file added or deleted"
    assert '+++ "b/name\\nwith\\tcontrols"\n' in diffs[2].diff


def test_snapshot_does_not_open_symlink_swapped_in_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import workspace.changes as changes

    root = tmp_path / "repo"
    root.mkdir()
    file = root / "file"
    file.write_text("original")
    secret = tmp_path / "secret"
    secret.write_text("private value")
    read = changes._hash_file

    def swapped(path: Path, limit: int | None):
        file.unlink()
        file.symlink_to(secret)
        return read(path, limit)

    monkeypatch.setattr(changes, "_hash_file", swapped)
    hashes, text = capture(root)
    assert hashes["file"].startswith("unreadable:")
    assert text.omitted["file"] == "File content could not be read safely"
    assert "private value" not in repr(text)
