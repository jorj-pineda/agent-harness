import hashlib

import pytest
from pydantic import ValidationError

from harness.checkpoint_spans import (
    CheckpointSpanSelection,
    CheckpointSpanUnavailable,
    read_checkpoint_span,
)
from harness.checkpoints import capture_checkpoint
from harness.session_store import SessionStore
from harness.state import Session


def saved(tmp_path, raw):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_bytes(raw)
    session = Session(user_id="dev", workspace_root=str(root))
    capture = capture_checkpoint(root, session.session_id)
    store = SessionStore(tmp_path / "sessions.db")
    store.save(session)
    store.save_checkpoint(capture)
    selection = CheckpointSpanSelection(
        path="a.txt", expected_sha256=hashlib.sha256(raw).hexdigest(), start_line=1, end_line=1
    )
    return root, store, capture.record, selection


def test_exact_crlf_unicode_final_terminators_and_restart_without_workspace(tmp_path):
    raw = "one\r\n🙂two\nlast".encode()
    root, store, record, selection = saved(tmp_path, raw)
    selection = selection.model_copy(update={"start_line": 2, "end_line": 3})
    before = store.get_checkpoint(record.checkpoint_id)
    first = read_checkpoint_span(store, record, selection)
    assert first.content == "🙂two\nlast"
    assert first.content_sha256 == hashlib.sha256("🙂two\nlast".encode()).hexdigest()
    assert first.file_sha256 == hashlib.sha256(raw).hexdigest()
    assert first.total_lines == 3 and not first.has_more
    assert first.selected_bytes == len(first.content.encode())
    store.close()
    (root / "a.txt").unlink()
    root.rmdir()
    restarted = SessionStore(tmp_path / "sessions.db")
    assert read_checkpoint_span(restarted, record, selection) == first
    assert restarted.get_checkpoint(record.checkpoint_id) == before
    restarted.close()


@pytest.mark.parametrize(
    "path",
    [
        "../a.txt",
        "/a.txt",
        "a/../b",
        "a//b",
        "./a",
        "a\\b",
        ".",
        ".env",
        "a/.env.local",
        ".git/config",
        "a\x00b",
    ],
)
def test_invalid_and_excluded_paths_reject(path):
    with pytest.raises(ValidationError):
        CheckpointSpanSelection(path=path, expected_sha256="a" * 64, start_line=1, end_line=1)


@pytest.mark.parametrize("start,end", [(True, 1), (1, False), (1.0, 2), (0, 1), (3, 2), (1, 201)])
def test_strict_ordered_bounded_ranges(start, end):
    with pytest.raises(ValidationError):
        CheckpointSpanSelection(path="a", expected_sha256="a" * 64, start_line=start, end_line=end)


@pytest.mark.parametrize(
    "change,code",
    [("hash", "hash_mismatch"), ("missing", "file_not_captured"), ("large", "file_too_large")],
)
def test_metadata_refusals_do_not_allocate_blob(tmp_path, monkeypatch, change, code):
    _, store, record, selection = saved(tmp_path, b"one\n")
    if change == "hash":
        selection = selection.model_copy(update={"expected_sha256": "0" * 64})
    elif change == "missing":
        selection = selection.model_copy(update={"path": "missing"})
    else:
        record.files["a.txt"].size = 1_048_577

    def forbidden(*args):
        pytest.fail("Must reject before blob read")

    monkeypatch.setattr(store, "checkpoint_content", forbidden)
    with pytest.raises(CheckpointSpanUnavailable) as exc:
        read_checkpoint_span(store, record, selection)
    assert exc.value.code == code
    store.close()


@pytest.mark.parametrize(
    "raw,end,code",
    [
        (b"", 1, "range_unavailable"),
        (b"one\n", 2, "range_unavailable"),
        (b"\xff", 1, "invalid_utf8"),
        (b"a" * 8193, 1, "span_too_large"),
        (b"\x01" * 8192, 1, "response_too_large"),
    ],
)
def test_no_partial_text_on_range_encoding_or_size_failure(tmp_path, raw, end, code):
    _, store, record, selection = saved(tmp_path, raw)
    selection = selection.model_copy(update={"end_line": end})
    with pytest.raises(CheckpointSpanUnavailable) as exc:
        read_checkpoint_span(store, record, selection)
    assert exc.value.code == code
    store.close()


def test_corrupt_blob_rejects_even_when_selected_prefix_unchanged(tmp_path):
    _, store, record, selection = saved(tmp_path, b"one\ntwo\n")
    store._conn.execute("UPDATE checkpoint_files SET content=?", (b"one\nbad\n",))
    store._conn.commit()
    with pytest.raises(CheckpointSpanUnavailable, match="content_unavailable"):
        read_checkpoint_span(store, record, selection)
    store.close()


def test_invalid_bytes_outside_selected_range_are_not_lossily_returned(tmp_path):
    _, store, record, selection = saved(tmp_path, b"one\r\n\xff")
    result = read_checkpoint_span(store, record, selection)
    assert result.content == "one\r\n"
    assert result.total_lines == 2 and result.has_more
    store.close()
