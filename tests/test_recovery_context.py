import hashlib

import pytest
from pydantic import ValidationError

from harness.checkpoint_spans import CheckpointSpanSelection, CheckpointSpanUnavailable
from harness.checkpoints import capture_checkpoint
from harness.recovery_context import (
    RecoveryContextSelection,
    RecoveryContextUnavailable,
    build_recovery_context,
)
from harness.runs import RunInput
from harness.session_store import SessionStore
from harness.state import (
    CheckRecord,
    Session,
    TurnResponse,
    WorkspaceChangeReport,
    WorkspaceFileDiff,
)


def fixture(tmp_path, raw=b"saved\r\nsecond\nlast"):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_bytes(raw)
    session = Session(user_id="dev", workspace_root=str(root))
    store = SessionStore(tmp_path / "sessions.db")
    store.save(session)
    record, _ = store.create_run(
        RunInput(user_id="dev", session_id=session.session_id, request_id="r", message="Fix"),
        provider="fake",
        configured_model=None,
    )
    store.bind_run_checkpoint(record.run_id, capture_checkpoint(root, session.session_id), "before")
    captured = capture_checkpoint(root, session.session_id)
    store.bind_run_checkpoint(record.run_id, captured, "after")
    store.finish_run(
        record.run_id,
        TurnResponse(
            run_id=record.run_id,
            answer="",
            provider="fake",
            latency_ms=0,
            completion_status="budget_exhausted",
            verification_status="failed",
            check_attempts=[CheckRecord(argv=["pytest"], status="failed", relevant=True)],
            workspace_changes=WorkspaceChangeReport(
                status="tracked", diffs=[WorkspaceFileDiff(path="a.txt", diff="-old\n+saved")]
            ),
        ),
    )
    record = store.get_run(record.run_id)
    checkpoint = store.get_checkpoint(record.after_checkpoint_id)
    span = CheckpointSpanSelection(
        path="a.txt", expected_sha256=hashlib.sha256(raw).hexdigest(), start_line=1, end_line=1
    )
    return root, store, record, checkpoint, span


def selection(record, spans):
    return RecoveryContextSelection(
        expected_record_sha256=hashlib.sha256(record.model_dump_json().encode()).hexdigest(),
        spans=spans,
    )


def test_join_retains_failures_exact_source_and_identity_after_restart(tmp_path):
    root, store, record, checkpoint, span = fixture(tmp_path)
    before = record.model_dump_json()
    request = selection(record, [span])
    first = build_recovery_context(store, record, request)
    assert first.spans[0].content == "saved\r\n"
    assert first.review.verification_status == "failed"
    assert first.review.completion_status == "budget_exhausted"
    assert first.review.diffs[0].diff.text == "-old\n+saved"
    assert first.selected_source_bytes == 7
    assert first.execution_supported is False
    evidence = first.review.model_dump_json() + "\n" + first.spans[0].model_dump_json()
    assert first.evidence_sha256 == hashlib.sha256(evidence.encode()).hexdigest()
    store.close()
    (root / "a.txt").write_text("later user work")
    root.rename(tmp_path / "moved")
    restarted = SessionStore(tmp_path / "sessions.db")
    assert build_recovery_context(restarted, record, request) == first
    assert restarted.get_checkpoint(checkpoint.checkpoint_id) == checkpoint
    restarted.close()
    assert record.model_dump_json() == before
    assert (tmp_path / "moved/a.txt").read_text() == "later user work"


@pytest.mark.parametrize("status", ["failed", "interrupted", "cancelled"])
def test_review_only_without_checkpoint_preserves_unknowns(tmp_path, status):
    _, store, record, _, _ = fixture(tmp_path)
    record.status = status
    record.response = None
    record.after_checkpoint_id = None
    result = build_recovery_context(store, record, selection(record, []))
    assert result.spans == [] and result.selected_source_bytes == 0
    assert result.review.action_count is result.review.diff_count is None
    assert result.review.final_response_available is False
    store.close()


@pytest.mark.parametrize(
    "change,code",
    [
        ("active", "run_active"),
        ("hash", "record_hash_mismatch"),
        ("missing", "checkpoint_mismatch"),
        ("session", "checkpoint_mismatch"),
        ("run", "checkpoint_mismatch"),
        ("kind", "checkpoint_mismatch"),
        ("pair", "checkpoint_mismatch"),
    ],
)
def test_identity_rejections_precede_blob_reads(tmp_path, monkeypatch, change, code):
    _, store, record, checkpoint, span = fixture(tmp_path)
    if change == "active":
        record.status = "running"
    elif change == "missing":
        record.after_checkpoint_id = None
    elif change == "session":
        checkpoint.session_id = "foreign"
    elif change == "run":
        checkpoint.run_id = "foreign"
    elif change == "kind":
        checkpoint.kind = "manual"
    elif change == "pair":
        checkpoint.paired_before_id = "foreign"
    request = selection(record, [span])
    if change == "hash":
        request = request.model_copy(update={"expected_record_sha256": "0" * 64})
    monkeypatch.setattr(store, "get_checkpoint", lambda _: checkpoint)
    monkeypatch.setattr(store, "checkpoint_content", lambda *args: pytest.fail("No blob read"))
    with pytest.raises(RecoveryContextUnavailable, match=code):
        build_recovery_context(store, record, request)
    store.close()


@pytest.mark.parametrize("ranges", [[(1, 1), (1, 1)], [(1, 3), (2, 4)], [(1, 1)] * 5])
def test_overlapping_or_excess_selections_reject(ranges):
    with pytest.raises(ValidationError):
        RecoveryContextSelection(
            expected_record_sha256="0" * 64,
            spans=[
                {"path": "a", "expected_sha256": "0" * 64, "start_line": start, "end_line": end}
                for start, end in ranges
            ],
        )


def test_aggregate_source_budget_no_partial_bundle(tmp_path):
    raw = (b"a" * 8191 + b"\n") * 3
    _, store, record, _, span = fixture(tmp_path, raw)
    spans = [span.model_copy(update={"start_line": i, "end_line": i}) for i in range(1, 4)]
    exact = build_recovery_context(store, record, selection(record, spans[:2]))
    assert exact.selected_source_bytes == 16384
    with pytest.raises(RecoveryContextUnavailable, match="source_budget_exceeded"):
        build_recovery_context(store, record, selection(record, spans))
    store.close()


def test_late_bad_span_refuses_without_returning_earlier_source(tmp_path):
    _, store, record, _, span = fixture(tmp_path)
    missing = span.model_copy(update={"path": "missing"})
    with pytest.raises(CheckpointSpanUnavailable, match="file_not_captured"):
        build_recovery_context(store, record, selection(record, [span, missing]))
    store.close()


def test_escaped_aggregate_json_limit_and_oversized_record(tmp_path):
    raw = (b"\x01" * 4800 + b"\n") * 3
    _, store, record, _, span = fixture(tmp_path, raw)
    spans = [span.model_copy(update={"start_line": i, "end_line": i}) for i in range(1, 4)]
    with pytest.raises(RecoveryContextUnavailable, match="context_too_large"):
        build_recovery_context(store, record, selection(record, spans))
    record.run_id = "x" * 65536
    with pytest.raises(RecoveryContextUnavailable, match="context_too_large"):
        build_recovery_context(store, record, selection(record, []))
    store.close()
