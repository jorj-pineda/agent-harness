import hashlib

import pytest

from harness.recovery_review import MAX_PACKET_BYTES, build_recovery_review
from harness.runs import RunInput, RunRecord
from harness.state import (
    CheckRecord,
    ToolCallRecord,
    TurnResponse,
    WorkspaceChangeReport,
    WorkspaceFileDiff,
)


def record(response=None):
    return RunRecord(
        request=RunInput(user_id="u", session_id="s", request_id="r", message="Fix the timer"),
        provider="scripted",
        status="finished",
        response=response,
    )


def response(**kwargs):
    return TurnResponse(answer="", provider="scripted", latency_ms=0, **kwargs)


def test_partial_patch_failed_and_stale_checks_are_retained_without_mutation():
    saved = record(
        response(
            completion_status="budget_exhausted",
            completion_reason="Context prevented next request",
            verification_status="stale",
            check_attempts=[
                CheckRecord(
                    argv=["pytest"], status="failed", relevant=True, superseded_by_edit=True
                ),
                CheckRecord(
                    argv=["ruff"], status="unavailable", relevant=False, error="Missing runner"
                ),
                CheckRecord(argv=["pytest"], status="passed", relevant=True),
            ],
            tool_calls=[
                ToolCallRecord(
                    name="edit_file", arguments={"secret": "do not include"}, result="raw"
                )
            ],
            workspace_changes=WorkspaceChangeReport(
                status="tracked", diffs=[WorkspaceFileDiff(path="a.py", diff="-old\n+new")]
            ),
        )
    )
    before = saved.model_dump_json()
    packet = build_recovery_review(saved)
    assert saved.model_dump_json() == before
    assert packet == build_recovery_review(saved)
    assert packet.source_record_sha256 == hashlib.sha256(before.encode()).hexdigest()
    assert packet.completion_status == "budget_exhausted"
    assert packet.verification_status == "stale"
    assert [c.status for c in packet.unresolved_checks] == ["failed", "unavailable"]
    assert packet.unresolved_checks[0].superseded_by_edit
    assert packet.diffs[0].diff.text == "-old\n+new"
    assert packet.recent_actions[0].outcome == "returned"
    assert "do not include" not in packet.model_dump_json()
    assert "arguments" not in packet.model_dump()


def test_absent_response_preserves_unknown_not_empty_work():
    saved = record()
    saved.status = "interrupted"
    saved.error = "Server stopped"
    packet = build_recovery_review(saved)
    assert packet.run_error.text == "Server stopped"
    assert packet.unresolved_check_count is packet.omitted_unresolved_checks is None
    assert not packet.final_response_available
    assert packet.completion_status is packet.verification_status is packet.workspace_status is None
    assert packet.action_count is packet.diff_count is packet.omitted_diffs is None
    assert packet.recent_actions == packet.diffs == []


@pytest.mark.parametrize("text", ["🙂" * 20000, '\x00\n\\"' * 10000])
def test_utf8_escaping_and_item_limits_fit_with_visible_omissions(text):
    saved = record(
        response(
            completion_reason=text,
            tool_calls=[
                ToolCallRecord(name=text, error=text, interrupted=i == 19) for i in range(20)
            ],
            check_attempts=[
                CheckRecord(argv=[text], status="failed", relevant=True, error=text)
                for _ in range(20)
            ],
            workspace_changes=WorkspaceChangeReport(
                status="tracked",
                diffs=[WorkspaceFileDiff(path=text, diff=text, reason=text) for _ in range(20)],
            ),
        )
    )
    saved.request.message = text
    packet = build_recovery_review(saved)
    assert len(packet.model_dump_json().encode()) <= MAX_PACKET_BYTES
    assert packet.objective.truncated and packet.objective.source_bytes == len(text.encode())
    assert packet.objective.source_sha256 == hashlib.sha256(text.encode()).hexdigest()
    assert packet.omitted_actions == packet.omitted_diffs == packet.omitted_unresolved_checks == 12
    assert packet.recent_actions[-1].index == 19
    assert packet.recent_actions[-1].outcome == "interrupted"
    assert packet.diffs[-1].diff.truncated


def test_oversized_legacy_metadata_fails_instead_of_returning_unbounded_packet():
    saved = record()
    saved.run_id = "x" * MAX_PACKET_BYTES
    with pytest.raises(ValueError, match="metadata"):
        build_recovery_review(saved)
