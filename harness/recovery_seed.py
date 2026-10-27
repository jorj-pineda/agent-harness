"""Projected recovery fork seeds and bounded provenance; no execution authorization."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from .recovery_admission import RecoveryApproval
from .recovery_draft import RecoveryDraft
from .recovery_identity import RecoverySource, recovery_messages_sha256
from .runs import RunRecord
from .session_store import SessionArchive
from .state import ContinuationSource, Session


def build_recovery_seed(
    record: RunRecord,
    archive: SessionArchive,
    approval: RecoveryApproval,
    draft: RecoveryDraft,
    *,
    fork_session_id: str,
) -> Session:
    """Prepare state for atomic persistence. Fresh execution admission remains required."""
    if (
        not fork_session_id
        or fork_session_id == archive.session.session_id
        or not archive.session.turns
        or not archive.responses
        or record.request.session_id != archive.session.session_id
        or record.request.user_id != archive.session.user_id
        or draft.run_id != record.run_id
        or record.response is None
        or archive.responses[-1].response != record.response
        or record.response.run_id != record.run_id
        or hashlib.sha256(record.model_dump_json().encode()).hexdigest()
        != approval.expected_record_sha256
        or hashlib.sha256(archive.model_dump_json().encode()).hexdigest()
        != approval.expected_archive_sha256
        or draft.source_record_sha256 != approval.expected_record_sha256
        or draft.source_archive_sha256 != approval.expected_archive_sha256
        or draft.messages_sha256 != approval.expected_messages_sha256
        or recovery_messages_sha256(draft.messages) != approval.expected_messages_sha256
        or draft.tool_specs_sha256 != approval.expected_tool_specs_sha256
        or draft.context.request_output_allowance != approval.expected_request_output_allowance
        or not draft.messages
        or draft.messages[-1].role != "user"
        or draft.messages[-1].content != approval.message
    ):
        raise ValueError("Recovery seed/source/approval identity mismatch")
    payload = approval.model_dump_json()
    provenance = RecoverySource(
        source_run_id=record.run_id,
        source_session_id=record.request.session_id,
        source_turn_id=archive.session.turns[-1].turn_id,
        source_record_sha256=approval.expected_record_sha256,
        source_archive_sha256=approval.expected_archive_sha256,
        messages_sha256=approval.expected_messages_sha256,
        tool_specs_sha256=approval.expected_tool_specs_sha256,
        request_output_allowance=approval.expected_request_output_allowance,
        approval_json=payload,
        approval_sha256=hashlib.sha256(payload.encode()).hexdigest(),
    )
    return Session(
        **{
            **archive.session.model_dump(),
            "session_id": fork_session_id,
            "created_at": datetime.now(UTC),
            "checkpoint_turns": True,
            "continuation_source": ContinuationSource(
                session_id=record.request.session_id,
                turn_id=provenance.source_turn_id,
                acknowledge_uncaptured_state=True,
            ),
            "recovery_source": provenance,
            # The runtime appends this exact proposed message once for the new turn.
            "messages": [m.model_copy(deep=True) for m in draft.messages[:-1]],
        }
    )
