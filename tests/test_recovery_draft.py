import hashlib
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from harness.config import Settings
from harness.context import TokenBudget
from harness.execution_profile import capture_execution_profile
from harness.recovery_draft import (
    DRAFT_GUIDANCE,
    EVIDENCE_PREFIX,
    RecoveryDraftSelection,
    RecoveryDraftUnavailable,
    build_recovery_draft,
)
from harness.runtime import build_registry
from harness.session_store import SavedTurnResponse, SessionArchive
from harness.state import Session, Turn, TurnResponse
from memory import FactStore
from providers.base import ChatMessage, ToolCall
from tests.test_recovery_context import fixture, selection


def prepared(tmp_path):
    root, store, record, _, span = fixture(tmp_path)
    earlier = Turn(
        user_input="Preserve all CRLF bytes.\nKeep my changes 🙂", finished_at=datetime.now(UTC)
    )
    latest = Turn(user_input=record.request.message, finished_at=datetime.now(UTC))
    archive = SessionArchive(
        session=Session(
            session_id=record.request.session_id,
            user_id="dev",
            workspace_root=str(root),
            turns=[earlier, latest],
            messages=[
                ChatMessage(role="system", content="Old system data should be replaced"),
                ChatMessage(role="user", content=earlier.user_input),
                ChatMessage(role="assistant", content="Previous agreement"),
                ChatMessage(role="user", content=latest.user_input),
                ChatMessage(role="assistant", tool_calls=[ToolCall(id="r", name="read_file")]),
                ChatMessage(
                    role="tool",
                    tool_call_id="r",
                    tool_name="read_file",
                    content="old raw payload" * 3000,
                ),
                ChatMessage(role="user", content="Runtime recovery guidance; run the check"),
                ChatMessage(role="assistant", content="Incomplete"),
            ],
        ),
        responses=[
            SavedTurnResponse(
                turn_id=earlier.turn_id,
                response=TurnResponse(answer="Previous agreement", provider="fake", latency_ms=0),
            ),
            SavedTurnResponse(
                turn_id=latest.turn_id, response=record.response.model_copy(deep=True)
            ),
        ],
    )
    request = RecoveryDraftSelection(
        **selection(record, [span]).model_dump(), message="Finish the original bounded task"
    )
    return root, store, record, archive, request


def draft(store, record, archive, request, facts, **settings):
    configured = Settings(_env_file=None, max_context_tokens=16384, **settings)
    registry = build_registry(
        fact_store=facts,
        user_id="dev",
        workspace_root=archive.session.workspace_root,
        coding_toolset=configured.coding_toolset,
        inspection_only=archive.session.inspection_only,
    )
    return build_recovery_draft(
        store,
        record,
        archive,
        request,
        settings=configured,
        provider="ollama",
        registry=registry,
        fact_store=facts,
        user_id="dev",
    )


@pytest.mark.parametrize("prompt", ["legacy", "compact"])
@pytest.mark.parametrize("toolset", ["full", "whole_file"])
def test_exact_user_history_and_current_shared_prompt_evidence_and_estimates(
    tmp_path, prompt, toolset
):
    root, store, record, archive, request = prepared(tmp_path)
    (root / "AGENTS.md").write_bytes(b"Keep conventions.\r\n")
    before = archive.model_dump_json(), record.model_dump_json()
    with FactStore(tmp_path / "facts.db") as facts:
        facts.add("dev", "current owner note", workspace_root=str(root))
        result = draft(
            store,
            record,
            archive,
            request,
            facts,
            load_root_agents=True,
            coding_prompt_profile=prompt,
            coding_toolset=toolset,
            project_check_argv=["pytest", "-q"],
            max_request_output_tokens=128,
            prompt_estimate_bytes_per_token=4.5,
        )
        assert [m.content for m in result.messages[1:4]] == [
            m.content for m in archive.session.messages if m.role == "user"
        ]
        system = result.messages[0].content
        assert "current owner note" in system and "Keep conventions.\\r\\n" in system
        assert '["pytest", "-q"]' in system and DRAFT_GUIDANCE in system
        assert "Old system" not in system
        assert result.repository_instructions.status == "loaded"
        assert result.messages[-1].content == request.message
        evidence = result.messages[result.evidence_message_index]
        assert evidence.content.startswith(EVIDENCE_PREFIX)
        packet = json.loads(evidence.content.removeprefix(EVIDENCE_PREFIX))
        assert packet["review"]["verification_status"] == "failed"
        assert packet["review"]["completion_status"] == "budget_exhausted"
        assert packet["spans"][0]["content"] == "saved\r\n"
        assert not any(
            m.role in {"tool", "assistant"} or m.tool_calls or m.continuation
            for m in result.messages
        )
        assert "old raw payload" not in result.model_dump_json()
        assert result.source_message_count == 8 and result.retained_user_message_count == 3
        assert result.omitted_nonuser_message_count == 5
        assert (
            result.execution_supported is False and result.budget_scope == "hypothetical-new-turn"
        )
        assert result.configuration.status == "not_recorded"
        assert (
            result.context.status == "estimated_fit"
            and result.context.request_output_allowance == 128
        )
        assert (
            result.messages_sha256
            == hashlib.sha256(
                "\n".join(m.model_dump_json() for m in result.messages).encode()
            ).hexdigest()
        )
        registry = build_registry(
            fact_store=facts, user_id="dev", workspace_root=str(root), coding_toolset=toolset
        )
        budget = TokenBudget(max_context=16384, max_request_output=128, bytes_per_token=4.5)
        assert result.context.estimated_prompt_tokens == budget.estimate_prompt(
            result.messages, registry.as_tool_specs()
        )
        assert (
            result.context.estimated_prompt_tokens < result.refreshed_full_history_estimated_tokens
        )
        assert facts.list("dev", workspace_root=str(root))[0].fact == "current owner note"
    assert before == (archive.model_dump_json(), record.model_dump_json())
    assert (root / "a.txt").read_bytes() == b"saved\r\nsecond\nlast"
    store.close()


@pytest.mark.parametrize(
    "max_total,request_cap,expected", [(50, 0, "blocked"), (64000, 1, "estimated_fit")]
)
def test_total_and_request_caps_use_shared_gate_without_granting_old_allowances(
    tmp_path, max_total, request_cap, expected
):
    _, store, record, archive, request = prepared(tmp_path)
    with FactStore(tmp_path / "facts.db") as facts:
        result = draft(
            store,
            record,
            archive,
            request,
            facts,
            max_total_tokens_per_turn=max_total,
            max_request_output_tokens=request_cap,
        )
    assert result.context.status == expected
    assert result.context.request_output_allowance == (None if expected == "blocked" else 1)
    assert result.execution_supported is False
    store.close()


def test_disabled_context_and_ollama_effective_window_are_explicit(tmp_path):
    _, store, record, archive, request = prepared(tmp_path)
    with FactStore(tmp_path / "facts.db") as facts:
        registry = build_registry(fact_store=facts, user_id="dev", workspace_root=None)
        for window, expected in [(None, "unbounded_context"), (256, "blocked")]:
            result = build_recovery_draft(
                store,
                record,
                archive,
                request,
                settings=Settings(_env_file=None, max_context_tokens=0, ollama_num_ctx=window),
                provider="ollama",
                registry=registry,
                fact_store=facts,
                user_id="dev",
            )
            assert result.context.status == expected
            assert result.context.max_context_tokens == (window or 0)
    store.close()


@pytest.mark.parametrize(
    "change,code",
    [
        ("owner", "archive_mismatch"),
        ("session", "archive_mismatch"),
        ("turn", "archive_mismatch"),
        ("response", "archive_mismatch"),
        ("input", "archive_mismatch"),
        ("missing_input", "archive_mismatch"),
        ("unfinished", "archive_mismatch"),
        ("invalid", "invalid_transcript"),
        ("count", "user_history_too_large"),
        ("bytes", "user_history_too_large"),
    ],
)
def test_source_integrity_and_whole_user_retention_refusals_precede_blobs(
    tmp_path, monkeypatch, change, code
):
    _, store, record, archive, request = prepared(tmp_path)
    if change == "owner":
        archive.session.user_id = "other"
    elif change == "session":
        archive.session.session_id = "other"
    elif change == "turn":
        archive.responses[-1].turn_id = "other"
    elif change == "response":
        archive.responses[-1].response.answer = "other"
    elif change == "input":
        archive.session.turns[-1].user_input = "other"
    elif change == "missing_input":
        archive.session.messages[3].content = "other"
    elif change == "unfinished":
        archive.session.turns[-1].finished_at = None
    elif change == "invalid":
        archive.session.messages.append(ChatMessage(role="tool", tool_call_id="orphan"))
    elif change == "count":
        archive.session.messages += [ChatMessage(role="user", content="x")] * 65
    else:
        archive.session.messages.append(ChatMessage(role="user", content="🙂" * 8193))
    monkeypatch.setattr(
        store, "checkpoint_content", lambda *args: pytest.fail("No source blob read")
    )
    with (
        FactStore(tmp_path / "facts.db") as facts,
        pytest.raises(RecoveryDraftUnavailable, match=code),
    ):
        draft(store, record, archive, request, facts)
    store.close()


def test_unavailable_current_instructions_refuse_without_partial_draft(tmp_path):
    root, store, record, archive, request = prepared(tmp_path)
    (root / "AGENTS.md").write_bytes(b"x" * 8193)
    with (
        FactStore(tmp_path / "facts.db") as facts,
        pytest.raises(RecoveryDraftUnavailable, match="instructions_unavailable"),
    ):
        draft(store, record, archive, request, facts, load_root_agents=True)
    store.close()


def test_inspection_registry_and_prompt_are_not_upgraded(tmp_path):
    _, store, record, archive, request = prepared(tmp_path)
    archive.session = archive.session.model_copy(update={"inspection_only": True})
    with FactStore(tmp_path / "facts.db") as facts:
        result = draft(store, record, archive, request, facts, project_check_argv=["pytest"])
        assert "Inspection-only session" in result.messages[0].content
        assert "After editing, run this project check" not in result.messages[0].content
        settings = Settings(_env_file=None, max_context_tokens=16384, project_check_argv=["pytest"])
        registry = build_registry(
            fact_store=facts,
            user_id="dev",
            workspace_root=archive.session.workspace_root,
            inspection_only=True,
        )
        assert (
            result.tool_specs_sha256
            == capture_execution_profile(
                settings, "ollama", registry, inspection_only=True
            ).tool_specs_sha256
        )
    store.close()


@pytest.mark.parametrize("message", ["", "   ", "🙂" * 1025])
def test_proposed_message_is_whole_nonblank_bounded_utf8(message):
    with pytest.raises(ValidationError):
        RecoveryDraftSelection(expected_record_sha256="0" * 64, spans=[], message=message)


def test_escaped_full_draft_limit_refuses_instead_of_truncating_user_text(tmp_path):
    _, store, record, archive, request = prepared(tmp_path)
    archive.session.messages.append(ChatMessage(role="user", content="\x01" * 30000))
    with (
        FactStore(tmp_path / "facts.db") as facts,
        pytest.raises(RecoveryDraftUnavailable, match="draft_too_large"),
    ):
        draft(store, record, archive, request, facts)
    store.close()


def test_small_history_can_make_a_larger_draft_without_hiding_the_estimates(tmp_path):
    _, store, record, archive, request = prepared(tmp_path)
    archive.session.messages[5].content = "short result"
    with FactStore(tmp_path / "facts.db") as facts:
        result = draft(store, record, archive, request, facts)
    assert result.context.estimated_prompt_tokens > result.refreshed_full_history_estimated_tokens
    assert result.execution_supported is False
    store.close()


@pytest.mark.parametrize("boundary", ["bytes", "count"])
def test_exact_user_retention_bounds_preserve_every_message(tmp_path, boundary):
    _, store, record, archive, request = prepared(tmp_path)
    users = [m for m in archive.session.messages if m.role == "user"]
    if boundary == "bytes":
        remaining = 32768 - sum(len(m.content.encode()) for m in users)
        archive.session.messages.append(ChatMessage(role="user", content="a" * remaining))
    else:
        archive.session.messages += [
            ChatMessage(role="user", content="x") for _ in range(64 - len(users))
        ]
    expected = [m.content for m in archive.session.messages if m.role == "user"]
    with FactStore(tmp_path / "facts.db") as facts:
        result = draft(store, record, archive, request, facts)
    assert [m.content for m in result.messages[1 : result.evidence_message_index]] == expected
    store.close()


def test_configured_profile_drift_is_visible_and_does_not_relabel_source(tmp_path):
    _, store, record, archive, request = prepared(tmp_path)
    with FactStore(tmp_path / "facts.db") as facts:
        settings = Settings(_env_file=None, max_context_tokens=16384)
        registry = build_registry(
            fact_store=facts, user_id="dev", workspace_root=archive.session.workspace_root
        )
        profile = capture_execution_profile(settings, "ollama", registry)
        record.provider = "ollama"
        record.configured_model = profile.configured_model
        record.execution_profile = profile
        record.response.provider = "ollama"
        archive.responses[-1].response = record.response.model_copy(deep=True)
        archive.responses[-1].execution_profile = profile.model_copy(deep=True)
        archive.responses[-1].configured_model = record.configured_model
        request = request.model_copy(
            update={
                "expected_record_sha256": hashlib.sha256(
                    record.model_dump_json().encode()
                ).hexdigest()
            }
        )
        before = record.model_dump_json(), archive.model_dump_json()
        result = draft(store, record, archive, request, facts, coding_prompt_profile="compact")
        assert result.configuration.status == "changed"
        assert "prompt_version" in result.configuration.changed_fields
        assert (record.model_dump_json(), archive.model_dump_json()) == before
    store.close()
