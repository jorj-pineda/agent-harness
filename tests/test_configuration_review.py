from datetime import UTC, datetime
from pathlib import Path

import pytest

from harness.config import Settings
from harness.configuration_review import compare_profiles, review_configuration
from harness.context import TokenBudget
from harness.execution_profile import capture_execution_profile
from harness.runtime import refresh_system_message
from harness.session_store import SavedTurnResponse, SessionArchive
from harness.state import Session, Turn, TurnResponse
from memory import FactStore
from providers.base import ChatMessage, ToolCall
from tools import ToolRegistry

from .test_harness_loop import _echo_tool


def archive_with_profile(settings: Settings, registry: ToolRegistry) -> SessionArchive:
    session = Session(
        user_id="dev",
        messages=[
            ChatMessage(role="user", content="original task"),
            ChatMessage(role="assistant", content="Done"),
        ],
        turns=[
            Turn(user_input="original task", final_answer="Done", finished_at=datetime.now(UTC))
        ],
    )
    profile = capture_execution_profile(settings, "ollama", registry)
    return SessionArchive(
        session=session,
        responses=[
            SavedTurnResponse(
                turn_id=session.turns[0].turn_id,
                configured_model=profile.configured_model,
                execution_profile=profile,
                response=TurnResponse(answer="Done", provider="ollama", latency_ms=1),
            )
        ],
    )


def test_matching_configuration_does_not_grant_resume_permission(tmp_path: Path):
    settings = Settings(_env_file=None, max_context_tokens=16000)
    registry = ToolRegistry()
    registry.register(_echo_tool())
    archive = archive_with_profile(settings, registry)
    saved = archive.model_dump_json()
    with FactStore(tmp_path / "facts.db") as store:
        store.add("dev", "current fixture context")
        result = review_configuration(
            archive,
            settings=settings,
            provider="ollama",
            registry=registry,
            fact_store=store,
            user_id="dev",
            message="proposed task",
        )
        detached = archive.session.model_copy(deep=True)
        refresh_system_message(detached, store, "dev")
        detached.messages.append(ChatMessage(role="user", content="proposed task"))
        budget = TokenBudget(
            max_context=16000, min_request_output=settings.min_request_output_tokens
        )
        assert result.context.estimated_prompt_tokens == budget.estimate_prompt(
            detached.messages, registry.as_tool_specs()
        )
        assert result.context.request_output_allowance == budget.request_allowance(
            detached.messages, registry.as_tool_specs()
        )
    assert result.configuration.status == "matching"
    assert result.context.status == "estimated_fit"
    assert result.context.includes_proposed_message
    assert result.resume_supported is False
    assert result.read_only
    assert archive.model_dump_json() == saved
    assert "current fixture context" not in result.model_dump_json()
    assert "proposed task" not in result.model_dump_json()


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("ollama_model", "different", "configured_model"),
        ("ollama_host", "http://fixture-other:11434", "endpoint_sha256"),
        ("coding_toolset", "whole_file", "settings.coding_toolset"),
        ("max_tool_calls_per_turn", 10, "settings.max_tool_calls_per_turn"),
        ("project_check_argv", ["pytest", "-q"], "project_check_sha256"),
        ("ollama_think", False, "ollama_think"),
    ],
)
def test_drift_reports_names_without_values(field, value, expected):
    registry = ToolRegistry()
    first = capture_execution_profile(Settings(_env_file=None), "ollama", registry)
    current = capture_execution_profile(
        Settings(_env_file=None, **{field: value}), "ollama", registry
    )
    review = compare_profiles(first, current)
    assert review.status == "changed"
    assert expected in review.changed_fields
    assert "fixture-other" not in review.model_dump_json()
    assert "different" not in review.model_dump_json()


def test_equal_unknown_identity_is_not_matching():
    profile = capture_execution_profile(Settings(_env_file=None), "custom", ToolRegistry())
    result = compare_profiles(profile, profile)
    assert result.status == "unknown"
    assert result.unknown_fields == ["configured_model", "endpoint_sha256"]
    assert compare_profiles(None, profile).status == "not_recorded"
    unknown_prompt = profile.model_copy(update={"prompt_version": None})
    assert "prompt_version" in compare_profiles(unknown_prompt, unknown_prompt).unknown_fields


def test_old_whole_file_prompt_label_is_drift_without_rewriting_history():
    settings = Settings(_env_file=None, coding_toolset="whole_file")
    current = capture_execution_profile(settings, "ollama", ToolRegistry())
    assert current.prompt_version == "coding-v2-whole-file-v1"
    saved = current.model_copy(update={"prompt_version": "coding-v2"})
    original = saved.model_dump_json()
    review = compare_profiles(saved, current)
    assert review.status == "changed"
    assert review.changed_fields == ["prompt_version"]
    assert saved.base_prompt_sha256 == current.base_prompt_sha256
    assert saved.model_dump_json() == original


@pytest.mark.parametrize(
    "kind",
    [
        "short_context",
        "short_total",
        "unbounded",
        "history_only",
        "invalid",
        "alignment",
        "identity",
        "legacy",
    ],
)
def test_context_limits_and_archive_problems(tmp_path: Path, kind):
    settings = Settings(
        _env_file=None,
        max_context_tokens=20 if kind == "short_context" else 0,
        max_total_tokens_per_turn=20 if kind == "short_total" else 0,
    )
    registry = ToolRegistry()
    archive = archive_with_profile(settings, registry)
    if kind == "invalid":
        archive.session.messages.append(
            ChatMessage(role="assistant", tool_calls=[ToolCall(id="pending", name="echo")])
        )
    elif kind == "alignment":
        archive.responses[0].turn_id = "other"
    elif kind == "identity":
        archive.responses[0].configured_model = "other"
    elif kind == "legacy":
        archive.responses[0].execution_profile = None
    before = archive.model_dump_json()
    with FactStore(tmp_path / "facts.db") as store:
        result = review_configuration(
            archive,
            settings=settings,
            provider="ollama",
            registry=registry,
            fact_store=store,
            user_id="dev",
            message="" if kind == "history_only" else "next task",
        )
    if kind in {"short_context", "short_total"}:
        assert result.context.status == "blocked"
        assert result.context.request_output_allowance is None
    elif kind in {"alignment", "identity", "invalid"}:
        assert result.context.status == "not_checked"
        assert result.context.estimated_prompt_tokens is None
    else:
        assert result.context.status == "unbounded_context"
        assert result.context.includes_proposed_message == (kind != "history_only")
    if kind == "legacy":
        assert result.configuration.status == "not_recorded"
    assert archive.model_dump_json() == before
