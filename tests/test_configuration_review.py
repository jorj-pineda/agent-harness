from datetime import UTC, datetime
from pathlib import Path

import pytest

from harness.config import Settings
from harness.configuration_review import compare_profiles, review_configuration
from harness.context import TokenBudget
from harness.execution_profile import capture_execution_profile
from harness.prompts import coding_prompt
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


@pytest.mark.parametrize("prompt_profile", ["legacy", "compact"])
@pytest.mark.parametrize("ratio", [3, 4.5])
def test_matching_configuration_does_not_grant_resume_permission(
    tmp_path: Path, prompt_profile, ratio
):
    settings = Settings(
        _env_file=None,
        max_context_tokens=16000,
        coding_prompt_profile=prompt_profile,
        prompt_estimate_bytes_per_token=ratio,
    )
    registry = ToolRegistry()
    registry.register(_echo_tool())
    archive = archive_with_profile(settings, registry)
    saved = archive.model_dump_json()
    with FactStore(tmp_path / "facts.db") as store:
        store.add("dev", "current fixture context")
        for i in range(20):
            store.add("dev", f"{i}:" + "é" * 2000)
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
        refresh_system_message(
            detached,
            store,
            "dev",
            system_prompt=coding_prompt(settings.coding_toolset, settings.coding_prompt_profile),
        )
        from memory.store import MEMORY_OMISSION_NOTICE

        assert MEMORY_OMISSION_NOTICE in detached.messages[0].content
        assert len(store.list("dev")) == 20
        detached.messages.append(ChatMessage(role="user", content="proposed task"))
        budget = TokenBudget(
            max_context=16000,
            min_request_output=settings.min_request_output_tokens,
            bytes_per_token=ratio,
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
        ("coding_prompt_profile", "compact", "prompt_version"),
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
    assert current.prompt_version == "coding-v2-whole-file-canonical-memory-v1"
    saved = current.model_copy(update={"prompt_version": "coding-v2"})
    original = saved.model_dump_json()
    review = compare_profiles(saved, current)
    assert review.status == "changed"
    assert review.changed_fields == ["prompt_version"]
    assert saved.base_prompt_sha256 == current.base_prompt_sha256
    assert saved.model_dump_json() == original


def test_compatible_transport_recipe_drift_and_legacy_absence():
    from harness.execution_profile import ExecutionProfile

    settings = Settings(
        _env_file=None,
        openai_compatible_base_url="https://fixture.example/v1",
        openai_compatible_model="fixture",
        openai_compatible_api_key="fixture-private-key",
    )
    generic = capture_execution_profile(settings, "openai_compatible", ToolRegistry())
    old = ExecutionProfile.model_validate(generic.model_dump(exclude={"provider_options_sha256"}))
    assert compare_profiles(old, generic).status == "matching"
    go = capture_execution_profile(
        settings.model_copy(update={"openai_compatible_transport": "opencode_go"}),
        "openai_compatible",
        ToolRegistry(),
    )
    result = compare_profiles(old, go)
    assert result.status == "changed"
    assert result.changed_fields == ["provider_options_sha256"]
    assert "fixture-private-key" not in result.model_dump_json()


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


def test_memory_prompt_boundary_recorded_and_legacy_absence_is_drift():
    from harness.execution_profile import ExecutionProfile
    from memory.store import MAX_MEMORY_PROMPT_BYTES

    current = capture_execution_profile(Settings(_env_file=None), "ollama", ToolRegistry())
    assert current.memory_prompt_max_bytes == MAX_MEMORY_PROMPT_BYTES
    old = ExecutionProfile.model_validate(current.model_dump(exclude={"memory_prompt_max_bytes"}))
    original = old.model_dump_json()
    assert old.memory_prompt_max_bytes is None
    review = compare_profiles(old, current)
    assert review.status == "changed"
    assert review.changed_fields == ["memory_prompt_max_bytes"]
    assert review.unknown_fields == ["memory_prompt_max_bytes"]
    assert compare_profiles(old, old).status == "unknown"
    assert old.model_dump_json() == original


def test_legacy_semantic_stub_exposure_is_configuration_drift(tmp_path: Path):
    from harness.runtime import build_registry
    from tools.semantic import register_semantic_search_stub

    with FactStore(tmp_path / "facts.db") as store:
        registry = build_registry(fact_store=store, user_id="dev", workspace_root=str(tmp_path))
        current = capture_execution_profile(Settings(_env_file=None), "ollama", registry)
        register_semantic_search_stub(registry)
        old = capture_execution_profile(Settings(_env_file=None), "ollama", registry)
    original = old.model_dump_json()
    review = compare_profiles(old, current)
    assert review.status == "changed"
    assert review.changed_fields == ["tool_names", "tool_specs_sha256"]
    assert old.model_dump_json() == original


def test_missing_memory_scope_version_cannot_match_current_profile():
    from harness.execution_profile import ExecutionProfile

    current = capture_execution_profile(Settings(_env_file=None), "ollama", ToolRegistry())
    old = ExecutionProfile.model_validate(current.model_dump(exclude={"memory_scope_version"}))
    result = compare_profiles(old, current)
    assert result.status == "changed"
    assert result.changed_fields == ["memory_scope_version"]
    assert "memory_scope_version" in result.unknown_fields
    assert compare_profiles(old, old).status == "unknown"


def test_configuration_context_review_uses_current_workspace_memory(tmp_path: Path):
    settings = Settings(_env_file=None, max_context_tokens=16000)
    registry = ToolRegistry()
    archive = archive_with_profile(settings, registry)
    archive.session.workspace_root = str(tmp_path / "a")
    original = archive.model_dump_json()
    with FactStore(tmp_path / "memory.db") as store:
        store.add("dev", "global")
        store.add("dev", "only A", workspace_root=str(tmp_path / "a"))
        store.add("dev", "B" * 4096, workspace_root=str(tmp_path / "b"))
        result = review_configuration(
            archive,
            settings=settings,
            provider="ollama",
            registry=registry,
            fact_store=store,
            user_id="dev",
            message="Explain",
        )
        detached = archive.session.model_copy(deep=True)
        refresh_system_message(
            detached, store, "dev", system_prompt=coding_prompt(settings.coding_toolset)
        )
        assert "only A" in detached.messages[0].content
        assert "B" * 4096 not in detached.messages[0].content
        detached.messages.append(ChatMessage(role="user", content="Explain"))
        assert result.context.estimated_prompt_tokens == TokenBudget(
            max_context=16000
        ).estimate_prompt(detached.messages, registry.as_tool_specs())
    assert archive.model_dump_json() == original
