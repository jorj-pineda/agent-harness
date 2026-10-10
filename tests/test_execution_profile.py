from pathlib import Path

import pytest

from harness.config import Settings
from harness.execution_profile import capture_execution_profile
from harness.prompts import PROMPT_VERSION
from harness.runtime import run_configured_turn
from harness.state import Session
from memory import FactStore
from providers.base import ChatMessage, ToolCall
from tools import ToolRegistry

from .api.conftest import ScriptedProvider, make_response
from .test_harness_loop import _echo_tool


def test_allowlist_does_not_serialize_secrets_paths_endpoints_or_check_arguments(monkeypatch):
    settings = Settings(
        _env_file=None,
        anthropic_api_key="fixture-private-anthropic",
        openai_api_key="fixture-private-openai",
        openai_compatible_api_key="fixture-private-compatible",
        openai_compatible_base_url="https://fixture.example/private-endpoint",
        openai_compatible_model="fixture-model",
        memory_db_path=Path("fixture-private-memory"),
        session_db_path=Path("fixture-private-session"),
        project_check_argv=["pytest", "fixture-private-check"],
        ollama_num_ctx=8192,
        ollama_think=False,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Never serialize the whole Settings object")

    monkeypatch.setattr(Settings, "model_dump", forbidden)
    registry = ToolRegistry()
    registry.register(_echo_tool())
    profile = capture_execution_profile(settings, "openai_compatible", registry)
    payload = profile.model_dump_json()
    assert "fixture-private" not in payload
    assert "fixture.example" not in payload
    assert "private-endpoint" not in payload
    assert "api_key" not in payload
    assert profile.endpoint_sha256 and len(profile.endpoint_sha256) == 64
    assert profile.project_check_sha256 and len(profile.project_check_sha256) == 64
    assert profile.configured_model == "fixture-model"
    assert profile.prompt_version == PROMPT_VERSION
    assert profile.tool_names == ["echo"]
    assert profile.ollama_num_ctx is None
    assert profile.ollama_think is None
    assert "resume permission" in profile.scope_note


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tool_calls_per_turn", 11),
        ("max_context_tokens", 4096),
        ("project_check_argv", ["pytest", "-q"]),
        ("ollama_model", "different-model"),
        ("ollama_host", "http://localhost:22222"),
        ("ollama_num_ctx", 4096),
        ("ollama_think", False),
        ("require_plan_before_edit", True),
        ("coding_toolset", "whole_file"),
        ("max_tracked_bytes", 100),
    ],
)
def test_relevant_configuration_changes_are_visible(field, value):
    registry = ToolRegistry()
    first = capture_execution_profile(Settings(_env_file=None), "ollama", registry)
    changed = capture_execution_profile(
        Settings(_env_file=None, **{field: value}), "ollama", registry
    )
    assert changed != first


def test_hashes_are_stable_and_detect_prompt_and_schema_changes():
    settings = Settings(_env_file=None)
    registry = ToolRegistry()
    first = capture_execution_profile(settings, "ollama", registry)
    assert capture_execution_profile(settings, "ollama", registry) == first
    custom = capture_execution_profile(settings, "ollama", registry, system_prompt="custom rules")
    assert custom.prompt_version is None
    assert custom.base_prompt_sha256 != first.base_prompt_sha256
    registry.register(_echo_tool())
    assert (
        capture_execution_profile(settings, "ollama", registry).tool_specs_sha256
        != first.tool_specs_sha256
    )
    # Unrelated credentials/storage paths do not change identity.
    changed = Settings(_env_file=None, openai_api_key="fixture", session_db_path=Path("elsewhere"))
    assert capture_execution_profile(changed, "ollama", ToolRegistry()) == first


@pytest.mark.parametrize(
    "host",
    [
        "http://user:fixture-password@localhost:11434",
        "http://localhost/?key=fixture",
        "not-a-url",
        "http://[bad",
        "http://localhost:not-a-port",
    ],
)
def test_sensitive_or_invalid_endpoint_is_unknown_without_exposing_it(host):
    profile = capture_execution_profile(
        Settings(_env_file=None, ollama_host=host), "ollama", ToolRegistry()
    )
    assert profile.endpoint_sha256 is None
    assert host not in profile.model_dump_json()


def test_custom_provider_identity_stays_unknown():
    profile = capture_execution_profile(Settings(_env_file=None), "custom", ToolRegistry())
    assert profile.configured_model is None
    assert profile.endpoint_sha256 is None


async def test_shared_runtime_callback_precedes_inference_and_finalization(tmp_path: Path):
    seen = []
    session = Session(user_id="dev")

    class ObservedProvider(ScriptedProvider):
        async def chat(self, messages, **kwargs):
            assert len(seen) == 1
            return await super().chat(messages, **kwargs)

    provider = ObservedProvider()
    provider.script(make_response(content="Done"))
    with FactStore(tmp_path / "memory.db") as store:

        def capture(profile):
            assert session.messages == []
            assert session.turns == []
            seen.append(profile)

        await run_configured_turn(
            settings=Settings(_env_file=None),
            session=session,
            user_id="dev",
            message="Explain",
            provider=provider,
            fact_store=store,
            registry=ToolRegistry(),
            on_execution_profile=capture,
            on_finalized=lambda *_: seen.append("final"),
            system_prompt="simulation override",
        )
    assert len(seen) == 2
    assert seen[0].prompt_version is None


async def test_profile_callback_failure_prevents_inference_and_mutation(tmp_path: Path):
    provider = ScriptedProvider()
    session = Session(user_id="dev")

    def fail(_):
        raise OSError("fixture storage failure")

    with FactStore(tmp_path / "memory.db") as store, pytest.raises(OSError):
        await run_configured_turn(
            settings=Settings(_env_file=None),
            session=session,
            user_id="dev",
            message="Explain",
            provider=provider,
            fact_store=store,
            registry=ToolRegistry(),
            on_execution_profile=fail,
        )
    assert provider.calls == []
    assert session.messages == []
    assert session.turns == []


async def test_malformed_history_is_rejected_before_profile_capture(tmp_path: Path):
    session = Session(
        messages=[ChatMessage(role="assistant", tool_calls=[ToolCall(id="a", name="echo")])]
    )

    def forbidden(_):
        pytest.fail("Invalid history must not record admitted execution")

    with FactStore(tmp_path / "memory.db") as store:
        result = await run_configured_turn(
            settings=Settings(_env_file=None),
            session=session,
            user_id="dev",
            message="Explain",
            provider=ScriptedProvider(),
            fact_store=store,
            registry=ToolRegistry(),
            on_execution_profile=forbidden,
        )
    assert result.completion_status == "blocked"
