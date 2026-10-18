import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.server import _build_providers
from harness.config import Settings, get_settings
from harness.configuration_review import compare_profiles
from harness.execution_profile import capture_execution_profile
from harness.model_profiles import ModelProfile
from harness.providers import build_configured_provider, configured_provider_names
from harness.runtime import run_configured_turn
from harness.state import Session
from memory import FactStore
from tests.api.conftest import ScriptedProvider, make_response
from tests.api.test_session_persistence import make_app
from tools import ToolRegistry

PROFILE = {
    "provider": "ollama",
    "model": "gemma4:12b",
    "max_context_tokens": 4096,
    "max_request_output_tokens": 512,
    "ollama_think": False,
}


def test_explicit_selection_wins_and_unselected_configuration_stays_inert(monkeypatch):
    catalog = {"local": PROFILE}
    ordinary = Settings(_env_file=None, model_profiles=catalog, ollama_model="original")
    assert ordinary.ollama_model == "original" and ordinary.profile_identity() is None
    monkeypatch.setenv("MODEL_PROFILES", json.dumps(catalog))
    monkeypatch.setenv("MODEL_PROFILE", "local")
    settings = Settings(
        _env_file=None,
        ollama_model="ignored",
        ollama_num_ctx=8192,
        max_context_tokens=16000,
        max_request_output_tokens=1000,
        prompt_estimate_bytes_per_token=4.5,
        max_completion_tokens_per_turn=6000,
    )
    assert settings.default_provider == "ollama" and settings.ollama_model == "gemma4:12b"
    assert settings.ollama_num_ctx == settings.max_context_tokens == 4096
    assert settings.max_request_output_tokens == 512 and settings.ollama_think is False
    assert settings.prompt_estimate_bytes_per_token == 4.5  # optional profile field inherits
    assert settings.max_completion_tokens_per_turn == 6000
    assert configured_provider_names(settings) == ("ollama",)
    with patch("harness.providers.create_chat_provider") as factory:
        build_configured_provider("ollama", settings)
    assert factory.call_args.kwargs["model"] == "gemma4:12b"
    assert factory.call_args.kwargs["num_ctx"] == 4096
    assert factory.call_args.kwargs["think"] is False
    monkeypatch.setenv("MODEL_PROFILE", "")
    assert Settings(_env_file=None).profile_identity() is None


@pytest.mark.parametrize(
    "change",
    [
        {"model": "bad\nmodel"},
        {"api_key": "private-marker"},
        {"endpoint": "private-marker"},
        {"max_context_tokens": 255},
        {"max_request_output_tokens": 4097},
        {"prompt_estimate_bytes_per_token": float("inf")},
        {"max_completion_retries": 4},
        {"provider": "openai", "ollama_think": False},
    ],
)
def test_invalid_profile_rejects_private_input_before_factory(change):
    with (
        patch("harness.providers.create_chat_provider") as factory,
        pytest.raises(ValidationError) as error,
    ):
        Settings(
            _env_file=None,
            model_profiles={"local": {**PROFILE, **change}},
            model_profile="local",
            openai_api_key="private-marker",
        )
    assert "private-marker" not in str(error.value)
    factory.assert_not_called()


@pytest.mark.parametrize(
    "selection,catalog",
    [
        ("missing", {"local": PROFILE}),
        ("bad name", {"local": PROFILE}),
        ("local", {"bad name": PROFILE}),
        ("p0", {f"p{i}": PROFILE for i in range(17)}),
    ],
)
def test_unknown_names_and_oversized_catalog_fail(selection, catalog):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_profiles=catalog, model_profile=selection)


def test_profile_identity_is_stable_selected_only_and_configuration_drift():
    settings = Settings(_env_file=None, model_profiles={"local": PROFILE}, model_profile="local")
    first = capture_execution_profile(settings, "ollama", ToolRegistry())
    reordered = dict(reversed(list(PROFILE.items())))
    current = capture_execution_profile(
        Settings(
            _env_file=None,
            model_profiles={"local": reordered, "unused": {**PROFILE, "model": "other"}},
            model_profile="local",
        ),
        "ollama",
        ToolRegistry(),
    )
    assert first == current
    assert first.model_profile.name == "local"
    assert len(first.model_profile.definition_sha256) == 64
    assert "model_profiles" not in first.model_dump_json()
    renamed = capture_execution_profile(
        Settings(_env_file=None, model_profiles={"renamed": PROFILE}, model_profile="renamed"),
        "ollama",
        ToolRegistry(),
    )
    assert compare_profiles(first, renamed).changed_fields == ["model_profile"]
    old = first.model_copy(update={"model_profile": None})
    assert compare_profiles(old, first).status == "changed"


def test_compatible_profile_and_credentials_share_api_eval_factory():
    compatible = {
        "provider": "openai_compatible",
        "model": "small-coder",
        "max_context_tokens": 8192,
        "max_request_output_tokens": 1024,
    }
    settings = Settings(
        _env_file=None,
        model_profiles={"hosted": compatible},
        model_profile="hosted",
        openai_compatible_base_url="https://fixture.example/v1",
        openai_compatible_api_key="private-key",
        anthropic_api_key="unused-private-key",
    )
    assert configured_provider_names(settings) == ("openai_compatible",)
    with patch("harness.providers.create_chat_provider") as factory:
        _build_providers(settings)
        build_configured_provider("openai_compatible", settings)
        assert factory.call_count == 2
        assert factory.call_args.kwargs["model"] == "small-coder"
        assert factory.call_args.kwargs["api_key"] == "private-key"
        with pytest.raises(ValueError):
            build_configured_provider("anthropic", settings)
        assert factory.call_count == 2
    assert (
        "private-key"
        not in capture_execution_profile(
            settings, "openai_compatible", ToolRegistry()
        ).model_dump_json()
    )


def test_profile_requires_hosted_setup_and_live_smoke_rejects_mismatched_provider():
    from evals.run import _build_live_provider, _validate_live_providers

    paid = {
        "provider": "anthropic",
        "model": "fixture",
        "max_context_tokens": 8192,
        "max_request_output_tokens": 1024,
    }
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_profiles={"hosted": paid}, model_profile="hosted")
    compatible = {**paid, "provider": "openai_compatible"}
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_profiles={"hosted": compatible}, model_profile="hosted")
    settings = Settings(_env_file=None, model_profiles={"local": PROFILE}, model_profile="local")
    with pytest.raises(SystemExit):
        _validate_live_providers(["ollama", "openai"], settings)
    with pytest.raises(ValueError):
        _build_live_provider("openai", settings)


async def test_shared_runtime_rejects_profile_provider_mismatch_before_mutation(tmp_path):
    settings = Settings(_env_file=None, model_profiles={"local": PROFILE}, model_profile="local")
    session = Session(user_id="dev")
    original = session.model_dump_json()
    provider = ScriptedProvider()
    with FactStore(tmp_path / "facts.db") as store:
        result = await run_configured_turn(
            settings=settings,
            session=session,
            user_id="dev",
            message="Inspect",
            provider=provider,
            fact_store=store,
            registry=ToolRegistry(),
        )
    assert result.completion_status == "blocked" and provider.calls == []
    assert session.model_dump_json() == original


def test_api_restart_preserves_profile_and_get_settings_binds_explicit_env(tmp_path, monkeypatch):
    catalog = {"local": PROFILE}
    monkeypatch.setenv("MODEL_PROFILES", json.dumps(catalog))
    monkeypatch.setenv("MODEL_PROFILE", "local")
    # Avoid user .env: exercise the cached entry point using a hermetic Settings instance.
    settings = Settings(_env_file=None)
    with patch("harness.config.Settings", return_value=settings):
        get_settings.cache_clear()
        assert get_settings().profile_identity() == settings.profile_identity()
    get_settings.cache_clear()
    provider = ScriptedProvider()
    provider.name = "ollama"
    provider.script(make_response(content="Reviewed", prompt_tokens=10, completion_tokens=5))
    with TestClient(
        make_app(tmp_path, provider, model_profiles=catalog, model_profile="local")
    ) as api:
        sid = api.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        assert (
            api.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Inspect"}
            ).json()["completion_status"]
            == "completed"
        )
        saved = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        assert (
            saved["responses"][0]["execution_profile"]["model_profile"]
            == settings.profile_identity().model_dump()
        )
    with TestClient(
        make_app(tmp_path, provider, model_profiles=catalog, model_profile="local")
    ) as api:
        assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
    assert len(provider.calls) == 1


def test_profile_optional_values_are_validated_and_frozen():
    profile = ModelProfile(
        **{**PROFILE, "prompt_estimate_bytes_per_token": 4.5, "max_completion_retries": 2}
    )
    assert profile.settings_overrides()["max_completion_retries"] == 2
    assert profile.settings_overrides()["prompt_estimate_bytes_per_token"] == 4.5
    with pytest.raises(ValidationError):
        profile.model = "changed"


@pytest.mark.parametrize("change", ["none", "rename", "iterations"])
def test_profile_continuation_identity_survives_restart_and_blocks_changes(tmp_path, change):
    from tests.api.test_continuation import body, known_provider, submit_and_finish

    root = tmp_path / "repo"
    root.mkdir()
    (root / "user.py").write_text("user_edit = 1\n")
    definition = {**PROFILE, "max_context_tokens": 8192}
    settings = {"model_profiles": {"local": definition}, "model_profile": "local"}
    provider = known_provider()
    provider.script(make_response(content="Inspected", prompt_tokens=10, completion_tokens=5))
    with TestClient(make_app(tmp_path, provider, **settings)) as api:
        sid = api.post(
            "/sessions",
            json={"user_id": "dev", "workspace_root": str(root), "checkpoint_turns": True},
        ).json()["session_id"]
        assert (
            api.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Inspect"}
            ).json()["completion_status"]
            == "completed"
        )
        saved = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
    later = known_provider()
    later.script(make_response(content="Reviewed again", prompt_tokens=12, completion_tokens=5))
    if change == "rename":
        settings = {"model_profiles": {"renamed": definition}, "model_profile": "renamed"}
    elif change == "iterations":
        settings = {
            "model_profiles": {"local": {**definition, "max_tool_iterations": 12}},
            "model_profile": "local",
        }
    with TestClient(make_app(tmp_path, later, **settings)) as api:
        run = submit_and_finish(api, sid, body(saved))
        assert run["response"]["completion_status"] == (
            "completed" if change == "none" else "blocked"
        ), run
        assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
    assert len(later.calls) == (1 if change == "none" else 0)
    assert (root / "user.py").read_text() == "user_edit = 1\n"


async def test_real_evaluator_uses_selected_profile_identity_and_bound_request():
    from evals.real_run import load_tasks, run_task

    class ObservedProvider(ScriptedProvider):
        name = "ollama"

        async def chat(self, messages, **kwargs):
            assert kwargs["max_tokens"] == 512
            return await super().chat(messages, **kwargs)

    settings = Settings(
        _env_file=None,
        model_profiles={"local": {**PROFILE, "max_context_tokens": 8192}},
        model_profile="local",
    )
    provider = ObservedProvider()
    provider.script(make_response(content="No patch", prompt_tokens=10, completion_tokens=5))
    result = await run_task(load_tasks()[0], provider=provider, settings=settings, mode="harness")
    assert result.runtime_config["model_profile"] == settings.profile_identity().model_dump()
    assert result.runtime_config["max_request_output_tokens"] == 512
    assert result.runtime_config["effective_max_context_tokens"] == 8192
    assert len(provider.calls) == 1
    assert not result.passed  # Independent acceptance rejects the unchanged fixture.


def test_optional_iteration_limit_inherits_and_preserves_old_profile_identity():
    import hashlib

    inherited = ModelProfile(**PROFILE)
    old_identity = hashlib.sha256(
        json.dumps(PROFILE, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert inherited.identity("local").definition_sha256 == old_identity
    assert (
        ModelProfile(**PROFILE, max_tool_iterations=32).settings_overrides()["max_tool_iterations"]
        == 32
    )
    assert ModelProfile(**PROFILE, max_tool_iterations=None).identity(
        "local"
    ) == inherited.identity("local")
    settings = Settings(
        _env_file=None,
        model_profiles={"local": PROFILE},
        model_profile="local",
        max_tool_iterations=5,
    )
    assert settings.max_tool_iterations == 5
    catalog = {"local": {**PROFILE, "max_tool_iterations": 12}}
    selected = Settings(
        _env_file=None, model_profiles=catalog, model_profile="local", max_tool_iterations=3
    )
    assert selected.max_tool_iterations == 12
    assert (
        Settings(_env_file=None, model_profiles=catalog, max_tool_iterations=3).max_tool_iterations
        == 3
    )
    assert selected.profile_identity() != settings.profile_identity()
    assert Settings(_env_file=None).max_tool_iterations == 8


@pytest.mark.parametrize("limit", [0, -1, 33, True, 1.5, "12"])
def test_profile_iteration_limit_rejects_invalid_values_before_provider_construction(limit):
    with patch("harness.providers.create_chat_provider") as factory, pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            model_profiles={"local": {**PROFILE, "max_tool_iterations": limit}},
            model_profile="local",
        )
    factory.assert_not_called()


@pytest.mark.parametrize("limit", [1, 2])
async def test_profile_iteration_limit_is_enforced_and_recorded_in_real_runtime(limit):
    from evals.real_run import load_tasks, run_task
    from providers.base import ToolCall

    class LocalProvider(ScriptedProvider):
        name = "ollama"

    task = load_tasks()[0]
    provider = LocalProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={
                        "path": "calc.py",
                        "content": (task.reference / "calc.py").read_text(),
                    },
                )
            ],
            prompt_tokens=10,
            completion_tokens=5,
        ),
        make_response(content="Fixed", prompt_tokens=10, completion_tokens=5),
    )
    settings = Settings(
        _env_file=None,
        model_profiles={
            "local": {**PROFILE, "max_context_tokens": 8192, "max_tool_iterations": limit}
        },
        model_profile="local",
        max_tool_iterations=9,
    )
    result = await run_task(task, provider=provider, settings=settings, mode="harness")
    assert result.passed
    assert len(provider.calls) == limit
    assert result.completion_status == ("completed" if limit == 2 else "budget_exhausted")
    assert result.runtime_config["max_tool_iterations"] == limit
    assert result.runtime_config["model_profile"] == settings.profile_identity().model_dump()
    assert result.verification_status == "not_run"
