from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from evals.run import _build_live_provider
from harness.config import Settings
from harness.configuration_review import compare_profiles, review_configuration
from harness.context import BudgetStop, TokenBudget
from harness.execution_profile import ExecutionProfile, capture_execution_profile
from harness.runtime import run_configured_turn
from harness.session_store import SessionArchive
from harness.state import Session
from harness.trace import BudgetStopTrace, RequestTrace
from memory import FactStore
from tests.api.conftest import ScriptedProvider, make_response
from tests.api.test_session_persistence import make_app
from tools import ToolRegistry


@pytest.mark.parametrize(
    "harness,ollama,provider,effective",
    [
        (0, None, "ollama", 0),
        (0, 4096, "ollama", 4096),
        (8192, 4096, "ollama", 4096),
        (4096, 8192, "ollama", 4096),
        (8192, 8192, "ollama", 8192),
        (8192, None, "ollama", 8192),
        (0, 4096, "openai", 0),
        (8192, 4096, "openai_compatible", 8192),
    ],
)
def test_effective_limit_only_uses_selected_provider(harness, ollama, provider, effective):
    settings = Settings(_env_file=None, max_context_tokens=harness, ollama_num_ctx=ollama)
    assert settings.context_limit_for(provider) == effective


@pytest.mark.parametrize(
    "harness,ollama,name",
    [
        (8192, 4096, "ollama"),
        (4096, 8192, "ollama"),
        (0, 4096, "ollama"),
        (0, 8192, "ollama"),
        (8192, 256, "openai"),
        (0, None, "ollama"),
    ],
)
async def test_review_runtime_capture_and_token_trace_use_same_effective_limit(
    tmp_path,
    harness,
    ollama,
    name,
):
    settings = Settings(
        _env_file=None,
        max_context_tokens=harness,
        ollama_num_ctx=ollama,
        min_request_output_tokens=16,
    )
    provider = ScriptedProvider()
    provider.name = name
    provider.script(make_response(content="Inspected", prompt_tokens=100, completion_tokens=1))
    trace, profiles = [], []
    message = "Inspect " + "x" * 12000
    with FactStore(tmp_path / "facts.db") as store:
        review = review_configuration(
            SessionArchive(session=Session(user_id="dev")),
            settings=settings,
            provider=name,
            registry=ToolRegistry(),
            fact_store=store,
            user_id="dev",
            message=message,
        )
        response = await run_configured_turn(
            settings=settings,
            session=Session(user_id="dev"),
            user_id="dev",
            message=message,
            provider=provider,
            fact_store=store,
            registry=ToolRegistry(),
            trace=trace,
            on_execution_profile=profiles.append,
        )
    context = review.context
    assert context.max_context_tokens == settings.context_limit_for(name)
    assert context.configured_max_context_tokens == harness
    assert context.ollama_num_ctx == (ollama if name == "ollama" else None)
    assert context.context_limit_policy == "configured-min-v1"
    assert profiles[0].effective_max_context_tokens == context.max_context_tokens
    assert profiles[0].context_limit_policy == context.context_limit_policy
    if context.status == "blocked":
        assert provider.calls == [] and response.completion_status == "budget_exhausted"
        stop = next(item for item in trace if isinstance(item, BudgetStopTrace))
        assert stop.max_context_tokens == context.max_context_tokens
        assert stop.footprint.estimated_prompt_tokens == context.estimated_prompt_tokens
    else:
        request = next(item for item in trace if isinstance(item, RequestTrace))
        sent, specs = provider.calls[0]
        budget = TokenBudget(max_context=context.max_context_tokens, min_request_output=16)
        assert request.estimated_prompt_tokens == context.estimated_prompt_tokens
        assert request.max_tokens == context.request_output_allowance
        assert not isinstance(budget.request_allowance(sent, specs or []), BudgetStop)
        if request.max_tokens is not None:
            assert (
                request.estimated_prompt_tokens + request.max_tokens <= context.max_context_tokens
            )


def test_old_policy_is_unknown_and_cannot_silently_match_current():
    current = capture_execution_profile(
        Settings(_env_file=None, ollama_num_ctx=4096), "ollama", ToolRegistry()
    )
    payload = current.model_dump(exclude={"context_limit_policy", "effective_max_context_tokens"})
    old = ExecutionProfile.model_validate(payload)
    original = old.model_dump_json()
    result = compare_profiles(old, current)
    assert result.status == "changed"
    assert result.changed_fields == ["context_limit_policy", "effective_max_context_tokens"]
    assert result.unknown_fields == result.changed_fields
    assert old.model_dump_json() == original


def test_restart_preserves_effective_policy_and_review_does_not_submit(tmp_path):
    provider = ScriptedProvider()
    provider.name = "ollama"
    settings = {"max_context_tokens": 8192, "ollama_num_ctx": 4096}
    with TestClient(make_app(tmp_path, provider, **settings)) as api:
        sid = api.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        result = api.post(
            "/chat", json={"user_id": "dev", "session_id": sid, "message": "Inspect " + "x" * 20000}
        ).json()
        assert result["completion_status"] == "budget_exhausted"
        saved = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        profile = saved["responses"][0]["execution_profile"]
        assert profile["settings"]["max_context_tokens"] == 8192
        assert profile["effective_max_context_tokens"] == 4096
    with TestClient(make_app(tmp_path, provider, **settings)) as api:
        assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
        runs = api.get("/runs", params={"user_id": "dev"}).json()
        review = api.post(f"/sessions/{sid}/configuration-review", json={"user_id": "dev"}).json()
        assert review["configuration"]["status"] == "matching"
        assert review["context"]["max_context_tokens"] == 4096
        assert api.get("/runs", params={"user_id": "dev"}).json() == runs
    assert provider.calls == []


def test_live_simulated_provider_sends_same_explicit_context_options():
    settings = Settings(_env_file=None, ollama_num_ctx=4096, ollama_think=False)
    with patch("evals.run.create_chat_provider") as factory:
        _build_live_provider("ollama", settings)
    assert factory.call_args.kwargs["num_ctx"] == 4096
    assert factory.call_args.kwargs["think"] is False
