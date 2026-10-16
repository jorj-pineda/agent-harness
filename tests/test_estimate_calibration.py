import math

import pytest
from pydantic import ValidationError

from harness.config import Settings
from harness.configuration_review import compare_profiles, review_configuration
from harness.context import TokenBudget, estimate_message_tokens, estimate_tool_tokens
from harness.execution_profile import ExecutionProfile, capture_execution_profile
from harness.runtime import run_configured_turn
from harness.state import Session
from harness.trace import PromptFootprint, RequestTrace, prompt_footprint
from memory import FactStore
from providers.base import ChatMessage, TokenUsage, ToolSpec
from tests.api.conftest import ScriptedProvider, make_response
from tests.test_configuration_review import archive_with_profile
from tools import ToolRegistry


@pytest.mark.parametrize("ratio", [0, 0.99, 8.01, math.inf, -math.inf, math.nan])
def test_invalid_calibration_is_rejected(ratio):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, prompt_estimate_bytes_per_token=ratio)
    with pytest.raises(ValueError):
        TokenBudget(bytes_per_token=ratio)


def test_env_ratio_and_reported_anchor_keep_larger_estimate(monkeypatch):
    monkeypatch.setenv("PROMPT_ESTIMATE_BYTES_PER_TOKEN", "4.5")
    settings = Settings(_env_file=None)
    assert settings.prompt_estimate_bytes_per_token == 4.5
    messages = [ChatMessage(role="user", content="é" * 90)]
    tools = [ToolSpec(name="echo", description="echo", parameters_schema={})]
    budget = TokenBudget(bytes_per_token=4.5)
    assert estimate_message_tokens(messages, bytes_per_token=4.5) == 44
    estimate = budget.estimate_prompt(messages, tools)
    assert estimate == 44 + estimate_tool_tokens(tools, bytes_per_token=4.5)
    assert estimate < TokenBudget().estimate_prompt(messages, tools)
    budget.record(TokenUsage(prompt_tokens=1, completion_tokens=1), 1)
    assert budget.estimate_prompt(messages, tools) == estimate
    budget.record(TokenUsage(prompt_tokens=1000, completion_tokens=1), 1)
    messages.append(ChatMessage(role="assistant", content="é" * 9))
    assert budget.estimate_prompt(messages, tools) == 1008
    footprint = prompt_footprint(messages, tools, budget)
    assert footprint.prompt_estimate_bytes_per_token == 4.5
    assert footprint.heuristic_prompt_tokens + footprint.anchor_adjustment_tokens == 1008


async def test_runtime_review_trace_and_durable_identity_share_calibration(tmp_path):
    settings = Settings(
        _env_file=None,
        prompt_estimate_bytes_per_token=4.5,
        max_context_tokens=5000,
        min_request_output_tokens=16,
    )
    registry = ToolRegistry()
    provider = ScriptedProvider()
    provider.script(make_response(content="Inspected"))
    session = Session(user_id="dev")
    trace, profiles = [], []
    with FactStore(tmp_path / "facts.db") as store:
        review = review_configuration(
            archive_with_profile(settings, registry),
            settings=settings,
            provider="ollama",
            registry=registry,
            fact_store=store,
            user_id="dev",
            message="Inspect " + "é" * 900,
        )
        assert review.context.prompt_estimate_bytes_per_token == 4.5
        await run_configured_turn(
            settings=settings,
            session=session,
            user_id="dev",
            message="Inspect " + "é" * 900,
            provider=provider,
            registry=registry,
            fact_store=store,
            trace=trace,
            on_execution_profile=profiles.append,
        )
    request = next(item for item in trace if isinstance(item, RequestTrace))
    budget = TokenBudget(max_context=5000, min_request_output=16, bytes_per_token=4.5)
    sent, specs = provider.calls[0]
    assert request.estimated_prompt_tokens == budget.estimate_prompt(sent, specs or [])
    assert request.max_tokens == budget.request_allowance(sent, specs or [])
    assert request.footprint.prompt_estimate_bytes_per_token == 4.5
    assert profiles[0].settings.prompt_estimate_bytes_per_token == 4.5
    assert ExecutionProfile.model_validate_json(profiles[0].model_dump_json()) == profiles[0]
    default = capture_execution_profile(Settings(_env_file=None), "scripted", registry)
    assert (
        "settings.prompt_estimate_bytes_per_token"
        in compare_profiles(default, profiles[0]).changed_fields
    )


def test_legacy_estimator_metadata_retains_known_fixed_ratio():
    current = capture_execution_profile(Settings(_env_file=None), "ollama", ToolRegistry())
    payload = current.model_dump()
    payload["settings"].pop("prompt_estimate_bytes_per_token")
    old = ExecutionProfile.model_validate(payload)
    assert old.settings.prompt_estimate_bytes_per_token == 3
    assert compare_profiles(old, current).status == "matching"
    footprint = prompt_footprint([], [], TokenBudget()).model_dump()
    footprint.pop("prompt_estimate_bytes_per_token")
    assert PromptFootprint.model_validate(footprint).prompt_estimate_bytes_per_token == 3


@pytest.mark.parametrize("ratio,expected_calls", [(3, 0), (4, 1)])
async def test_calibration_changes_pre_request_gate_without_changing_usage(ratio, expected_calls):
    from harness.loop import run_turn

    provider = ScriptedProvider()
    provider.script(make_response(content="done", prompt_tokens=90, completion_tokens=5))
    response = await run_turn(
        session=Session(),
        user_input="x" * 300,
        provider=provider,
        registry=ToolRegistry(),
        max_context_tokens=100,
        prompt_estimate_bytes_per_token=ratio,
    )
    assert len(provider.calls) == expected_calls
    assert response.completion_status == ("completed" if expected_calls else "budget_exhausted")
    assert response.token_usage.prompt_tokens == (90 if expected_calls else 0)
