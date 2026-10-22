"""Allowlisted boundary observations are independent of completion and usage gates."""

import pytest
from pydantic import ValidationError

from providers.base import ProviderCallObservation, ProviderFailure, ProviderResponse
from providers.ollama import _response_from_ollama


@pytest.mark.parametrize(
    ("fields", "prompt_state", "output_state", "reason"),
    [
        ({}, "absent", "absent", "absent"),
        (
            {"prompt_eval_count": None, "eval_count": None, "done_reason": None},
            "null",
            "null",
            "null",
        ),
        (
            {"prompt_eval_count": 0, "eval_count": 0, "done_reason": "stop"},
            "present",
            "present",
            "stop",
        ),
        ({"prompt_eval_count": 3, "done_reason": "length"}, "present", "absent", "length"),
        ({"eval_count": 2, "done_reason": "private-reason"}, "absent", "present", "other"),
    ],
)
def test_empty_response_retains_field_presence_without_inventing_usage(
    fields, prompt_state, output_state, reason
):
    response = _response_from_ollama(
        {"message": {"content": ""}, **fields}, fallback_model="local", latency_ms=12.5
    )
    observation = response.call_observation
    assert observation == ProviderCallObservation(
        duration_ms=12.5,
        prompt_usage_field=prompt_state,
        completion_usage_field=output_state,
        terminal_reason=reason,
        content_empty=True,
        tool_call_count=0,
    )
    assert response.usage.prompt_tokens == fields.get("prompt_eval_count")
    assert response.usage.completion_tokens == fields.get("eval_count")
    assert response.finish_reason == ("length" if reason == "length" else "stop")
    assert "private" not in observation.model_dump_json()


def test_tool_finish_stays_distinct_from_native_terminal_reason():
    response = _response_from_ollama(
        {
            "message": {
                "content": "private text",
                "tool_calls": [
                    {"function": {"name": "private_tool", "arguments": {"secret": "value"}}}
                ],
            },
            "done_reason": "length",
        },
        fallback_model="local",
        latency_ms=1.0,
    )
    assert response.finish_reason == "tool_use"  # Existing normalization remains unchanged.
    observation = response.call_observation
    assert observation.terminal_reason == "length"
    assert observation.content_empty is False and observation.tool_call_count == 1
    assert "private" not in observation.model_dump_json()
    assert "secret" not in observation.model_dump_json()


@pytest.mark.parametrize(
    "fields",
    [
        {"duration_ms": -1.0},
        {"duration_ms": float("inf")},
        {"duration_ms": float("nan")},
        {"duration_ms": "1"},
        {"terminal_reason": "private-reason"},
        {"prompt_usage_field": "private"},
        {"content_empty": "false"},
        {"tool_call_count": True},
        {"tool_call_count": -1},
        {"body": "private"},
    ],
)
def test_observation_rejects_unbounded_or_invalid_metadata(fields):
    with pytest.raises(ValidationError):
        ProviderCallObservation.model_validate({"duration_ms": 0.0, **fields})


def test_legacy_and_other_provider_observations_remain_unknown():
    response = ProviderResponse(content="ok", finish_reason="stop", model="other", latency_ms=0)
    assert response.call_observation is None
    assert ProviderFailure(kind="unknown").call_observation is None
