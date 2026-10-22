"""Saved metadata presentation is bounded, conservative and read-only."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from cli.main import app, provider_observation_text
from tests.test_cli import _saved_run

runner = CliRunner()


def observation(**changes):
    return {
        "duration_ms": 0.0,
        "prompt_usage_field": "present",
        "completion_usage_field": "null",
        "terminal_reason": "other",
        "content_empty": False,
        "tool_call_count": 0,
        **changes,
    }


@pytest.mark.parametrize("value", [None, [], "private", 1])
def test_unknown_observation_is_unrecorded(value):
    assert provider_observation_text(value) == "unrecorded"


def test_zero_false_and_missing_fields_are_not_conflated():
    text = provider_observation_text(observation())
    assert "HTTP=0.0ms" in text and "native_usage=present/null" in text
    assert "zero_length_content=no tool_calls=0" in text
    unsupported = provider_observation_text(
        observation(
            duration_ms=True,
            prompt_usage_field="private",
            completion_usage_field=[],
            terminal_reason="private",
            content_empty="false",
            tool_call_count=True,
        )
    )
    assert unsupported == (
        "HTTP=unknown native_usage=unknown/unknown native_terminal=unknown "
        "zero_length_content=unknown tool_calls=unknown"
    )
    assert "private" not in unsupported
    assert "HTTP=unknown" in provider_observation_text(observation(duration_ms=10**400))
    assert "HTTP=unknown" in provider_observation_text(observation(duration_ms=float("inf")))


@pytest.mark.parametrize("evidence", [None, {}, {"events": [None, 2, {"kind": "response"}]}])
def test_saved_legacy_or_partial_records_remain_review_only(evidence):
    record = _saved_run("failed")
    record["budget_evidence"] = evidence
    with patch("cli.main.AgentClient") as mock:
        client = mock.return_value
        client.run.return_value = record
        result = runner.invoke(app, ["run", "r1"])
    assert result.exit_code == 0, result.output
    assert "unrecorded" in result.output and "prefix_truncated=unknown" in result.output
    client.run.assert_called_once()
    client.chat.assert_not_called()
    client.create_session.assert_not_called()
    client.cancel.assert_not_called()


def test_truncated_prefix_and_separate_final_failure_do_not_imply_complete_coverage():
    record = _saved_run("failed")
    record["budget_evidence"] = {
        "events": [{"kind": "response", "iteration": 0, "call_observation": observation()}] * 101,
        "observed_events": 150,
        "events_truncated": True,
    }
    record["provider_failure"] = {"call_observation": observation(duration_ms=60000.0)}
    with patch("cli.main.AgentClient") as mock:
        mock.return_value.run.return_value = record
        result = runner.invoke(app, ["run", "r1"])
    assert result.exit_code == 0, result.output
    assert result.output.count("request=1 response:") == 100
    assert "saved_events=100 observed_events=150 prefix_truncated=yes" in result.output
    assert "final failure (separate record): HTTP=60000.0ms" in result.output
    assert "later observations may be absent" in result.output
    assert "not valid usage or budget compliance" in result.output
    mock.return_value.chat.assert_not_called()


def test_request_failure_observation_is_not_duplicated_or_inferred_from_legacy():
    record = _saved_run("failed")
    record["budget_evidence"] = {
        "events": [
            {"kind": "request_failure", "iteration": 2, "call_observation": observation()},
        ],
        "observed_events": 1,
        "events_truncated": False,
    }
    record["provider_failure"] = {"call_observation": observation()}
    with patch("cli.main.AgentClient") as mock:
        mock.return_value.run.return_value = record
        result = runner.invoke(app, ["run", "r1"])
    assert result.exit_code == 0, result.output
    assert "request=3 request_failure:" in result.output
    assert "separate record" not in result.output
