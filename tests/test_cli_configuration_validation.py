"""Configuration review states must not imply execution or resume authority."""

from unittest.mock import patch

import httpx
import pytest
from typer.testing import CliRunner

from cli._client import AgentClient
from cli.main import app

runner = CliRunner()


def review():
    return {
        "status": "reviewed",
        "read_only": True,
        "resume_supported": False,
        "scope_note": "Estimation only.",
        "optional_evidence": {"preserve": True},
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "approved"},
        {"read_only": False},
        {"resume_supported": True},
        {"configuration": {"status": "compatible"}},
        {"transcript": {"status": "repaired"}},
        {"context": {"status": "guaranteed_fit"}},
        {"context": {"status": "blocked"}},
        {
            "context": {
                "status": "estimated_fit",
                "estimated_prompt_tokens": True,
                "includes_proposed_message": True,
            }
        },
        {
            "configuration": {
                "status": "matching",
                "changed_fields": [],
                "unknown_fields": "workspace",
            }
        },
        {"scope_note": None},
    ],
)
def test_unsupported_review_refused_before_status_display_without_retry(changes):
    data = review() | changes
    response = httpx.Response(200, json=data, request=httpx.Request("POST", "http://fixture"))
    with patch("httpx.post", return_value=response) as post, patch("httpx.get") as get:
        result = runner.invoke(app, ["configuration-review", "s1"])
    assert result.exit_code == 1
    assert "review=" not in result.output
    assert "No automatic retry" in result.output
    post.assert_called_once()
    assert post.call_args.args[0].endswith("/sessions/s1/configuration-review")
    get.assert_not_called()


@pytest.mark.parametrize("status", ["reviewed", "busy", "unavailable"])
def test_supported_envelope_keeps_optional_evidence(status):
    data = review() | {"status": status}
    response = httpx.Response(200, json=data, request=httpx.Request("POST", "http://fixture"))
    with patch("httpx.post", return_value=response) as post:
        assert AgentClient().configuration_review("dev", "s1") == data
    post.assert_called_once()
