"""Unsupported or contradictory transcript review must not look like usable evidence."""

from copy import deepcopy
from unittest.mock import patch

import httpx
import pytest
from typer.testing import CliRunner

from cli._client import AgentClient
from cli.main import app

runner = CliRunner()


def review():
    return {
        "status": "valid",
        "message_count": 2,
        "issue_count": 0,
        "issues": [],
        "issues_truncated": False,
        "read_only": True,
        "scope_note": "Structure only.",
        "optional_evidence": "preserve",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "repaired"},
        {"read_only": False},
        {"message_count": -1},
        {"status": "invalid", "issue_count": 0},
        {"issue_count": 1, "issues": [{"message_index": 0, "code": "orphan_tool_result"}]},
        {
            "status": "invalid",
            "issue_count": 1,
            "issues_truncated": True,
            "issues": [{"message_index": 9, "code": "orphan_tool_result"}],
        },
        {"issues_truncated": True},
        {"scope_note": None},
        {"scope_note": ""},
        {"scope_note": ["structure only"]},
    ],
)
def test_bad_review_refused_before_printing_structure_status(changes):
    data = review() | changes
    response = httpx.Response(200, json=data, request=httpx.Request("GET", "http://fixture"))
    with patch("httpx.get", return_value=response) as get, patch("httpx.post") as post:
        result = runner.invoke(app, ["transcript-review", "s1"])
    assert result.exit_code == 1
    assert "transcript=" not in result.output and "message[" not in result.output
    get.assert_called_once()
    post.assert_not_called()


@pytest.mark.parametrize("invalid", [False, True])
def test_supported_review_preserves_original_evidence(invalid):
    data = review()
    if invalid:
        data.update(
            status="invalid",
            issue_count=2,
            issues_truncated=True,
            issues=[{"message_index": 1, "code": "future_diagnostic"}],
        )
    original = deepcopy(data)
    response = httpx.Response(200, json=data, request=httpx.Request("GET", "http://fixture"))
    with patch("httpx.get", return_value=response), patch("httpx.post") as post:
        assert AgentClient().transcript_review("dev", "s1", max_issues=1) == original
    post.assert_not_called()
