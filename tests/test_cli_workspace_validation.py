"""Reject unsupported workspace evidence before showing a usable review status."""

from unittest.mock import patch

import httpx
import pytest
from typer.testing import CliRunner

from cli._client import AgentClient
from cli.main import app

runner = CliRunner()


def review():
    return {
        "status": "changed",
        "modified": ["a.py"],
        "modified_count": 1,
        "paths_truncated": False,
        "mode_coverage": "compared",
        "read_only": True,
        "scope_note": "Non-atomic, scoped review only.",
        "optional_evidence": {"keep": True},
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "safe_to_resume"},
        {"read_only": False},
        {"mode_coverage": "verified"},
        {"modified_count": -1},
        {"modified_count": 0},
        {"modified": "a.py"},
        {"modified_count": 2},
        {"status": "unchanged"},
        {"status": "unavailable"},
        {"modified": [], "modified_count": 0},
    ],
)
def test_bad_workspace_evidence_refused_before_display(changes):
    data = review() | changes
    response = httpx.Response(200, json=data, request=httpx.Request("GET", "http://fixture"))
    with patch("httpx.get", return_value=response) as get, patch("httpx.post") as post:
        result = runner.invoke(app, ["workspace-review", "s1"])
    assert result.exit_code == 1
    assert "workspace=" not in result.output and "modified:" not in result.output
    get.assert_called_once()
    post.assert_not_called()


@pytest.mark.parametrize(
    "status", ["unchanged", "changed", "unavailable", "busy", "not_recorded", "not_tracked"]
)
def test_all_supported_states_preserve_original_response(status):
    data = review() if status == "changed" else {"scope_note": "Unknown evidence."}
    data["status"] = status
    response = httpx.Response(200, json=data, request=httpx.Request("GET", "http://fixture"))
    with patch("httpx.get", return_value=response), patch("httpx.post") as post:
        assert AgentClient().workspace_review("dev", "s1") == data
    post.assert_not_called()


def test_truncated_paths_and_unknown_mode_evidence_are_preserved():
    data = review() | {
        "modified_count": 2,
        "paths_truncated": True,
        "mode_coverage": "not_recorded",
    }
    response = httpx.Response(200, json=data, request=httpx.Request("GET", "http://fixture"))
    with patch("httpx.get", return_value=response):
        assert AgentClient().workspace_review("dev", "s1", max_paths=1) == data


def test_changes_to_ignored_entry_metadata_can_explain_changed_status():
    data = review() | {
        "modified": [],
        "modified_count": 0,
        "ignored_entries": {"status": "changed", "modified": [".git"]},
    }
    response = httpx.Response(200, json=data, request=httpx.Request("GET", "http://fixture"))
    with patch("httpx.get", return_value=response):
        assert AgentClient().workspace_review("dev", "s1") == data
