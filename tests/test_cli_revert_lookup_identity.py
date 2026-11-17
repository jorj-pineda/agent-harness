"""A keyed journal lookup never proves another owner's file action succeeded."""

import json
from unittest.mock import patch

import httpx
import pytest
from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


def record(status="applied"):
    return {
        "revert_id": "rv1",
        "status": status,
        "request": {"user_id": "dev", "run_id": "r1", "path": "a.py", "request_id": "key1"},
        "reason": None,
        "scope_note": "Saved one-file action only; not current workspace state.",
        "extra_saved_evidence": {"legacy": "preserve"},
    }


@pytest.mark.parametrize(
    "fault", ["owner", "key", "status", "request", "identifier", "scope", "container"]
)
def test_revert_lookup_refuses_unrelated_or_malformed_record_before_display(fault):
    data = record()
    if fault == "owner":
        data["request"]["user_id"] = "other"
    elif fault == "key":
        data["request"]["request_id"] = "other-key"
    elif fault == "status":
        data["status"] = "finished"
    elif fault == "request":
        data["request"] = None
    elif fault == "identifier":
        data["revert_id"] = 42
    elif fault == "scope":
        data.pop("scope_note")
    else:
        data = ["private-marker"]
    response = httpx.Response(
        200, json=data, request=httpx.Request("GET", "http://server/file-reverts")
    )
    with patch("httpx.get", return_value=response) as get, patch("httpx.post") as post:
        result = runner.invoke(
            app, ["revert-status", "--request-id", "key1", "--user-id", "dev", "--json"]
        )
    assert result.exit_code == 1
    assert "Revert lookup failed" in result.output
    assert "private-marker" not in result.output and '"extra_saved_evidence"' not in result.output
    get.assert_called_once()
    post.assert_not_called()


@pytest.mark.parametrize("status", ["pending", "applied", "rejected", "uncertain", "interrupted"])
def test_matching_journal_states_are_read_only_and_json_is_unchanged(status):
    data = record(status)
    response = httpx.Response(
        200, json=data, request=httpx.Request("GET", "http://server/file-reverts")
    )
    with patch("httpx.get", return_value=response) as get, patch("httpx.post") as post:
        result = runner.invoke(
            app, ["revert-status", "--request-id", "key1", "--user-id", "dev", "--json"]
        )
    assert result.exit_code == 0 and json.loads(result.output) == data
    get.assert_called_once()
    post.assert_not_called()


def test_missing_journal_remains_null_without_submitting_any_action():
    response = httpx.Response(
        200, content=b"null", request=httpx.Request("GET", "http://server/file-reverts")
    )
    with patch("httpx.get", return_value=response) as get, patch("httpx.post") as post:
        result = runner.invoke(
            app, ["revert-status", "--request-id", "key1", "--user-id", "dev", "--json"]
        )
    assert result.exit_code == 0 and json.loads(result.output) is None
    get.assert_called_once()
    post.assert_not_called()
