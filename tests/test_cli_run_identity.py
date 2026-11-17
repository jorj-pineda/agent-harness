"""Saved inspection must identify the requested run and scoped lookup filters."""

import json
from unittest.mock import patch

import httpx
import pytest
from typer.testing import CliRunner

from cli._client import AgentClient
from cli.main import app

runner = CliRunner()


def record():
    return {
        "run_id": "r1",
        "status": "interrupted",
        "provider": "ollama",
        "configured_model": None,
        "request": {
            "user_id": "dev",
            "session_id": "s1",
            "request_id": "key1",
            "message": "inspect",
        },
        "response": None,
        "error": None,
        "extra_saved_evidence": {"legacy": "preserve"},
    }


@pytest.mark.parametrize("fault", ["run", "owner", "status", "request", "shape"])
def test_run_lookup_refuses_unrelated_or_unsupported_record_before_display(fault):
    data = record()
    if fault == "run":
        data["run_id"] = "wrong-run"
    elif fault == "owner":
        data["request"]["user_id"] = "other"
    elif fault == "status":
        data["status"] = "unknown-state"
    elif fault == "request":
        data["request"] = None
    else:
        data = ["private-marker"]
    response = httpx.Response(200, json=data, request=httpx.Request("GET", "http://server/runs/r1"))
    with patch("httpx.get", return_value=response) as get, patch("httpx.post") as post:
        result = runner.invoke(app, ["run", "r1", "--user-id", "dev", "--json"])
    assert result.exit_code == 1
    assert "Run lookup failed" in result.output
    assert "private-marker" not in result.output and '"extra_saved_evidence"' not in result.output
    get.assert_called_once()
    post.assert_not_called()


@pytest.mark.parametrize("fault", ["owner", "session", "key", "container", "mixed"])
def test_filtered_list_refuses_mismatches_without_partial_results_or_dispatch(fault):
    data = record()
    body = [data]
    if fault == "owner":
        data["request"]["user_id"] = "other"
    elif fault == "session":
        data["request"]["session_id"] = "other-session"
    elif fault == "key":
        data["request"]["request_id"] = "other-key"
    elif fault == "container":
        body = {"private-marker": data}
    else:
        bad = record()
        bad["request"]["user_id"] = "other"
        body.append(bad)
    response = httpx.Response(200, json=body, request=httpx.Request("GET", "http://server/runs"))
    with patch("httpx.get", return_value=response) as get, patch("httpx.post") as post:
        result = runner.invoke(
            app, ["runs", "--user-id", "dev", "--session-id", "s1", "--request-id", "key1"]
        )
    assert result.exit_code == 1
    assert "Run lookup failed" in result.output
    assert "r1 interrupted" not in result.output and "private-marker" not in result.output
    get.assert_called_once()
    post.assert_not_called()


def test_valid_record_and_empty_list_preserve_existing_json_contract():
    data = record()
    response = httpx.Response(200, json=data, request=httpx.Request("GET", "http://server/runs/r1"))
    with patch("httpx.get", return_value=response), patch("httpx.post") as post:
        result = runner.invoke(app, ["run", "r1", "--user-id", "dev", "--json"])
    assert result.exit_code == 0 and json.loads(result.output) == data
    post.assert_not_called()
    response = httpx.Response(200, json=[], request=httpx.Request("GET", "http://server/runs"))
    with patch("httpx.get", return_value=response):
        assert AgentClient().runs("dev", request_id="missing") == []
