"""Run progress must refer to the selected run and supported read-only coverage."""

import json
from unittest.mock import patch

import httpx
import pytest
from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


def progress():
    return {
        "run_id": "r1",
        "coverage": "recorded",
        "read_only": True,
        "events": [],
        "observed_events": 0,
        "events_truncated": False,
        "scope_note": "Metadata only; not permission to resume.",
        "optional_evidence": {"preserve": True},
    }


@pytest.mark.parametrize(
    "field,value",
    [("run_id", "other"), ("run_id", None), ("coverage", "complete"), ("read_only", False)],
)
def test_invalid_progress_is_rejected_before_json_display(field, value):
    data = progress()
    data[field] = value
    response = httpx.Response(200, json=data, request=httpx.Request("GET", "http://fixture"))
    with patch("httpx.get", return_value=response) as get, patch("httpx.post") as post:
        result = runner.invoke(app, ["tool-progress", "r1", "--json"])
    assert result.exit_code == 1
    assert '"optional_evidence"' not in result.output
    assert '"run_id"' not in result.output
    get.assert_called_once()
    post.assert_not_called()


@pytest.mark.parametrize("coverage", ["recorded", "not_recorded"])
def test_supported_progress_preserves_original_json_without_submission(coverage):
    data = progress()
    data["coverage"] = coverage
    response = httpx.Response(200, json=data, request=httpx.Request("GET", "http://fixture"))
    with patch("httpx.get", return_value=response), patch("httpx.post") as post:
        result = runner.invoke(app, ["tool-progress", "r1", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == data
    post.assert_not_called()
