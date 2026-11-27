"""Checkpoint metadata must belong to the requested session and selected capture."""

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
        "checkpoint_id": "cp1",
        "session_id": "s1",
        "file_count": 1,
        "byte_count": 7,
        "root": "/saved/workspace",
        "captured_at": "2026-11-17T00:00:00Z",
        "restore_supported": False,
        "scope_note": "Saved metadata only; no restore permission.",
        "extra_saved_evidence": {"legacy": "preserve"},
    }


@pytest.mark.parametrize("command", ["checkpoint-create", "checkpoint", "checkpoints"])
@pytest.mark.parametrize(
    "fault", ["session", "identifier", "container", "bytes", "files", "root", "time", "restore"]
)
def test_checkpoint_response_refused_before_display_without_retry(command, fault):
    data = record()
    if fault == "session":
        data["session_id"] = "other-session"
    elif fault == "identifier":
        data["checkpoint_id"] = None
    elif fault == "bytes":
        data["byte_count"] = -1
    elif fault == "files":
        data["file_count"] = True
    elif fault == "root":
        data.pop("root")
    elif fault == "time":
        data["captured_at"] = None
    elif fault == "restore":
        data["restore_supported"] = True
    else:
        data = ["private-marker"]
    body = [data] if command == "checkpoints" and fault != "container" else data
    response = httpx.Response(
        201 if command == "checkpoint-create" else 200,
        json=body,
        request=httpx.Request("GET", "http://server/checkpoints"),
    )
    args = [command, "s1", *(["cp1"] if command == "checkpoint" else []), "--json"]
    with (
        patch("httpx.get", return_value=response) as get,
        patch("httpx.post", return_value=response) as post,
    ):
        result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "private-marker" not in result.output and '"extra_saved_evidence"' not in result.output
    if command == "checkpoint-create":
        assert "Outcome may be unknown" in result.output
        post.assert_called_once()
        get.assert_not_called()
    else:
        get.assert_called_once()
        post.assert_not_called()


def test_detail_cannot_substitute_different_checkpoint_in_same_session():
    data = record()
    data["checkpoint_id"] = "cp2"
    response = httpx.Response(
        200, json=data, request=httpx.Request("GET", "http://server/checkpoint")
    )
    with patch("httpx.get", return_value=response) as get, patch("httpx.post") as post:
        result = runner.invoke(app, ["checkpoint", "s1", "cp1", "--json"])
    assert result.exit_code == 1 and '"checkpoint_id"' not in result.output
    get.assert_called_once()
    post.assert_not_called()


def test_mixed_session_list_does_not_partially_display_valid_metadata():
    bad = record()
    bad["session_id"] = "other"
    response = httpx.Response(
        200, json=[record(), bad], request=httpx.Request("GET", "http://server/checkpoints")
    )
    with patch("httpx.get", return_value=response):
        result = runner.invoke(app, ["checkpoints", "s1"])
    assert result.exit_code == 1 and "checkpoint=cp1" not in result.output


def test_valid_metadata_is_returned_unchanged_and_empty_page_stays_empty():
    data = record()
    response = httpx.Response(
        200, json=data, request=httpx.Request("GET", "http://server/checkpoint")
    )
    with patch("httpx.get", return_value=response):
        result = runner.invoke(app, ["checkpoint", "s1", "cp1", "--json"])
    assert result.exit_code == 0 and json.loads(result.output) == data
    response = httpx.Response(
        200, json=[], request=httpx.Request("GET", "http://server/checkpoints")
    )
    with patch("httpx.get", return_value=response), patch("httpx.post") as post:
        assert AgentClient().checkpoints("dev", "s1") == []
    post.assert_not_called()
