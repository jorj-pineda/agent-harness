"""Saved evidence must not issue terminal commands; raw JSON stays evidence-preserving."""

import json
from copy import deepcopy
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from cli._client import TurnSummary
from cli.main import app

runner = CliRunner()
CONTROL = "café\x1b[2J\r\x08\x07\x7f\x9b31m"


def saved_run():
    return {
        "run_id": "r1",
        "status": "finished",
        "provider": "ollama",
        "configured_model": CONTROL,
        "request": {
            "user_id": "cli-user",
            "session_id": "s1",
            "request_id": CONTROL,
            "message": CONTROL,
        },
        "error": CONTROL,
        "response": {
            "answer": CONTROL + "\nsecond line",
            "completion_status": "completed",
            "verification_status": "not_run",
            "workspace_changes": {
                "diffs": [{"path": CONTROL, "diff": "-old\n+new\t" + CONTROL + "\n"}]
            },
        },
    }


@pytest.mark.parametrize("command", ["run", "runs", "checkpoint", "checkpoints", "revert-status"])
def test_saved_human_display_escapes_controls_without_mutation_or_post(command):
    run = saved_run()
    checkpoint = {
        "checkpoint_id": "cp1",
        "session_id": "s1",
        "file_count": 1,
        "byte_count": 1,
        "root": CONTROL,
        "captured_at": "today",
        "scope_note": CONTROL,
    }
    revert = {
        "revert_id": "rv1",
        "status": "applied",
        "request": {"run_id": "r1", "path": CONTROL, "request_id": "key1"},
        "reason": CONTROL,
        "scope_note": CONTROL,
    }
    data = {
        "run": run,
        "runs": [run],
        "checkpoint": checkpoint,
        "checkpoints": [checkpoint],
        "revert-status": revert,
    }[command]
    original = deepcopy(data)
    args = {
        "run": ["r1"],
        "runs": [],
        "checkpoint": ["s1", "cp1"],
        "checkpoints": ["s1"],
        "revert-status": ["--request-id", "key1"],
    }[command]
    with patch("cli.main.AgentClient") as client, patch("httpx.post") as post:
        getattr(client.return_value, command.replace("-", "_")).return_value = data
        # color=True avoids Click silently removing ANSI sequences from the evidence.
        result = runner.invoke(app, [command, *args], color=True)
    assert result.exit_code == 0, result.output
    for character in ("\x1b", "\r", "\x08", "\x07", "\x7f", "\x9b"):
        assert character not in result.output
    assert "café" in result.output and "\\u001b[2J" in result.output
    if command == "run":
        assert "\nsecond line" in result.output and "-old\n+new\t" in result.output
    assert data == original
    post.assert_not_called()


@pytest.mark.parametrize(
    "command,args",
    [("run", ["r1"]), ("checkpoint", ["s1", "cp1"]), ("revert-status", ["--request-id", "key1"])],
)
def test_json_display_keeps_original_control_bytes(command, args):
    data = saved_run() if command == "run" else {"saved": CONTROL}
    with patch("cli.main.AgentClient") as client:
        getattr(client.return_value, command.replace("-", "_")).return_value = data
        result = runner.invoke(app, [command, *args, "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == data


def test_live_chat_escapes_turn_evidence_without_altering_the_result():
    turn = TurnSummary(
        answer=CONTROL + "\nsecond line",
        confidence=None,
        escalated=False,
        provider="scripted",
        latency_ms=1,
        tool_errors=[CONTROL],
        workspace_changes={"diffs": [{"path": "a.py", "diff": CONTROL}]},
    )
    original = deepcopy(turn)
    with patch("cli.main.AgentClient") as client:
        client.return_value.create_session.return_value = "s1"
        client.return_value.chat.return_value = turn
        result = runner.invoke(app, ["chat"], input="inspect\n/quit\n")
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output
    assert "café\\u001b" in result.output
    assert "\nsecond line" in result.output
    assert turn == original
