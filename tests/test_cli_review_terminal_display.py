"""Review-only CLI views must render saved/server text without terminal commands."""

import json
from copy import deepcopy
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()
CONTROL = "café\x1b[2J\r\x08\x07\x7f\x9b31m"


def records():
    return {
        "tool-progress": {
            "run_id": "r1",
            "coverage": "recorded",
            "observed_events": 1,
            "events_truncated": False,
            "scope_note": CONTROL,
            "events": [
                {"sequence": 1, "type": "tool_start", "tool": CONTROL, "observed_at": "today"}
            ],
        },
        "workspace-review": {
            "status": "unavailable",
            "turn_id": CONTROL,
            "reason": CONTROL,
            "ignored_entries": {"status": "not_recorded", "scope_note": CONTROL},
            "scope_note": CONTROL,
        },
        "transcript-review": {
            "status": "invalid",
            "message_count": 1,
            "issue_count": 1,
            "issues": [{"message_index": 0, "code": CONTROL}],
            "scope_note": CONTROL,
        },
        "configuration-review": {
            "status": "unavailable",
            "provider": CONTROL,
            "reason": CONTROL + "\nline\tindent",
            "scope_note": CONTROL,
        },
    }


@pytest.mark.parametrize("command", list(records()))
def test_review_controls_are_visible_without_mutation_or_extra_requests(command):
    data = records()[command]
    original = deepcopy(data)
    with patch("cli.main.AgentClient") as client:
        method = getattr(client.return_value, command.replace("-", "_"))
        method.return_value = data
        result = runner.invoke(
            app, [command, "r1" if command == "tool-progress" else "s1"], color=True
        )
    assert result.exit_code == 0, result.output
    for char in ("\x1b", "\r", "\x08", "\x07", "\x7f", "\x9b"):
        assert char not in result.output
    assert "café" in result.output and "\\u001b[2J" in result.output
    if command == "configuration-review":
        assert "\nline\tindent" in result.output
    assert data == original
    method.assert_called_once()
    assert len(client.return_value.method_calls) == 1


def test_raw_tool_progress_preserves_original_values():
    data = records()["tool-progress"]
    with patch("cli.main.AgentClient") as client:
        client.return_value.tool_progress.return_value = data
        result = runner.invoke(app, ["tool-progress", "r1", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == data
