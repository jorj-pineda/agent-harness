"""Tests for the Typer CLI (cli/).

All tests run offline — no network calls, no real uvicorn process.
- serve:  patches uvicorn.run; verifies args forwarded correctly.
- chat:   patches cli.main.AgentClient; drives the REPL via stdin injection.
- _client: exercises AgentClient with httpx.MockTransport (no real server).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from cli._client import AgentClient, TurnSummary
from cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def test_serve_invokes_uvicorn_defaults() -> None:
    with patch("uvicorn.run") as mock_run:
        result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0, result.output
    mock_run.assert_called_once_with("api.server:app", host="127.0.0.1", port=8000, reload=False)


def test_serve_custom_host_port() -> None:
    with patch("uvicorn.run") as mock_run:
        runner.invoke(app, ["serve", "--host", "0.0.0.0", "--port", "9000"])
    mock_run.assert_called_once_with("api.server:app", host="0.0.0.0", port=9000, reload=False)


def test_serve_reload_flag() -> None:
    with patch("uvicorn.run") as mock_run:
        runner.invoke(app, ["serve", "--reload"])
    mock_run.assert_called_once_with("api.server:app", host="127.0.0.1", port=8000, reload=True)


# ---------------------------------------------------------------------------
# chat — success paths
# ---------------------------------------------------------------------------


def _make_turn(**kwargs: Any) -> MagicMock:
    turn = MagicMock(spec=TurnSummary)
    turn.answer = kwargs.get("answer", "Done.")
    turn.confidence = kwargs.get("confidence", 0.9)
    turn.escalated = kwargs.get("escalated", False)
    turn.provider = kwargs.get("provider", "fake")
    turn.latency_ms = kwargs.get("latency_ms", 10.0)
    turn.files_touched = kwargs.get("files_touched", [])
    return turn


def test_chat_creates_session_and_sends_message() -> None:
    with patch("cli.main.AgentClient") as MockClient:
        instance = MockClient.return_value
        instance.create_session.return_value = "sess-abc"
        instance.chat.return_value = _make_turn(answer="Bug fixed.", files_touched=["foo.py"])

        result = runner.invoke(app, ["chat"], input="fix the bug\n/quit\n")

    assert result.exit_code == 0, result.output
    assert "sess-abc" in result.output
    assert "Bug fixed." in result.output
    assert "foo.py" in result.output
    instance.create_session.assert_called_once_with("cli-user", None)
    instance.chat.assert_called_once_with("cli-user", "sess-abc", "fix the bug", None)


def test_chat_quit_immediately() -> None:
    with patch("cli.main.AgentClient") as MockClient:
        instance = MockClient.return_value
        instance.create_session.return_value = "sess-xyz"

        result = runner.invoke(app, ["chat"], input="/quit\n")

    assert result.exit_code == 0, result.output
    assert "Bye" in result.output
    instance.chat.assert_not_called()


def test_chat_eof_exits_cleanly() -> None:
    with patch("cli.main.AgentClient") as MockClient:
        instance = MockClient.return_value
        instance.create_session.return_value = "sess-eof"

        result = runner.invoke(app, ["chat"], input="")  # immediate EOF

    assert result.exit_code == 0, result.output
    assert "Bye" in result.output


def test_chat_passes_provider_and_workspace() -> None:
    with patch("cli.main.AgentClient") as MockClient:
        instance = MockClient.return_value
        instance.create_session.return_value = "sess-ws"
        instance.chat.return_value = _make_turn()

        runner.invoke(
            app,
            ["chat", "--provider", "anthropic", "--workspace", "/tmp/repo", "--user-id", "u1"],
            input="hello\n/quit\n",
        )

    instance.create_session.assert_called_once_with("u1", "/tmp/repo")
    instance.chat.assert_called_once_with("u1", "sess-ws", "hello", "anthropic")


def test_chat_skips_blank_lines() -> None:
    with patch("cli.main.AgentClient") as MockClient:
        instance = MockClient.return_value
        instance.create_session.return_value = "sess-blank"
        instance.chat.return_value = _make_turn()

        runner.invoke(app, ["chat"], input="\n\nhi\n/quit\n")

    # blank lines should not trigger a chat call
    instance.chat.assert_called_once()
    _, _, call_msg, _ = instance.chat.call_args[0]
    assert call_msg == "hi"


def test_chat_displays_envelope_fields() -> None:
    with patch("cli.main.AgentClient") as MockClient:
        instance = MockClient.return_value
        instance.create_session.return_value = "sess-env"
        instance.chat.return_value = _make_turn(
            confidence=0.45, escalated=True, provider="ollama", latency_ms=999.0
        )

        result = runner.invoke(app, ["chat"], input="task\n/quit\n")

    assert "conf=0.45" in result.output
    assert "low" in result.output  # confidence label
    assert "escalated=True" in result.output
    assert "provider=ollama" in result.output
    assert "999ms" in result.output


# ---------------------------------------------------------------------------
# chat — error paths
# ---------------------------------------------------------------------------


def test_chat_exits_on_session_error() -> None:
    with patch("cli.main.AgentClient") as MockClient:
        instance = MockClient.return_value
        instance.create_session.side_effect = Exception("Connection refused")

        result = runner.invoke(app, ["chat"])

    assert result.exit_code == 1


def test_chat_continues_after_message_error() -> None:
    with patch("cli.main.AgentClient") as MockClient:
        instance = MockClient.return_value
        instance.create_session.return_value = "sess-err"
        instance.chat.side_effect = [Exception("timeout"), _make_turn(answer="Recovered.")]

        result = runner.invoke(app, ["chat"], input="msg1\nmsg2\n/quit\n")

    assert result.exit_code == 0
    assert "Recovered." in result.output


# ---------------------------------------------------------------------------
# AgentClient — unit tests with mocked httpx.post
# ---------------------------------------------------------------------------


def _ok_response(json_data: dict[str, Any]) -> MagicMock:
    """Fake httpx response: 200 OK, no request binding needed."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


def _err_response(status_code: int) -> MagicMock:
    """Fake httpx response that raises HTTPStatusError on raise_for_status."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=MagicMock(),
        response=MagicMock(),
    )
    return resp


def test_agent_client_create_session() -> None:
    with patch("httpx.post", return_value=_ok_response({"session_id": "s-1"})) as mock_post:
        client = AgentClient("http://localhost:8000")
        sid = client.create_session("user1")
    assert sid == "s-1"
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["user_id"] == "user1"
    assert "workspace_root" not in kwargs["json"]


def test_agent_client_create_session_with_workspace() -> None:
    with patch("httpx.post", return_value=_ok_response({"session_id": "s-2"})) as mock_post:
        client = AgentClient("http://localhost:8000")
        client.create_session("u", "/tmp/repo")
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["workspace_root"] == "/tmp/repo"


def test_agent_client_chat_parses_envelope() -> None:
    envelope = {
        "answer": "Fixed.",
        "confidence": 0.85,
        "escalated": False,
        "provider": "ollama",
        "latency_ms": 123.4,
        "files_touched": ["a.py"],
        "tool_calls": [],
    }
    with patch("httpx.post", return_value=_ok_response(envelope)):
        client = AgentClient()
        summary = client.chat("u", "s", "do it")

    assert summary.answer == "Fixed."
    assert summary.confidence == pytest.approx(0.85)
    assert summary.escalated is False
    assert summary.provider == "ollama"
    assert summary.files_touched == ["a.py"]


def test_agent_client_raises_on_http_error() -> None:
    with patch("httpx.post", return_value=_err_response(404)):
        client = AgentClient()
        with pytest.raises(httpx.HTTPStatusError):
            client.chat("u", "bad-session", "hi")
