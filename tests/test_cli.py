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
    turn.run_id = kwargs.get("run_id")
    turn.answer = kwargs.get("answer", "Done.")
    turn.confidence = kwargs.get("confidence", 0.9)
    turn.escalated = kwargs.get("escalated", False)
    turn.provider = kwargs.get("provider", "fake")
    turn.latency_ms = kwargs.get("latency_ms", 10.0)
    turn.workspace_review_ms = kwargs.get("workspace_review_ms")
    turn.files_touched = kwargs.get("files_touched", [])
    turn.completion_status = kwargs.get("completion_status", "completed")
    turn.completion_reason = kwargs.get("completion_reason")
    turn.verification_status = kwargs.get("verification_status", "not_run")
    turn.check_attempts = kwargs.get("check_attempts", [])
    turn.tool_errors = kwargs.get("tool_errors", [])
    turn.token_usage = kwargs.get("token_usage", {})
    turn.workspace_changes = kwargs.get("workspace_changes", {})
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
    assert instance.chat.call_args.args == ("cli-user", "sess-abc", "fix the bug", None)
    assert instance.chat.call_args.kwargs["request_id"] in result.output


def test_chat_shows_observed_workspace_changes() -> None:
    with patch("cli.main.AgentClient") as MockClient:
        instance = MockClient.return_value
        instance.create_session.return_value = "sess-changes"
        instance.chat.side_effect = [
            _make_turn(
                confidence=None,
                workspace_changes={
                    "status": "tracked",
                    "added": ["generated.txt"],
                    "modified": ["calc.py"],
                    "deleted": [],
                    "diffs": [
                        {"path": "calc.py", "diff": "--- a/calc.py\n+++ b/calc.py\n-old\n+new\n"},
                        {"path": "generated.txt", "reason": "Text capture byte limit exceeded"},
                    ],
                },
            ),
            _make_turn(
                workspace_changes={"status": "unavailable", "reason": "Workspace too large"}
            ),
        ]

        result = runner.invoke(app, ["chat"], input="fix\nagain\n/quit\n")

    assert result.exit_code == 0, result.output
    assert "workspace_added=['generated.txt']" in result.output
    assert "legacy_evidence_score=n/a" in result.output
    assert "workspace_modified=['calc.py']" in result.output
    assert "workspace_deleted" not in result.output
    assert "--- a/calc.py\n+++ b/calc.py\n-old\n+new" in result.output
    assert "diff omitted: Text capture byte limit exceeded" in result.output
    assert "workspace_changes=unavailable (Workspace too large)" in result.output


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
    assert instance.chat.call_args.args == ("u1", "sess-ws", "hello", "anthropic")


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

    assert "legacy_evidence_score=0.45" in result.output
    assert "not patch correctness" in result.output
    assert "escalated=True" in result.output
    assert "provider=ollama" in result.output
    assert "999ms" in result.output


@pytest.mark.parametrize("verification", ["failed", "stale", "not_run"])
def test_high_legacy_score_does_not_hide_partial_work(verification: str) -> None:
    with patch("cli.main.AgentClient") as mock_client:
        client = mock_client.return_value
        client.create_session.return_value = "partial"
        client.chat.return_value = _make_turn(
            confidence=1.0,
            completion_status="incomplete",
            completion_reason="Remaining work requires review.",
            verification_status=verification,
            check_attempts=[
                {
                    "argv": ["pytest", "-q"],
                    "status": "unavailable",
                    "error": "Executable unavailable",
                }
            ],
            tool_errors=["read_file: missing"],
        )
        result = runner.invoke(app, ["chat"], input="fix\n/quit\n")
    assert result.exit_code == 0, result.output
    assert f"status=incomplete checks={verification}" in result.output
    assert "Remaining work requires review." in result.output
    assert "check_error=Executable unavailable" in result.output
    assert "tool_error=read_file: missing" in result.output
    assert "legacy_evidence_score=1.00 (tool heuristic; not patch correctness)" in result.output
    assert result.output.index("status=incomplete") < result.output.index("legacy_evidence_score")
    assert "(high)" not in result.output


# ---------------------------------------------------------------------------
# chat — error paths
# ---------------------------------------------------------------------------


def test_chat_exits_on_session_error() -> None:
    with patch("cli.main.AgentClient") as MockClient:
        instance = MockClient.return_value
        instance.create_session.side_effect = Exception("Connection refused")

        result = runner.invoke(app, ["chat"])

    assert result.exit_code == 1


def test_chat_retains_key_and_blocks_new_message_after_error() -> None:
    with patch("cli.main.AgentClient") as MockClient:
        instance = MockClient.return_value
        instance.create_session.return_value = "sess-err"
        instance.chat.side_effect = [Exception("timeout"), _make_turn(answer="Recovered.")]

        result = runner.invoke(app, ["chat"], input="msg1\nmsg2\n/retry\n/quit\n")

    assert result.exit_code == 0
    assert "Recovered." in result.output
    assert "Outcome unknown" in result.output
    assert instance.chat.call_count == 2
    assert instance.chat.call_args_list[0] == instance.chat.call_args_list[1]


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
        "completion_status": "incomplete",
        "completion_reason": "Check failed.",
        "verification_status": "failed",
        "check_attempts": [{"argv": ["pytest", "-q"], "status": "failed"}],
        "tool_errors": ["read_file: missing"],
        "token_usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "workspace_changes": {"status": "tracked", "modified": ["a.py"]},
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
    assert summary.completion_status == "incomplete"
    assert summary.check_attempts[0]["status"] == "failed"
    assert summary.tool_errors == ["read_file: missing"]
    assert summary.token_usage == {"prompt_tokens": 10, "completion_tokens": 5}
    assert summary.workspace_changes == {"status": "tracked", "modified": ["a.py"]}


def test_agent_client_raises_on_http_error() -> None:
    with patch("httpx.post", return_value=_err_response(404)):
        client = AgentClient()
        with pytest.raises(httpx.HTTPStatusError):
            client.chat("u", "bad-session", "hi")


@pytest.mark.parametrize("envelope", [{"confidence": None}, {}])
def test_agent_client_preserves_unavailable_confidence(envelope: dict) -> None:
    with patch("httpx.post", return_value=_ok_response(envelope)):
        summary = AgentClient().chat("u", "s", "inspect")
    assert summary.confidence is None


def test_cancel_command_and_ctrl_c_request_explicit_cancellation() -> None:
    with patch("cli.main.AgentClient") as mock:
        client = mock.return_value
        client.cancel.return_value = "Cancellation requested; wait for cleanup."
        result = runner.invoke(app, ["cancel", "s1", "--user-id", "dev"])
        assert result.exit_code == 0, result.output
        client.cancel.assert_called_once_with("dev", "s1")
        assert "wait for cleanup" in result.output
        client.reset_mock()
        client.create_session.return_value = "s2"
        client.chat.side_effect = KeyboardInterrupt
        result = runner.invoke(app, ["chat"], input="fix\n")
        assert result.exit_code == 0, result.output
        client.cancel.assert_called_once_with("cli-user", "s2")
        assert "Inspect saved session review" in result.output


def test_client_cancel_uses_owner_scoped_endpoint() -> None:
    with patch("httpx.post", return_value=_ok_response({"detail": "Requested"})) as post:
        assert AgentClient().cancel("dev", "s1") == "Requested"
    assert post.call_args.args[0] == "http://localhost:8000/sessions/s1/cancel"
    assert post.call_args.kwargs["json"] == {"user_id": "dev"}


def _saved_run(status="finished", response=None):
    return {
        "run_id": "r1",
        "status": status,
        "provider": "ollama",
        "configured_model": "local",
        "request": {"session_id": "s1", "request_id": "key1", "message": "Fix"},
        "response": response,
        "error": "Workspace edits may exist" if status == "interrupted" else None,
    }


def test_check_recovers_review_without_resubmission() -> None:
    with patch("cli.main.AgentClient") as mock:
        client = mock.return_value
        client.create_session.return_value = "s1"
        client.chat.side_effect = httpx.ReadTimeout("lost response")
        client.runs.return_value = [
            _saved_run(
                response={
                    "answer": "Partial",
                    "completion_status": "incomplete",
                    "verification_status": "failed",
                    "workspace_changes": {"diffs": [{"path": "a.py", "diff": "-old\n+new\n"}]},
                }
            )
        ]
        result = runner.invoke(app, ["chat"], input="Fix\n/check\nnew edit\n")
    assert result.exit_code == 0, result.output
    assert "status=incomplete checks=failed" in result.output
    assert "-old\n+new" in result.output
    assert "Review only" in result.output
    client.chat.assert_called_once()
    client.runs.assert_called_once_with(
        "cli-user", request_id=client.chat.call_args.kwargs["request_id"]
    )


@pytest.mark.parametrize("status", ["running", "failed", "interrupted", "cancelled"])
def test_history_commands_only_read_and_label_run_state(status: str) -> None:
    with patch("cli.main.AgentClient") as mock:
        client = mock.return_value
        record = _saved_run(status)
        client.runs.return_value = [record]
        client.run.return_value = record
        listing = runner.invoke(app, ["runs", "--user-id", "dev", "--request-id", "key1"])
        detail = runner.invoke(app, ["run", "r1", "--user-id", "dev"])
        raw = runner.invoke(app, ["run", "r1", "--user-id", "dev", "--json"])
    assert listing.exit_code == detail.exit_code == raw.exit_code == 0
    assert f"r1 {status}" in listing.output
    assert f"status={status}" in detail.output
    assert "No saved final response" in detail.output
    assert '"run_id": "r1"' in raw.output
    client.chat.assert_not_called()
    client.create_session.assert_not_called()
    client.cancel.assert_not_called()


def test_client_history_requests_are_gets_and_chat_retains_explicit_key() -> None:
    with patch("httpx.get", return_value=_ok_response([])) as get:
        AgentClient().runs("dev", session_id="s1", request_id="key1", limit=20, offset=2)
        assert get.call_args.kwargs["params"] == {
            "user_id": "dev",
            "session_id": "s1",
            "request_id": "key1",
            "limit": 20,
            "offset": 2,
        }
    with patch("httpx.get", return_value=_ok_response(_saved_run())) as get:
        AgentClient().run("dev", "r1")
        assert get.call_args.args[0].endswith("/runs/r1")
        assert get.call_args.kwargs["params"] == {"user_id": "dev"}
    with patch("httpx.post", return_value=_ok_response({"run_id": "r1"})) as post:
        summary = AgentClient().chat("dev", "s1", "Fix", "ollama", request_id="retained")
        assert post.call_args.kwargs["json"]["request_id"] == "retained"
        assert summary.run_id == "r1"


def test_history_lookup_failure_does_not_start_work() -> None:
    with patch("cli.main.AgentClient") as mock:
        client = mock.return_value
        client.run.side_effect = httpx.ConnectError("offline")
        result = runner.invoke(app, ["run", "missing"])
    assert result.exit_code == 1
    assert "Run lookup failed" in result.output
    client.chat.assert_not_called()


def test_cli_lost_response_retry_reuses_one_real_tool_turn(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from providers.base import ToolCall
    from tests.api.conftest import ScriptedProvider, make_response
    from tests.api.test_session_persistence import make_app

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("user = 1\n")
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "a.py", "content": "user = 1\nagent = 2\n"},
                )
            ]
        ),
        make_response(content="Saved review"),
    )
    chat_requests = []
    with TestClient(make_app(tmp_path, provider)) as api_client:

        def post(url, **kwargs):
            response = api_client.post(url, json=kwargs["json"])
            if url.endswith("/chat"):
                chat_requests.append(kwargs["json"])
                if len(chat_requests) == 1:
                    raise httpx.ReadTimeout("Lost final response after archive commit")
            return response

        with patch("httpx.post", side_effect=post):
            result = runner.invoke(
                app, ["chat", "--workspace", str(repo)], input="Update a.py\n/retry\n"
            )
        assert result.exit_code == 0, result.output
        assert "Saved review" in result.output
        assert "Recovered result is for review" in result.output
        assert "--- a/a.py" in result.output
        assert chat_requests[0] == chat_requests[1]
        record = api_client.get(
            "/runs", params={"user_id": "cli-user", "request_id": chat_requests[0]["request_id"]}
        ).json()
        assert len(record) == 1
        assert len(provider.calls) == 2
        assert (repo / "a.py").read_text() == "user = 1\nagent = 2\n"


def test_workspace_review_command_only_reads_and_explains_scope():
    review = {
        "status": "changed",
        "turn_id": "t1",
        "modified": ["a.py"],
        "modified_count": 2,
        "paths_truncated": True,
        "ignored_entries": {
            "status": "changed",
            "modified": [".git"],
            "modified_count": 2,
            "paths_truncated": True,
            "scope_note": "Metadata only; rewriting existing children may be missed.",
        },
        "excluded_patterns": [".git", ".env"],
        "scope_note": "Read-only review, not permission to resume or revert.",
    }
    with patch("httpx.get", return_value=_ok_response(review)) as get, patch("httpx.post") as post:
        result = runner.invoke(
            app, ["workspace-review", "s1", "--user-id", "dev", "--max-paths", "1"]
        )
    assert result.exit_code == 0, result.output
    assert "workspace=changed baseline_turn=t1" in result.output
    assert "ignored top-level metadata: changed" in result.output
    assert "modified: ['.git'] (total 2)" in result.output
    assert "rewriting existing children" in result.output
    assert "Regular-file mode coverage: not_recorded" in result.output
    assert "modified: ['a.py'] (total 2)" in result.output
    assert "Path list truncated" in result.output
    assert "not permission" in result.output
    assert get.call_args.args[0].endswith("/sessions/s1/workspace-review")
    assert get.call_args.kwargs["params"] == {"user_id": "dev", "max_paths": 1}
    post.assert_not_called()


def test_transcript_review_only_reads_with_bounded_diagnostics():
    review = {
        "status": "invalid",
        "message_count": 5,
        "issue_count": 2,
        "issues": [{"message_index": 1, "code": "missing_tool_result"}],
        "issues_truncated": True,
        "scope_note": "Read-only, not permission to resume.",
    }
    with patch("httpx.get", return_value=_ok_response(review)) as get, patch("httpx.post") as post:
        result = runner.invoke(
            app, ["transcript-review", "s1", "--user-id", "dev", "--max-issues", "1"]
        )
    assert result.exit_code == 0, result.output
    assert "transcript=invalid messages=5 issues=2" in result.output
    assert "message[1]: missing_tool_result" in result.output
    assert "Issue list truncated" in result.output
    assert "not permission" in result.output
    assert get.call_args.args[0].endswith("/sessions/s1/transcript-review")
    assert get.call_args.kwargs["params"] == {"user_id": "dev", "max_issues": 1}
    post.assert_not_called()


def test_run_command_shows_configured_profile_and_keeps_json_read_only():
    from harness.config import Settings
    from harness.execution_profile import capture_execution_profile
    from tools import ToolRegistry

    record = _saved_run()
    record["execution_profile"] = capture_execution_profile(
        Settings(_env_file=None, max_context_tokens=8192), "ollama", ToolRegistry()
    ).model_dump(mode="json")
    with patch("cli.main.AgentClient") as mock:
        mock.return_value.run.return_value = record
        result = runner.invoke(app, ["run", "r1"])
        raw = runner.invoke(app, ["run", "r1", "--json"])
        mock.return_value.chat.assert_not_called()
    assert result.exit_code == 0, result.output
    assert "execution_profile=v1" in result.output
    assert "context_limit=8192" in result.output
    assert "Configured settings only" in result.output
    assert '"execution_profile"' in raw.output


def test_configuration_review_calls_only_read_only_computation_endpoint():
    review = {
        "status": "reviewed",
        "provider": "ollama",
        "configuration": {
            "status": "changed",
            "changed_fields": ["settings.max_context_tokens"],
            "unknown_fields": [],
        },
        "transcript": {"status": "valid"},
        "context": {
            "status": "blocked",
            "estimated_prompt_tokens": 500,
            "includes_proposed_message": True,
            "reason": "Does not fit",
        },
        "scope_note": "Read-only review, not resume permission.",
    }
    with patch("httpx.post", return_value=_ok_response(review)) as post, patch("httpx.get") as get:
        result = runner.invoke(
            app,
            [
                "configuration-review",
                "s1",
                "--user-id",
                "dev",
                "--provider",
                "ollama",
                "--message",
                "next task",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "configuration=changed" in result.output
    assert "context=blocked" in result.output
    assert "not resume permission" in result.output
    assert post.call_count == 1
    assert post.call_args.args[0].endswith("/sessions/s1/configuration-review")
    assert post.call_args.kwargs["json"] == {
        "user_id": "dev",
        "provider": "ollama",
        "message": "next task",
    }
    get.assert_not_called()


@pytest.mark.parametrize("value", [None, 0, 12.5])
def test_review_timing_is_separate_and_nullable(value, capsys):
    from cli._client import summarize
    from cli.main import show_turn

    data = {"answer": "Done", "provider": "scripted", "latency_ms": 20}
    if value is not None:
        data["workspace_review_ms"] = value
    result = summarize(data)
    show_turn(result)
    text = capsys.readouterr().out
    assert "latency=20ms" in text
    if value is None:
        assert "workspace_review=unmeasured" in text
    else:
        assert f"workspace_review={value:.0f}ms (separate from loop latency)" in text
