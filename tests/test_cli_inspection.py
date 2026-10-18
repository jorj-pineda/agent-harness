from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cli.main import app, show_tool_policy
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tests.api.test_session_persistence import make_app

runner = CliRunner()


@pytest.mark.parametrize(
    "response",
    [
        {"session_id": "old-server"},
        {"session_id": "s", "inspection_only": False},
        {"session_id": "s", "inspection_only": "true"},
        {"session_id": "s", "inspection_only": 1},
        {"session_id": "s", "inspection_only": None},
        {"inspection_only": True},
        {"session_id": "", "inspection_only": True},
        {"session_id": 42, "inspection_only": True},
        ["private-marker"],
    ],
)
def test_cli_inspection_refuses_unacknowledged_policy_without_chat_or_retry(response):
    reply = httpx.Response(
        200, json=response, request=httpx.Request("POST", "http://server/sessions")
    )
    with patch("httpx.post", return_value=reply) as post, patch("httpx.get") as get:
        result = runner.invoke(app, ["chat", "--inspection-only"], input="Edit a.py\n/retry\n")
    assert result.exit_code == 1, result.output
    assert "Server did not acknowledge an inspection-only session" in result.output
    assert "No chat was sent" in result.output and "No automatic retry" in result.output
    assert "Connected" not in result.output and "private-marker" not in result.output
    assert post.call_count == 1
    assert post.call_args.args[0].endswith("/sessions")
    assert post.call_args.kwargs["json"] == {"user_id": "cli-user", "inspection_only": True}
    get.assert_not_called()


@pytest.mark.parametrize("action", ["read", "write"])
def test_cli_inspection_real_api_then_restart_review_preserves_policy_and_user_file(
    tmp_path, action
):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "a.py"
    target.write_text("user = 1\n")
    provider = ScriptedProvider()
    if action == "read":
        provider.script(
            make_response(
                tool_calls=[ToolCall(id="r", name="read_file", arguments={"path": "a.py"})]
            ),
            make_response(content="The module assigns user = 1."),
        )
    else:
        provider.script(
            make_response(
                tool_calls=[
                    ToolCall(
                        id="w", name="write_file", arguments={"path": "a.py", "content": "bad"}
                    )
                ]
            )
        )
    with TestClient(make_app(tmp_path, provider)) as api:
        payloads = []

        def post(url, **kwargs):
            payloads.append(kwargs["json"])
            return api.post(url.removeprefix("http://localhost:8000"), json=kwargs["json"])

        with patch("httpx.post", side_effect=post):
            result = runner.invoke(
                app,
                [
                    "chat",
                    "--user-id",
                    "dev",
                    "--workspace",
                    str(root),
                    "--inspection-only",
                    "--checkpoints",
                ],
                input="Explain a.py\n/quit\n",
            )
        assert result.exit_code == 0, result.output
        assert "inspection only (server acknowledged)" in result.output
        assert "Not host process isolation" in result.output
        assert (
            "status=completed checks=not_run" in result.output
            if action == "read"
            else "status=blocked checks=not_run" in result.output
        )
        assert payloads[0] == {
            "user_id": "dev",
            "workspace_root": str(root),
            "checkpoint_turns": True,
            "inspection_only": True,
        }
        assert len(payloads) == 2
        record = api.get("/runs", params={"user_id": "dev"}).json()[0]
        sid = record["request"]["session_id"]
        saved = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
    calls = len(provider.calls)
    with TestClient(make_app(tmp_path, provider)) as api:

        def get(url, **kwargs):
            return api.get(url.removeprefix("http://localhost:8000"), params=kwargs["params"])

        with patch("httpx.get", side_effect=get) as get_calls, patch("httpx.post") as posted:
            reviewed = runner.invoke(app, ["run", record["run_id"], "--user-id", "dev"])
        assert reviewed.exit_code == 0, reviewed.output
        assert "saved_tool_policy=inspection_only" in reviewed.output
        assert "not host isolation or permission to continue" in reviewed.output
        assert get_calls.call_count == 1
        posted.assert_not_called()
        assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
    assert len(provider.calls) == calls
    assert target.read_text() == "user = 1\n"


@pytest.mark.parametrize("value", [None, "true\x1b[2J", 1, [], {}])
def test_cli_saved_policy_missing_or_malformed_is_unknown(value, capsys):
    show_tool_policy(value)
    output = capsys.readouterr().out
    assert "saved_tool_policy=unrecorded_or_unsupported" in output
    assert "\x1b" not in output


def test_cli_saved_policy_false_is_explicit_standard(capsys):
    show_tool_policy(False)
    assert "saved_tool_policy=standard" in capsys.readouterr().out
