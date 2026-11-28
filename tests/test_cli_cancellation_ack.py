"""Cancellation acknowledgment is separate from cleanup and final saved evidence."""

from unittest.mock import patch

import httpx
import pytest
from typer.testing import CliRunner

from cli._client import AgentClient
from cli.main import app

runner = CliRunner()


@pytest.mark.parametrize(
    "status,body",
    [
        (200, {"status": "requested", "detail": "Requested"}),
        (202, {"detail": "Requested"}),
        (202, {"status": "finished", "detail": "Cleanup done"}),
        (202, {"status": "requested", "detail": None}),
        (202, {"status": "requested", "detail": " "}),
        (202, ["private-marker"]),
    ],
)
def test_invalid_cancel_acknowledgment_never_claims_confirmation_or_retries(status, body):
    response = httpx.Response(
        status, json=body, request=httpx.Request("POST", "http://server/cancel")
    )
    with patch("httpx.post", return_value=response) as post, patch("httpx.get") as get:
        result = runner.invoke(app, ["cancel", "s1"])
    assert result.exit_code == 1
    assert "Cancellation not confirmed" in result.output
    assert "private-marker" not in result.output
    assert post.call_count == 1
    get.assert_not_called()


def test_lost_cancel_response_preserves_uncertainty_without_exception_payload():
    with (
        patch("httpx.post", side_effect=httpx.ReadTimeout("private-marker")) as post,
        patch("httpx.get") as get,
    ):
        result = runner.invoke(app, ["cancel", "s1"])
    assert result.exit_code == 1
    assert "Outcome may be unknown" in result.output
    assert "No automatic retry" in result.output
    assert "private-marker" not in result.output
    post.assert_called_once()
    get.assert_not_called()


def test_interrupt_cancel_failure_has_same_uncertainty_without_replaying_chat():
    with patch("cli.main.AgentClient") as factory:
        client = factory.return_value
        client.create_session.return_value = "s1"
        client.chat.side_effect = KeyboardInterrupt()
        client.cancel.side_effect = httpx.ReadTimeout("private-marker")
        result = runner.invoke(app, ["chat"], input="fix it\n")
    assert result.exit_code == 0
    assert "Cancellation not confirmed" in result.output
    assert "Outcome may be unknown" in result.output
    assert "private-marker" not in result.output
    client.chat.assert_called_once()
    client.cancel.assert_called_once_with("cli-user", "s1")


def test_valid_requested_acknowledgment_preserves_api_detail():
    response = httpx.Response(
        202,
        json={"status": "requested", "detail": "Wait for final evidence."},
        request=httpx.Request("POST", "http://server/cancel"),
    )
    with patch("httpx.post", return_value=response):
        assert AgentClient().cancel("dev", "s1") == "Wait for final evidence."


def test_cli_cancel_real_api_then_restart_reads_final_evidence_preserving_user_work(
    tmp_path, monkeypatch
):
    import asyncio
    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from fastapi.testclient import TestClient

    from tests.api.conftest import ScriptedProvider
    from tests.api.test_session_persistence import make_app

    root = tmp_path / "repo"
    root.mkdir()
    target = root / "user.py"
    target.write_text("pre_existing = 'keep'\n")
    provider = ScriptedProvider()
    started = threading.Event()
    calls = []

    async def hold(*args, **kwargs):
        calls.append(True)
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(provider, "chat", hold)
    with TestClient(make_app(tmp_path, provider)) as api:
        sid = api.post("/sessions", json={"user_id": "dev", "workspace_root": str(root)}).json()[
            "session_id"
        ]
        with ThreadPoolExecutor(max_workers=1) as pool:
            active = pool.submit(
                api.post,
                "/chat",
                json={
                    "user_id": "dev",
                    "session_id": sid,
                    "message": "inspect user.py",
                    "request_id": "cancel-proof",
                },
            )
            try:
                assert started.wait(2)
                with patch(
                    "httpx.post",
                    side_effect=lambda url, **kw: api.post(
                        url.removeprefix("http://localhost:8000"), json=kw["json"]
                    ),
                ) as posted:
                    result = runner.invoke(app, ["cancel", sid, "--user-id", "dev"])
                assert result.exit_code == 0, result.output
                assert "wait for final evidence" in result.output
                posted.assert_called_once()
            finally:
                if not active.done():
                    api.post(f"/sessions/{sid}/cancel", json={"user_id": "dev"})
                final = active.result(timeout=5).json()
        assert final["completion_status"] == "cancelled"
    with TestClient(make_app(tmp_path, ScriptedProvider())) as api:
        with (
            patch(
                "httpx.get",
                side_effect=lambda url, **kw: api.get(
                    url.removeprefix("http://localhost:8000"), params=kw["params"]
                ),
            ) as get,
            patch("httpx.post") as post,
        ):
            reviewed = runner.invoke(app, ["run", final["run_id"], "--user-id", "dev", "--json"])
        assert reviewed.exit_code == 0, reviewed.output
        assert json.loads(reviewed.output)["response"] == final
        get.assert_called_once()
        post.assert_not_called()
    assert len(calls) == 1
    assert target.read_text() == "pre_existing = 'keep'\n"


@pytest.mark.parametrize("interactive", [False, True])
def test_cli_cancellation_ack_escapes_terminal_controls(interactive):
    with patch("cli.main.AgentClient") as factory:
        client = factory.return_value
        client.create_session.return_value = "s1"
        client.chat.side_effect = KeyboardInterrupt()
        client.cancel.return_value = "Requested\x1b[2J"
        result = runner.invoke(app, ["chat"] if interactive else ["cancel", "s1"], input="fix\n")
    assert result.exit_code == 0, result.output
    assert "Requested\\u001b[2J" in result.output
    assert "\x1b" not in result.output
    client.cancel.assert_called_once()
