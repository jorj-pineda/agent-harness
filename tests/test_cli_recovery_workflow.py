"""Scripted-provider client workflow, not live-model or release acceptance evidence."""

from __future__ import annotations

import asyncio
import json
import re
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cli.main import app
from providers.base import ToolCall
from tests.api.conftest import make_response
from tests.api.test_continuation import async_wait, known_provider
from tests.api.test_session_persistence import make_app


def test_cli_partial_edit_cancel_restart_fork_and_reverse_both_turns(tmp_path, monkeypatch):
    runner = CliRunner()
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "a.py"
    before = "value = -1\n# pre-existing user line\n"
    partial = before.replace("-1", "0")
    final = before.replace("-1", "1")
    target.write_text(before)
    target.chmod(0o640)
    sentinel = root / "notes.txt"
    sentinel.write_text("pre-existing unrelated work\n")
    (root / "test_visible.py").write_text(
        "from a import value\n\ndef test_value():\n    assert value == 1\n"
    )
    visible = (root / "test_visible.py").read_bytes()
    settings = {
        "ollama_model": "gemma4:12b",  # Configured identity only; no Ollama client.
        "ollama_num_ctx": 16384,
        "max_context_tokens": 16384,
        "coding_toolset": "whole_file",
        "coding_prompt_profile": "compact",
        "max_turn_wall_seconds": 150,
        "max_total_tokens_per_turn": 32000,
        "max_completion_tokens_per_turn": 4096,
        "max_tool_calls_per_turn": 12,
        "max_tool_iterations": 6,
        "max_request_output_tokens": 2048,
        "min_request_output_tokens": 256,
        "require_verification_before_finish": True,
        "project_check_argv": ["python", "-m", "pytest", "-q", "test_visible.py"],
    }
    provider = known_provider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="partial", name="write_file", arguments={"path": "a.py", "content": partial}
                )
            ],
            prompt_tokens=100,
            completion_tokens=20,
        )
    )
    after_edit = threading.Event()
    original_chat = provider.chat

    async def hold_after_edit(*args, **kwargs):
        if provider.calls:
            after_edit.set()
            await asyncio.Event().wait()
        return await original_chat(*args, **kwargs)

    monkeypatch.setattr(provider, "chat", hold_after_edit)
    requests = []
    active = None

    with (
        ThreadPoolExecutor(max_workers=1) as pool,
        TestClient(make_app(tmp_path, provider, **settings)) as api,
    ):

        def post(url, **kwargs):
            nonlocal active
            path = url.removeprefix("http://localhost:8000")
            requests.append((path, kwargs["json"]))
            if path == "/chat":
                active = pool.submit(api.post, path, json=kwargs["json"])
                assert after_edit.wait(5), "Scripted provider never reached post-edit request"
                raise httpx.ReadTimeout("Simulated lost response; server still owns turn")
            return api.post(path, json=kwargs["json"])

        with patch("httpx.post", side_effect=post):
            result = runner.invoke(
                app,
                [
                    "chat",
                    "--workspace",
                    str(root),
                    "--checkpoints",
                    "--response-timeout",
                    "360",
                ],
                input="Set value to 1, preserve the user line and run the visible check\n/quit\n",
            )
            assert result.exit_code == 0, result.output
            assert "Outcome unknown" in result.output
            sid = re.search(r"session=([0-9a-f]{32})", result.output).group(1)
            key = re.search(r"request_id=([0-9a-f]{32})", result.output).group(1)
            assert target.read_text() == partial
            cancelled = runner.invoke(app, ["cancel", sid])
            assert cancelled.exit_code == 0, cancelled.output
        assert active is not None
        response = active.result(timeout=5).json()
        assert response["completion_status"] == "cancelled"
        assert response["verification_status"] == "not_run"
        assert response["token_usage"] == {"prompt_tokens": None, "completion_tokens": None}
        assert [path for path, _ in requests] == [
            "/sessions",
            "/chat",
            f"/sessions/{sid}/cancel",
        ]
        assert requests[1][1]["request_id"] == key
        saved = api.get(f"/sessions/{sid}", params={"user_id": "cli-user"}).json()
        source_run = api.get(f"/runs/{response['run_id']}", params={"user_id": "cli-user"}).json()
        assert source_run["before_checkpoint_id"] and source_run["after_checkpoint_id"]
        events = source_run["budget_evidence"]["events"]
        assert events[-1]["kind"] == "request_failure" and events[-1]["usage"] is None
        assert events[1]["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}

    resumed = known_provider()
    resumed.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="finish", name="write_file", arguments={"path": "a.py", "content": final}
                )
            ],
            prompt_tokens=200,
            completion_tokens=30,
        ),
        make_response(
            tool_calls=[
                ToolCall(
                    id="check",
                    name="run_command",
                    arguments={"argv": settings["project_check_argv"]},
                )
            ],
            prompt_tokens=300,
            completion_tokens=30,
        ),
        make_response(
            content="Visible check passed. Scripted workflow only.",
            prompt_tokens=400,
            completion_tokens=20,
        ),
    )
    with TestClient(make_app(tmp_path, resumed, **settings)) as api:

        def post(url, **kwargs):
            return api.post(url.removeprefix("http://localhost:8000"), json=kwargs["json"])

        def get(url, **kwargs):
            return api.get(url.removeprefix("http://localhost:8000"), params=kwargs["params"])

        def cli(arguments, *, raw=False):
            with patch("httpx.post", side_effect=post), patch("httpx.get", side_effect=get):
                result = runner.invoke(app, arguments)
            assert result.exit_code == 0, result.output
            return json.loads(result.output) if raw else result.output

        assert "workspace=unchanged" in cli(["workspace-review", sid])
        assert "transcript=valid" in cli(["transcript-review", sid])
        assert "configuration=matching" in cli(
            ["configuration-review", sid, "--message", "Continue the same task"]
        )
        assert resumed.calls == []  # Saved review grants no execution.
        continuation = [
            "continue-session",
            sid,
            "Continue the same task",
            "--expected-turn-id",
            saved["session"]["turns"][-1]["turn_id"],
            "--request-id",
            "phase-two",
            "--acknowledge-uncaptured-state",
            "--json",
        ]
        submitted = cli(continuation, raw=True)
        api.portal.call(async_wait, api.app.state.components, submitted["run_id"])
        fork = cli(["run", submitted["run_id"], "--json"], raw=True)
        assert fork["response"]["completion_status"] == "completed"
        assert fork["response"]["verification_status"] == "passed"
        assert fork["execution_profile"] == source_run["execution_profile"]
        assert fork["before_checkpoint_id"] != source_run["before_checkpoint_id"]
        assert target.read_text() == final
        assert fork["response"]["check_attempts"][0]["exit_code"] == 0
        assert cli(["run", response["run_id"], "--json"], raw=True) == source_run
        assert api.get(f"/sessions/{sid}", params={"user_id": "cli-user"}).json() == saved
        sentinel.write_text("pre-existing unrelated work\nlater user edit\n")
        reversals = []
        # Reverse the fork first, then the source. Neither operation resets the repo.
        for index, run in enumerate((fork, source_run)):
            record = cli(
                [
                    "revert-file",
                    run["run_id"],
                    "a.py",
                    "--request-id",
                    f"reverse-{index}",
                    "--json",
                ],
                raw=True,
            )
            assert record["status"] == "applied"
            reversals.append(record)
            assert target.read_text() == (partial if index == 0 else before)
        assert sentinel.read_text().endswith("later user edit\n")
        assert stat.S_IMODE(target.stat().st_mode) == 0o640
        assert (root / "test_visible.py").read_bytes() == visible
        assert len(resumed.calls) == 3

    idle = known_provider()
    with TestClient(make_app(tmp_path, idle, **settings)) as api:
        with patch("httpx.get", side_effect=get), patch("httpx.post") as posted:
            for index, expected in enumerate(reversals):
                result = runner.invoke(
                    app, ["revert-status", "--request-id", f"reverse-{index}", "--json"]
                )
                assert result.exit_code == 0, result.output
                assert json.loads(result.output) == expected
            posted.assert_not_called()
        assert (
            api.get(f"/runs/{source_run['run_id']}", params={"user_id": "cli-user"}).json()
            == source_run
        )
        assert api.get(f"/runs/{fork['run_id']}", params={"user_id": "cli-user"}).json() == fork
    assert idle.calls == []
    assert target.read_text() == before
    assert sentinel.read_text() == "pre-existing unrelated work\nlater user edit\n"
