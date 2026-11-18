"""Abrupt CLI loss leaves bounded app-owned work, not an implicit cancellation."""

import json
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from providers.base import ToolCall
from tests.api.conftest import make_response
from tests.cli_process_support import ROOT, cli, environment, server
from tests.test_cli_process_sigint import wait_for


@pytest.mark.parametrize("settlement", ["cancel", "wall_budget"])
def test_cli_disconnect_leaves_owned_turn_until_explicit_cancel_or_budget(tmp_path, settlement):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "a.py"
    target.write_text("user = 1\n")
    target.chmod(0o640)
    sentinel = repo / "notes.txt"
    sentinel.write_text("unrelated user work\n")
    partial = "user = 1\npartial = 2\n"
    edit = make_response(
        tool_calls=[
            ToolCall(
                id="partial", name="write_file", arguments={"path": "a.py", "content": partial}
            )
        ],
        prompt_tokens=100,
        completion_tokens=20,
    ).model_dump(mode="json")
    stores = tmp_path / "stores"
    limits = {"max_turn_wall_seconds": 5 if settlement == "wall_budget" else 30}
    with server(tmp_path / "active", stores, [edit, {"hang": True}], overrides=limits) as (
        url,
        api,
    ):
        process = subprocess.Popen(
            [
                str(Path(sys.executable).parent / "agent-harness"),
                "chat",
                "--base-url",
                url,
                "--workspace",
                str(repo),
                "--checkpoints",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=ROOT,
            env=environment(),
            start_new_session=True,
        )
        try:
            process.stdin.write("Edit a.py then finish\n")
            process.stdin.flush()
            wait_for(
                lambda: (
                    target.read_text() == partial and api.get("/__preflight").json()["calls"] == 2
                )
            )
            process.terminate()  # SIGTERM ends this owned client, without its Ctrl-C cancel handler.
            process.communicate(timeout=5)
            assert process.returncode == -signal.SIGTERM
            rows = api.get("/runs", params={"user_id": "cli-user"}).json()
            assert len(rows) == 1 and rows[0]["status"] == "running"
            rid, sid = rows[0]["run_id"], rows[0]["request"]["session_id"]
            if settlement == "cancel":
                requested = cli(url, ["cancel", sid])
                assert requested.returncode == 0, requested.stdout + requested.stderr

            def finished():
                record = api.get(f"/runs/{rid}", params={"user_id": "cli-user"}).json()
                return record if record["status"] != "running" else None

            record = wait_for(finished)
            assert record["response"]["completion_status"] == (
                "cancelled" if settlement == "cancel" else "budget_exhausted"
            )
            assert record["response"]["token_usage"] == {
                "prompt_tokens": None,
                "completion_tokens": None,
            }
            assert record["before_checkpoint_id"] and record["after_checkpoint_id"]
            assert api.get("/__preflight").json()["calls"] == 2
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)
    with server(tmp_path / "restart", stores, [], overrides=limits) as (url, api):
        result = cli(url, ["run", rid, "--json"])
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == record
        assert api.get("/__preflight").json()["calls"] == 0
        assert len(api.get("/runs", params={"user_id": "cli-user"}).json()) == 1
    assert target.read_text() == partial and stat.S_IMODE(target.stat().st_mode) == 0o640
    assert sentinel.read_text() == "unrelated user work\n"
