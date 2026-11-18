"""Actual SIGINT to a CLI HTTP wait; scripted partial work must survive settlement."""

import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

from providers.base import ToolCall
from tests.api.conftest import make_response
from tests.cli_process_support import ROOT, cli, environment, server


def wait_for(check):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.025)
    raise AssertionError("Owned scripted cancellation did not settle")


def test_real_sigint_preserves_cancelled_partial_edit_and_restart_review(tmp_path):
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
    with server(tmp_path / "first", stores, [edit, {"hang": True}]) as (url, api):
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
            process.stdin.write("Edit a.py and then check it\n")
            process.stdin.flush()
            wait_for(
                lambda: (
                    target.read_text() == partial and api.get("/__preflight").json()["calls"] == 2
                )
            )
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stdout + stderr
            assert "Inspect saved session review after cleanup" in stdout
            assert "Cancellation requested" in stdout
            sid = re.search(r"session=([0-9a-f]{32})", stdout).group(1)
            key = re.search(r"request_id=([0-9a-f]{32})", stdout).group(1)

            def finished():
                rows = api.get("/runs", params={"user_id": "cli-user", "session_id": sid}).json()
                assert len(rows) == 1
                return rows[0] if rows[0]["status"] != "running" else None

            record = wait_for(finished)
            assert record["status"] == "cancelled"
            assert record["request"]["request_id"] == key
            assert record["response"]["completion_status"] == "cancelled"
            assert record["response"]["token_usage"] == {
                "prompt_tokens": None,
                "completion_tokens": None,
            }
            assert record["before_checkpoint_id"] and record["after_checkpoint_id"]
            assert api.get("/__preflight").json()["calls"] == 2
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=5)
                raise AssertionError("Owned CLI required forced cleanup") from None
    with server(tmp_path / "second", stores, []) as (url, api):
        result = cli(url, ["run", record["run_id"], "--json"])
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == record
        assert api.get("/__preflight").json()["calls"] == 0
    assert target.read_text() == partial and stat.S_IMODE(target.stat().st_mode) == 0o640
    assert sentinel.read_text() == "unrelated user work\n"
