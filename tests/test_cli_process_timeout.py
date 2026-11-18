"""A real read timeout must retain one keyed run and make recovered results review-only."""

import json
import os
import re
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


@pytest.mark.parametrize("recovery", ["/check", "/retry"])
def test_real_timeout_recovers_one_keyed_result_without_another_turn(tmp_path, recovery):
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
    output = tmp_path / "cli.stdout"
    with (
        server(
            tmp_path / "active",
            stores,
            [edit, {"hang": True}],
            overrides={"max_turn_wall_seconds": 5},
        ) as (url, api),
        output.open("w") as stdout,
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
                "--response-timeout",
                "1",
            ],
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=subprocess.PIPE,
            text=True,
            cwd=ROOT,
            env=environment(),
            start_new_session=True,
        )
        try:
            process.stdin.write("Edit a.py then finish\n")
            process.stdin.flush()
            wait_for(lambda: "Outcome unknown. /check reads saved state" in output.read_text())
            rows = api.get("/runs", params={"user_id": "cli-user"}).json()
            assert len(rows) == 1 and rows[0]["status"] == "running"
            rid = rows[0]["run_id"]
            key = re.search(r"request_id=([0-9a-f]{32})", output.read_text()).group(1)
            assert rows[0]["request"]["request_id"] == key
            process.stdin.write("Do not submit this new edit\n")
            process.stdin.flush()
            wait_for(
                lambda: (
                    "Use /check, /retry, or /quit before starting new work" in output.read_text()
                )
            )
            assert api.get("/__preflight").json()["calls"] == 2

            def finished():
                record = api.get(f"/runs/{rid}", params={"user_id": "cli-user"}).json()
                return record if record["status"] != "running" else None

            record = wait_for(finished)
            assert record["response"]["completion_status"] == "budget_exhausted"
            process.stdin.write(recovery + "\n")
            process.stdin.flush()
            _, stderr = process.communicate(timeout=5)
            assert process.returncode == 0, stderr + output.read_text()
            assert "+partial = 2" in output.read_text()
            assert (
                "Review only" if recovery == "/check" else "Recovered result is for review"
            ) in output.read_text()
            assert api.get("/runs", params={"user_id": "cli-user"}).json() == [record]
            assert api.get("/__preflight").json()["calls"] == 2
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)
    with server(tmp_path / "restart", stores, []) as (url, api):
        result = cli(url, ["run", rid, "--json"])
        assert result.returncode == 0 and json.loads(result.stdout) == record
        assert api.get("/__preflight").json()["calls"] == 0
    assert target.read_text() == partial and stat.S_IMODE(target.stat().st_mode) == 0o640
    assert sentinel.read_text() == "unrelated user work\n"
