"""Process-crash evidence is interrupted work, not a fabricated completed response."""

import json
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path

from providers.base import ToolCall
from tests.api.conftest import make_response
from tests.cli_process_support import ROOT, cli, environment, server
from tests.test_cli_process_sigint import wait_for


def test_owned_api_sigkill_recovers_honest_partial_evidence_without_dispatch(tmp_path):
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
    with server(
        tmp_path / "crash", stores, [edit, {"hang": True}], expected_crash=True, use_cli_server=True
    ) as (
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
            process.stdin.write("Edit a.py then finish\n/quit\n")
            process.stdin.flush()
            wait_for(
                lambda: (
                    target.read_text() == partial and api.get("/__preflight").json()["calls"] == 2
                )
            )
            rows = api.get("/runs", params={"user_id": "cli-user"}).json()
            assert len(rows) == 1 and rows[0]["status"] == "running"
            before = rows[0]
            rid = before["run_id"]
            progress = api.get(f"/runs/{rid}/tool-progress", params={"user_id": "cli-user"}).json()
            assert progress["coverage"] == "recorded"
            assert [event["type"] for event in progress["events"]] == ["tool_start", "tool_end"]
            assert before["before_checkpoint_id"] and before["after_checkpoint_id"] is None
            assert before["response"] is None
            budget = before["budget_evidence"]
            assert [event["kind"] for event in budget["events"]] == [
                "request",
                "response",
                "request",
            ]
            assert budget["events"][1]["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}
            assert budget["events"][-1]["usage"] is None
            # This identity is the helper's registered, readiness-verified owned fixture child.
            owned_pid = api.get("/__preflight").json()["pid"]
            os.kill(owned_pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=5)
            assert process.returncode == 0, stdout + stderr
            assert "Outcome unknown" in stdout
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)
    with server(tmp_path / "restart", stores, []) as (url, api):
        result = cli(url, ["run", rid, "--json"])
        assert result.returncode == 0, result.stderr
        recovered = json.loads(result.stdout)
        assert recovered["status"] == "interrupted"
        assert recovered["response"] is None
        assert recovered["budget_evidence"] == budget
        assert "before a durable final result" in recovered["error"]
        assert recovered["request"] == before["request"]
        assert recovered["execution_profile"] == before["execution_profile"]
        assert recovered["before_checkpoint_id"] == before["before_checkpoint_id"]
        assert recovered["after_checkpoint_id"] is None
        observed = cli(url, ["tool-progress", rid, "--json"])
        assert observed.returncode == 0 and json.loads(observed.stdout) == progress
        human = cli(url, ["run", rid])
        assert human.returncode == 0 and "status=interrupted" in human.stdout
        assert "No saved final response" in human.stdout
        assert api.get("/__preflight").json()["calls"] == 0
        assert api.get("/runs", params={"user_id": "cli-user"}).json() == [recovered]
    with server(tmp_path / "again", stores, []) as (url, api):
        result = cli(url, ["run", rid, "--json"])
        assert result.returncode == 0 and json.loads(result.stdout) == recovered
        assert api.get("/__preflight").json()["calls"] == 0
    assert target.read_text() == partial and stat.S_IMODE(target.stat().st_mode) == 0o640
    assert sentinel.read_text() == "unrelated user work\n"
