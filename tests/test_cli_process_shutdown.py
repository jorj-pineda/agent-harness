"""Serve must reach app-owned cancellation even while an HTTP turn is waiting."""

import json
import subprocess
import sys
from pathlib import Path

from providers.base import ToolCall
from tests.api.conftest import make_response
from tests.cli_process_support import ROOT, cli, environment, server
from tests.test_cli_process_sigint import wait_for


def test_cli_serve_shutdown_reaches_owned_cancellation_and_preserves_partial_work(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "a.py"
    target.write_text("user = 1\n")
    sentinel = repo / "notes.txt"
    sentinel.write_text("preserve user work\n")
    partial = "user = 1\npartial = 2\n"
    edit = make_response(
        tool_calls=[
            ToolCall(
                id="partial", name="write_file", arguments={"path": "a.py", "content": partial}
            )
        ]
    ).model_dump(mode="json")
    stores = tmp_path / "stores"
    process = None
    try:
        with server(tmp_path / "active", stores, [edit, {"hang": True}], use_cli_server=True) as (
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
            )
            process.stdin.write("Edit then finish\n/quit\n")
            process.stdin.flush()
            wait_for(
                lambda: (
                    target.read_text() == partial and api.get("/__preflight").json()["calls"] == 2
                )
            )
            rows = api.get("/runs", params={"user_id": "cli-user"}).json()
            assert len(rows) == 1 and rows[0]["status"] == "running"
            rid = rows[0]["run_id"]
            # Leaving the context signals the API, not the CLI or application cancel route.
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stdout + stderr
        assert "Outcome unknown" in stdout
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)
    with server(tmp_path / "restart", stores, []) as (url, api):
        result = cli(url, ["run", rid, "--json"])
        assert result.returncode == 0, result.stderr
        record = json.loads(result.stdout)
        assert record["status"] == "cancelled"
        assert record["response"]["completion_status"] == "cancelled"
        assert record["before_checkpoint_id"] and record["after_checkpoint_id"]
        assert api.get("/__preflight").json()["calls"] == 0
    assert target.read_text() == partial and sentinel.read_text() == "preserve user work\n"
