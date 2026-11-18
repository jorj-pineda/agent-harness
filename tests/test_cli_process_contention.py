"""Separate CLI clients must respect one active writer, including overlapping roots."""

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


@pytest.mark.parametrize("root_mode", ["same", "child", "symlink"])
def test_real_cli_workspace_contention_blocks_then_releases_without_dispatch(tmp_path, root_mode):
    repo = tmp_path / "repo"
    repo.mkdir()
    child = repo / "child"
    child.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)
    competing_root = {"same": repo, "child": child, "symlink": alias}[root_mode]
    target = repo / "a.py"
    target.write_text("user = 1\n")
    target.chmod(0o640)
    sentinel = repo / "notes.txt"
    sentinel.write_text("unrelated user work\n")
    partial = "user = 1\npartial = 2\n"
    responses = [
        make_response(
            tool_calls=[
                ToolCall(
                    id="partial", name="write_file", arguments={"path": "a.py", "content": partial}
                )
            ]
        ).model_dump(mode="json"),
        {"hang": True},
        make_response(content="Scripted read-only answer.").model_dump(mode="json"),
    ]
    stores = tmp_path / "stores"
    with server(tmp_path / "active", stores, responses) as (url, api):
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
            source = api.get("/runs", params={"user_id": "cli-user"}).json()[0]
            refused = cli(
                url,
                ["chat", "--workspace", str(competing_root), "--checkpoints"],
                input="Competing task must not dispatch\n/quit\n",
            )
            assert refused.returncode == 0, refused.stdout + refused.stderr
            assert "status=blocked" in refused.stdout and "workspace overlaps" in refused.stdout
            rows = api.get("/runs", params={"user_id": "cli-user"}).json()
            assert len(rows) == 2
            blocked = next(row for row in rows if row["run_id"] != source["run_id"])
            assert blocked["response"]["completion_status"] == "blocked"
            assert blocked["response"]["tool_calls"] == []
            assert (
                blocked["before_checkpoint_id"] is None and blocked["after_checkpoint_id"] is None
            )
            assert (
                api.get(f"/runs/{source['run_id']}", params={"user_id": "cli-user"}).json()[
                    "status"
                ]
                == "running"
            )
            assert api.get("/__preflight").json()["calls"] == 2
            assert target.read_text() == partial
            cancelled = cli(url, ["cancel", source["request"]["session_id"]])
            assert cancelled.returncode == 0, cancelled.stderr
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stdout + stderr
            accepted = cli(
                url, ["chat", "--workspace", str(competing_root)], input="Explain only\n/quit\n"
            )
            assert accepted.returncode == 0 and "status=completed" in accepted.stdout
            assert api.get("/__preflight").json()["calls"] == 3
            records = api.get("/runs", params={"user_id": "cli-user"}).json()
            assert len(records) == 3
            assert next(r for r in records if r["run_id"] == blocked["run_id"]) == blocked
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)
    with server(tmp_path / "restart", stores, []) as (url, api):
        for record in records:
            result = cli(url, ["run", record["run_id"], "--json"])
            assert result.returncode == 0 and json.loads(result.stdout) == record
        assert api.get("/__preflight").json()["calls"] == 0
    assert target.read_text() == partial and stat.S_IMODE(target.stat().st_mode) == 0o640
    assert sentinel.read_text() == "unrelated user work\n"
