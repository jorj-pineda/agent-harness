"""Real CLI cancellation must settle the tool process group, including descendants."""

import json
import os
import signal
import stat
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

from providers.base import ToolCall
from tests.api.conftest import make_response
from tests.cli_process_support import ROOT, cli, environment, server
from tests.test_cli_process_sigint import wait_for


def live(pid):
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=2
    )
    assert result.returncode in {0, 1} and not result.stderr, "Process-state probe failed"
    # A terminated orphan can remain a zombie until the host reaps it; it cannot run code.
    return bool(result.stdout.strip()) and not result.stdout.strip().startswith("Z")


def test_cli_cancel_stops_owned_check_and_term_ignoring_descendant(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "a.py"
    target.write_text("user = 1\n")
    target.chmod(0o640)
    sentinel = repo / "notes.txt"
    sentinel.write_text("unrelated user work\n")
    parent_pid, child_pid = tmp_path / "parent.json", tmp_path / "child.json"
    (repo / "child.py").write_text(
        "import json,os,signal,time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(child_pid)!r}).write_text(json.dumps({{'pid':os.getpid(),'pgid':os.getpgrp()}}))\n"
        "time.sleep(30)\n"
    )
    (repo / "test_visible.py").write_text(
        "import json,os,subprocess,sys\nfrom pathlib import Path\n"
        "def test_holding_check():\n"
        f"    Path({str(parent_pid)!r}).write_text(json.dumps({{'pid':os.getpid(),'pgid':os.getpgrp()}}))\n"
        "    child = subprocess.Popen([sys.executable, 'child.py'])\n"
        "    child.wait(timeout=40)\n"
    )
    fixture_bytes = {name: (repo / name).read_bytes() for name in ("child.py", "test_visible.py")}
    partial = "user = 1\npartial = 2\n"
    check = ["python", "-m", "pytest", "-q", "test_visible.py"]
    response = make_response(
        tool_calls=[
            ToolCall(
                id="partial", name="write_file", arguments={"path": "a.py", "content": partial}
            ),
            ToolCall(id="check", name="run_command", arguments={"argv": check}),
        ],
        prompt_tokens=100,
        completion_tokens=20,
    ).model_dump(mode="json")
    stores = tmp_path / "stores"
    group = None
    with server(
        tmp_path / "active",
        stores,
        [response],
        overrides={"project_check_argv": check, "max_turn_wall_seconds": 30},
    ) as (url, api):
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
            process.stdin.write("Edit and run the configured check\n/quit\n")
            process.stdin.flush()
            wait_for(lambda: child_pid.exists() and child_pid.read_text())
            parent, child = json.loads(parent_pid.read_text()), json.loads(child_pid.read_text())
            group = parent["pgid"]
            assert parent["pid"] == group and child["pgid"] == group
            assert os.getpgid(child["pid"]) == group
            assert live(parent["pid"]) and live(child["pid"])
            rows = api.get("/runs", params={"user_id": "cli-user"}).json()
            assert len(rows) == 1 and rows[0]["status"] == "running"
            rid, sid = rows[0]["run_id"], rows[0]["request"]["session_id"]
            requested = cli(url, ["cancel", sid])
            assert requested.returncode == 0, requested.stdout + requested.stderr
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stdout + stderr
            record = api.get(f"/runs/{rid}", params={"user_id": "cli-user"}).json()
            assert record["status"] == "cancelled"
            assert record["response"]["completion_status"] == "cancelled"
            assert record["before_checkpoint_id"] and record["after_checkpoint_id"]
            wait_for(lambda: not live(parent["pid"]) and not live(child["pid"]))
            assert api.get("/__preflight").json()["calls"] == 1
        finally:
            if group is not None:
                with suppress(ProcessLookupError):
                    os.killpg(group, signal.SIGKILL)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)
    with server(tmp_path / "restart", stores, []) as (url, api):
        result = cli(url, ["run", rid, "--json"])
        assert result.returncode == 0 and json.loads(result.stdout) == record
        assert api.get("/__preflight").json()["calls"] == 0
    assert target.read_text() == partial and stat.S_IMODE(target.stat().st_mode) == 0o640
    assert sentinel.read_text() == "unrelated user work\n"
    assert {name: (repo / name).read_bytes() for name in fixture_bytes} == fixture_bytes
