"""Executable cancelled-edit recovery composition; scripted responses, real tools/processes."""

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


def reply(name, arguments):
    return make_response(
        tool_calls=[ToolCall(id=name, name=name, arguments=arguments)],
        prompt_tokens=100,
        completion_tokens=20,
    ).model_dump(mode="json")


def checked_cli(url, arguments, *, raw=False):
    result = cli(url, arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout) if raw else result.stdout


@pytest.mark.parametrize("stale_review", [False, True])
def test_cancelled_edit_restart_checked_fork_and_reverse_order_restoration(tmp_path, stale_review):
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
    check = root / "test_visible.py"
    check.write_text("from a import value\n\ndef test_value():\n    assert value == 1\n")
    check_bytes = check.read_bytes()
    git_env = environment() | {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
    for arguments in (["init", "--quiet"], ["add", "a.py", "notes.txt", "test_visible.py"]):
        subprocess.run(
            ["git", "-C", str(root), *arguments], env=git_env, check=True, capture_output=True
        )
    index = (root / ".git/index").read_bytes()
    stores = tmp_path / "stores"
    settings = {
        "max_context_tokens": 16384,
        "ollama_num_ctx": 16384,
        "require_verification_before_finish": True,
        "project_check_argv": ["python", "-m", "pytest", "-q", "test_visible.py"],
    }
    edit = reply("write_file", {"path": "a.py", "content": partial})
    with server(tmp_path / "source", stores, [edit, {"hang": True}], overrides=settings) as (
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
                str(root),
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
            process.stdin.write("Set value to 1, preserve the user line and check it\n/quit\n")
            process.stdin.flush()
            wait_for(
                lambda: (
                    target.read_text() == partial and api.get("/__preflight").json()["calls"] == 2
                )
            )
            source = api.get("/runs", params={"user_id": "cli-user"}).json()[0]
            sid = source["request"]["session_id"]
            assert "Cancellation requested" in checked_cli(url, ["cancel", sid])
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stdout + stderr

            def settled():
                record = api.get(f"/runs/{source['run_id']}", params={"user_id": "cli-user"}).json()
                return record if record["status"] != "running" else None

            source = wait_for(settled)
            assert source["status"] == "cancelled"
            assert source["response"]["completion_status"] == "cancelled"
            assert source["response"]["verification_status"] == "not_run"
            assert source["response"]["token_usage"] == {
                "prompt_tokens": None,
                "completion_tokens": None,
            }
            assert source["before_checkpoint_id"] and source["after_checkpoint_id"]
            archive = api.get(f"/sessions/{sid}", params={"user_id": "cli-user"}).json()
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
    responses = [
        reply("write_file", {"path": "a.py", "content": final}),
        reply("run_command", {"argv": settings["project_check_argv"]}),
        make_response(
            content="Configured check passed; scripted workflow only.",
            prompt_tokens=100,
            completion_tokens=20,
        ).model_dump(mode="json"),
    ]
    with server(
        tmp_path / "continue", stores, [] if stale_review else responses, overrides=settings
    ) as (url, api):
        assert "workspace=unchanged" in checked_cli(url, ["workspace-review", sid])
        assert "transcript=valid" in checked_cli(url, ["transcript-review", sid])
        assert "configuration=matching" in checked_cli(
            url, ["configuration-review", sid, "--message", "Continue the same task"]
        )
        assert api.get("/__preflight").json()["calls"] == 0
        assert target.read_text() == partial
        if stale_review:
            target.write_text(partial + "# user edit after successful review\n")
        reviewed_bytes = target.read_bytes()
        continuation = [
            "continue-session",
            sid,
            "Continue the same task",
            "--expected-turn-id",
            archive["session"]["turns"][-1]["turn_id"],
            "--request-id",
            "checked-fork",
            "--acknowledge-uncaptured-state",
            "--json",
        ]
        submitted = checked_cli(url, continuation, raw=True)
        rid = submitted["run_id"]

        def finished():
            record = checked_cli(url, ["run", rid, "--json"], raw=True)
            return record if record["status"] != "running" else None

        fork = wait_for(finished)
        assert fork["request"]["session_id"] != sid
        if stale_review:
            assert fork["response"]["completion_status"] == "blocked"
            assert "content/modes or root changed" in fork["response"]["completion_reason"]
            assert fork["before_checkpoint_id"] is None
            assert fork["execution_profile"] is None
            assert target.read_bytes() == reviewed_bytes
            assert api.get("/__preflight").json()["calls"] == 0
        else:
            assert fork["response"]["completion_status"] == "completed", fork
            assert fork["response"]["verification_status"] == "passed"
            assert fork["response"]["check_attempts"][0]["exit_code"] == 0
            assert fork["execution_profile"] == source["execution_profile"]
            assert fork["before_checkpoint_id"] != source["before_checkpoint_id"]
            assert target.read_text() == final
            assert api.get("/__preflight").json()["calls"] == 3
        assert checked_cli(url, ["run", source["run_id"], "--json"], raw=True) == source
        assert api.get(f"/sessions/{sid}", params={"user_id": "cli-user"}).json() == archive
    sentinel.write_text("pre-existing unrelated work\nlater user edit\n")
    if stale_review:
        reversal = ["revert-file", source["run_id"], "a.py", "--request-id", "stale-undo", "--json"]
        with server(tmp_path / "reject", stores, [], overrides=settings) as (url, api):
            result = cli(url, reversal)
            assert result.returncode == 1, result.stdout + result.stderr
            journal = json.loads(result.stdout)
            assert journal["status"] == "rejected"
            assert "Current content or mode differs" in journal["reason"]
            assert target.read_bytes() == reviewed_bytes
            assert api.get("/__preflight").json()["calls"] == 0
        target.write_text("# later user bytes after rejected recovery\n")
        target.chmod(0o600)
        later_bytes = target.read_bytes()
        with server(tmp_path / "replay", stores, [], overrides=settings) as (url, api):
            assert checked_cli(url, continuation, raw=True) == fork
            assert (
                checked_cli(
                    url, ["revert-status", "--request-id", "stale-undo", "--json"], raw=True
                )
                == journal
            )
            replay = cli(url, reversal)
            assert replay.returncode == 1 and json.loads(replay.stdout) == journal
            assert checked_cli(url, ["run", source["run_id"], "--json"], raw=True) == source
            assert api.get(f"/sessions/{sid}", params={"user_id": "cli-user"}).json() == archive
            assert len(api.get("/runs", params={"user_id": "cli-user"}).json()) == 2
            assert api.get("/__preflight").json()["calls"] == 0
        assert target.read_bytes() == later_bytes
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert sentinel.read_text() == "pre-existing unrelated work\nlater user edit\n"
        assert check.read_bytes() == check_bytes
        assert (root / ".git/index").read_bytes() == index
        return
    journals = []
    with server(tmp_path / "reverse", stores, [], overrides=settings) as (url, api):
        for number, record in enumerate((fork, source)):
            journal = checked_cli(
                url,
                [
                    "revert-file",
                    record["run_id"],
                    "a.py",
                    "--request-id",
                    f"undo-{number}",
                    "--json",
                ],
                raw=True,
            )
            assert journal["status"] == "applied"
            journals.append(journal)
            assert target.read_text() == (partial if number == 0 else before)
        assert api.get("/__preflight").json()["calls"] == 0
    with server(tmp_path / "inspect", stores, [], overrides=settings) as (url, api):
        for number, journal in enumerate(journals):
            assert (
                checked_cli(
                    url, ["revert-status", "--request-id", f"undo-{number}", "--json"], raw=True
                )
                == journal
            )
        for record in (source, fork):
            assert checked_cli(url, ["run", record["run_id"], "--json"], raw=True) == record
        assert api.get(f"/sessions/{sid}", params={"user_id": "cli-user"}).json() == archive
        assert api.get("/__preflight").json()["calls"] == 0
    assert target.read_text() == before
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert sentinel.read_text() == "pre-existing unrelated work\nlater user edit\n"
    assert check.read_bytes() == check_bytes
    assert (root / ".git/index").read_bytes() == index
