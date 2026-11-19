"""Separate CLI processes reverse only guarded artifacts and preserve user Git state."""

import json
import re
import stat
import subprocess

import pytest

from tests.cli_process_support import cli, environment, server
from tests.test_cli_process_continuation import replies
from tests.test_cli_process_sigint import wait_for


@pytest.mark.parametrize("drift", [None, "content", "mode"])
def test_process_reversal_after_restart_preserves_index_and_later_edits(tmp_path, drift):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "a.py"
    original = "user = 1\n"
    target.write_text(original)
    target.chmod(0o640)
    sentinel = repo / "notes.txt"
    sentinel.write_text("unrelated user work\n")
    git_env = environment() | {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
    subprocess.run(
        ["git", "init", "--quiet", str(repo)], env=git_env, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "add", "a.py", "notes.txt"],
        env=git_env,
        check=True,
        capture_output=True,
    )
    index = (repo / ".git/index").read_bytes()
    source_content = original + "agent = 2\n"
    fork_content = source_content + "continued = 3\n"
    stores = tmp_path / "stores"
    settings = {"max_context_tokens": 8192, "ollama_num_ctx": 8192}
    with server(
        tmp_path / "source", stores, replies(source_content, call_id="source"), overrides=settings
    ) as (url, api):
        result = cli(
            url, ["chat", "--workspace", str(repo), "--checkpoints"], input="Edit a.py\n/quit\n"
        )
        assert result.returncode == 0, result.stderr + result.stdout
        sid = re.search(r"session=([0-9a-f]{32})", result.stdout).group(1)
        source = api.get("/runs", params={"user_id": "cli-user"}).json()[0]
        archive = api.get(f"/sessions/{sid}", params={"user_id": "cli-user"}).json()
        tid = archive["session"]["turns"][-1]["turn_id"]
    with server(
        tmp_path / "fork", stores, replies(fork_content, call_id="fork"), overrides=settings
    ) as (url, api):
        result = cli(
            url,
            [
                "continue-session",
                sid,
                "Continue a.py",
                "--expected-turn-id",
                tid,
                "--request-id",
                "fork-once",
                "--acknowledge-uncaptured-state",
                "--json",
            ],
        )
        assert result.returncode == 0, result.stderr + result.stdout
        rid = json.loads(result.stdout)["run_id"]

        def finished():
            record = api.get(f"/runs/{rid}", params={"user_id": "cli-user"}).json()
            return record if record["status"] != "running" else None

        fork = wait_for(finished)
        assert fork["response"]["completion_status"] == "completed"
        assert target.read_text() == fork_content
    if drift == "content":
        target.write_text(fork_content + "later_user_edit = 4\n")
    elif drift == "mode":
        target.chmod(0o600)
        assert target.read_text() == fork_content
    expected_mode = 0o600 if drift == "mode" else 0o640
    expected = target.read_bytes()
    with server(tmp_path / "reverse", stores, [], overrides=settings) as (url, api):
        args = ["revert-file", rid, "a.py", "--request-id", "reverse-fork", "--json"]
        result = cli(url, args)
        journal = json.loads(result.stdout)
        if drift:
            assert result.returncode == 1 and journal["status"] == "rejected"
            assert "Current content or mode differs" in journal["reason"]
            assert target.read_bytes() == expected
            assert stat.S_IMODE(target.stat().st_mode) == expected_mode
        else:
            assert result.returncode == 0 and journal["status"] == "applied"
            assert target.read_text() == source_content
            source_args = [
                "revert-file",
                source["run_id"],
                "a.py",
                "--request-id",
                "reverse-source",
                "--json",
            ]
            result = cli(url, source_args)
            assert result.returncode == 0 and json.loads(result.stdout)["status"] == "applied"
            assert target.read_text() == original
        assert api.get(f"/runs/{source['run_id']}", params={"user_id": "cli-user"}).json() == source
        assert api.get(f"/runs/{rid}", params={"user_id": "cli-user"}).json() == fork
        assert api.get(f"/sessions/{sid}", params={"user_id": "cli-user"}).json() == archive
        assert api.get("/__preflight").json()["calls"] == 0
    target.write_text("later user bytes after journal\n")
    later = target.read_bytes()
    with server(tmp_path / "journal", stores, [], overrides=settings) as (url, api):
        result = cli(url, ["revert-status", "--request-id", "reverse-fork", "--json"])
        assert result.returncode == 0 and json.loads(result.stdout) == journal
        replay = cli(url, args)
        assert replay.returncode == (1 if drift else 0)
        assert json.loads(replay.stdout) == journal
        assert target.read_bytes() == later
        assert api.get("/__preflight").json()["calls"] == 0
    assert sentinel.read_text() == "unrelated user work\n"
    assert stat.S_IMODE(target.stat().st_mode) == expected_mode
    assert (repo / ".git/index").read_bytes() == index
