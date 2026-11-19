"""Guarded continuation and keyed replay through real CLI/API processes, scripted only."""

import json
import re
import stat

import pytest

from providers.base import ToolCall
from tests.api.conftest import make_response
from tests.cli_process_support import cli, server
from tests.test_cli_process_sigint import wait_for


def replies(content, *, call_id):
    return [
        make_response(
            tool_calls=[
                ToolCall(
                    id=call_id, name="write_file", arguments={"path": "a.py", "content": content}
                )
            ],
            prompt_tokens=100,
            completion_tokens=20,
        ).model_dump(mode="json"),
        make_response(
            content="Scripted completion only.", prompt_tokens=100, completion_tokens=20
        ).model_dump(mode="json"),
    ]


@pytest.mark.parametrize("drift", [None, "content", "mode"])
def test_process_continuation_after_restart_replays_key_or_refuses_drift(tmp_path, drift):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "a.py"
    target.write_text("user = 1\n")
    target.chmod(0o640)
    sentinel = repo / "notes.txt"
    sentinel.write_text("unrelated user work\n")
    source_content = "user = 1\nagent = 2\n"
    fork_content = source_content + "continued = 3\n"
    stores = tmp_path / "stores"
    settings = {"max_context_tokens": 8192, "ollama_num_ctx": 8192}
    with server(
        tmp_path / "source", stores, replies(source_content, call_id="source"), overrides=settings
    ) as (
        url,
        api,
    ):
        result = cli(
            url, ["chat", "--workspace", str(repo), "--checkpoints"], input="Edit a.py\n/quit\n"
        )
        assert result.returncode == 0, result.stderr + result.stdout
        sid = re.search(r"session=([0-9a-f]{32})", result.stdout).group(1)
        archived = api.get(f"/sessions/{sid}", params={"user_id": "cli-user"}).json()
        original = api.get("/runs", params={"user_id": "cli-user"}).json()[0]
        tid = archived["session"]["turns"][-1]["turn_id"]
    args = [
        "continue-session",
        sid,
        "Continue a.py",
        "--expected-turn-id",
        tid,
        "--request-id",
        "process-continue-once",
        "--acknowledge-uncaptured-state",
        "--json",
    ]
    if drift == "content":
        target.write_text(source_content + "later_user_edit = 4\n")
    elif drift == "mode":
        target.chmod(0o600)
        assert target.read_text() == source_content
    expected_mode = 0o600 if drift == "mode" else 0o640
    expected = target.read_bytes()
    with server(
        tmp_path / "fork",
        stores,
        [] if drift else replies(fork_content, call_id="fork"),
        overrides=settings,
    ) as (url, api):
        result = cli(url, args)
        assert result.returncode == 0, result.stderr + result.stdout
        admitted = json.loads(result.stdout)
        assert admitted["request"]["session_id"] != sid
        rid = admitted["run_id"]

        def finished():
            record = api.get(f"/runs/{rid}", params={"user_id": "cli-user"}).json()
            return record if record["status"] != "running" else None

        completed = wait_for(finished)
        assert completed["status"] == "finished"
        if drift:
            assert completed["response"]["completion_status"] == "blocked"
            assert "content/modes or root changed" in completed["response"]["completion_reason"]
            assert completed["execution_profile"] is None
            assert completed["before_checkpoint_id"] is None
            assert target.read_bytes() == expected
            assert stat.S_IMODE(target.stat().st_mode) == expected_mode
            assert api.get("/__preflight").json()["calls"] == 0
        else:
            assert completed["response"]["completion_status"] == "completed", completed["response"][
                "completion_reason"
            ]
            assert target.read_text() == fork_content
            assert api.get("/__preflight").json()["calls"] == 2
        assert api.get(f"/sessions/{sid}", params={"user_id": "cli-user"}).json() == archived
        assert (
            api.get(f"/runs/{original['run_id']}", params={"user_id": "cli-user"}).json()
            == original
        )
    if not drift:
        with server(tmp_path / "replay", stores, [], overrides=settings) as (url, api):
            result = cli(url, args)
            assert result.returncode == 0, result.stderr + result.stdout
            assert json.loads(result.stdout) == completed
            assert len(api.get("/runs", params={"user_id": "cli-user"}).json()) == 2
            assert api.get("/__preflight").json()["calls"] == 0
            assert target.read_text() == fork_content
    assert stat.S_IMODE(target.stat().st_mode) == expected_mode
    assert sentinel.read_text() == "unrelated user work\n"
