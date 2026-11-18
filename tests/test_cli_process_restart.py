"""Real CLI + HTTP + restart; scripted mechanics, not model-quality evidence."""

import json
import re
import stat

from providers.base import ToolCall
from tests.api.conftest import make_response
from tests.cli_process_support import cli, server


def test_cli_executable_saved_diff_survives_api_process_restart(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "a.py"
    target.write_text("user = 1\n")
    target.chmod(0o640)
    sentinel = repo / "notes.txt"
    sentinel.write_text("unrelated user work\n")
    replies = [
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "a.py", "content": "user = 1\nagent = 2\n"},
                )
            ]
        ),
        make_response(content="Scripted completion, not acceptance evidence."),
    ]
    stores = tmp_path / "stores"
    with server(tmp_path / "first", stores, [r.model_dump(mode="json") for r in replies]) as (
        url,
        api,
    ):
        result = cli(
            url, ["chat", "--workspace", str(repo), "--checkpoints"], input="Edit a.py\n/quit\n"
        )
        assert result.returncode == 0, result.stderr + result.stdout
        assert "+agent = 2" in result.stdout
        sid = re.search(r"session=([0-9a-f]{32})", result.stdout).group(1)
        records = api.get("/runs", params={"user_id": "cli-user", "session_id": sid}).json()
        assert len(records) == 1
        record = records[0]
        assert api.get("/__preflight").json()["calls"] == 2
    before = {p.name: (p.read_bytes(), stat.S_IMODE(p.stat().st_mode)) for p in repo.iterdir()}
    with server(tmp_path / "second", stores, []) as (url, api):
        result = cli(url, ["run", record["run_id"], "--json"])
        assert result.returncode == 0, result.stderr + result.stdout
        assert json.loads(result.stdout) == record
        human = cli(url, ["run", record["run_id"]])
        assert human.returncode == 0 and "+agent = 2" in human.stdout
        history = cli(url, ["runs", "--session-id", sid])
        assert history.returncode == 0 and record["run_id"] in history.stdout
        assert api.get("/__preflight").json()["calls"] == 0
        assert api.get("/runs", params={"user_id": "cli-user"}).json() == [record]
    assert {
        p.name: (p.read_bytes(), stat.S_IMODE(p.stat().st_mode)) for p in repo.iterdir()
    } == before
    assert sentinel.read_text() == "unrelated user work\n"
