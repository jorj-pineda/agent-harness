import json
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cli.main import app
from tests.api.conftest import ScriptedProvider
from tests.api.test_session_persistence import make_app

runner = CliRunner()


def test_checkpoint_capture_timeout_never_retries_or_submits_work():
    with (
        patch("httpx.post", side_effect=httpx.ReadTimeout("private-marker")) as post,
        patch("httpx.get") as get,
    ):
        result = runner.invoke(app, ["checkpoint-create", "s1"])
    assert result.exit_code == 1
    assert "Outcome may be unknown" in result.output
    assert "private-marker" not in result.output
    assert "No automatic retry" in result.output
    assert post.call_count == 1
    assert post.call_args.args[0].endswith("/sessions/s1/checkpoints")
    assert post.call_args.kwargs["json"] == {"user_id": "cli-user"}
    get.assert_not_called()


def test_checkpoint_cli_real_capture_then_restart_reads_only_metadata(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("pre-existing user edit\n")
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as api:
        sid = api.post("/sessions", json={"user_id": "dev", "workspace_root": str(root)}).json()[
            "session_id"
        ]

        def post(url, **kwargs):
            return api.post(url.removeprefix("http://localhost:8000"), json=kwargs["json"])

        with patch("httpx.post", side_effect=post) as called:
            created = runner.invoke(app, ["checkpoint-create", sid, "--user-id", "dev", "--json"])
        assert created.exit_code == 0, created.output
        record = json.loads(created.output)
        assert called.call_count == 1
        assert record["file_count"] == 1
        assert "pre-existing user edit" not in created.output
    (root / "a.py").write_text("later user edit\n")
    with TestClient(make_app(tmp_path, provider)) as api:

        def get(url, **kwargs):
            return api.get(url.removeprefix("http://localhost:8000"), params=kwargs["params"])

        with patch("httpx.get", side_effect=get) as called, patch("httpx.post") as post:
            listed = runner.invoke(
                app, ["checkpoints", sid, "--user-id", "dev", "--limit", "1", "--offset", "0"]
            )
            detail = runner.invoke(
                app, ["checkpoint", sid, record["checkpoint_id"], "--user-id", "dev", "--json"]
            )
            assert listed.exit_code == detail.exit_code == 0
            assert "restore_supported=False" in listed.output
            assert json.loads(detail.output) == record
            assert called.call_count == 2
            post.assert_not_called()
        assert (root / "a.py").read_text() == "later user edit\n"
        assert api.get("/runs", params={"user_id": "dev"}).json() == []
    assert provider.calls == []


def test_checkpoint_list_invalid_page_does_not_dispatch_http():
    with patch("httpx.get") as get:
        result = runner.invoke(app, ["checkpoints", "s1", "--limit", "0"])
    assert result.exit_code != 0
    get.assert_not_called()
