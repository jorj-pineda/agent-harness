import json
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cli.main import app
from tests.api.conftest import ScriptedProvider
from tests.api.test_file_revert import prepared
from tests.api.test_session_persistence import make_app

runner = CliRunner()


def test_cli_revert_and_status_use_explicit_key_and_never_repeat_on_lookup(tmp_path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as api:
        root, _sid, run = prepared(tmp_path, api, provider)

        def post(url, **kwargs):
            return api.post(url.removeprefix("http://localhost:8000"), json=kwargs["json"])

        def get(url, **kwargs):
            return api.get(url.removeprefix("http://localhost:8000"), params=kwargs["params"])

        with patch("httpx.post", side_effect=post) as posted:
            result = runner.invoke(
                app,
                [
                    "revert-file",
                    run["run_id"],
                    "a.py",
                    "--request-id",
                    "cli-once",
                    "--user-id",
                    "dev",
                    "--json",
                ],
            )
        assert result.exit_code == 0, result.output
        record = json.loads(result.output)
        assert record["status"] == "applied"
        assert posted.call_count == 1
        (root / "a.py").write_text("later target user edit\n")
        with patch("httpx.get", side_effect=get) as looked, patch("httpx.post") as posted:
            result = runner.invoke(
                app, ["revert-status", "--request-id", "cli-once", "--user-id", "dev", "--json"]
            )
        assert result.exit_code == 0
        assert json.loads(result.output) == record
        looked.assert_called_once()
        posted.assert_not_called()
        assert (root / "a.py").read_text() == "later target user edit\n"


def test_cli_revert_timeout_has_no_retry_or_exception_payload_leak():
    with (
        patch("httpx.post", side_effect=httpx.ReadTimeout("private-marker")) as posted,
        patch("httpx.get") as looked,
    ):
        result = runner.invoke(app, ["revert-file", "r1", "a.py", "--request-id", "known-key"])
    assert result.exit_code == 1
    assert "known-key" in result.output
    assert "private-marker" not in result.output
    assert "No automatic retry" in result.output
    posted.assert_called_once()
    looked.assert_not_called()


def test_cli_revert_requires_explicit_request_key_before_dispatch():
    with patch("httpx.post") as posted:
        result = runner.invoke(app, ["revert-file", "r1", "a.py"])
    assert result.exit_code != 0
    posted.assert_not_called()


def test_cli_rejects_wrong_file_response_without_claiming_success_or_retrying():
    response = httpx.Response(
        200,
        json={
            "revert_id": "wrong",
            "status": "applied",
            "request": {
                "user_id": "cli-user",
                "run_id": "r1",
                "path": "other.py",
                "request_id": "known-key",
            },
        },
        request=httpx.Request("POST", "http://test"),
    )
    with patch("httpx.post", return_value=response) as posted, patch("httpx.get") as looked:
        result = runner.invoke(app, ["revert-file", "r1", "a.py", "--request-id", "known-key"])
    assert result.exit_code == 1
    assert "status=applied" not in result.output
    posted.assert_called_once()
    looked.assert_not_called()
