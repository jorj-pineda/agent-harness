import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cli.main import app
from tests.api.conftest import make_response
from tests.api.test_continuation import SETTINGS, async_wait, known_provider, source
from tests.api.test_session_persistence import make_app

runner = CliRunner()


def args(sid="s1", tid="t1"):
    return [
        "continue-session",
        sid,
        "Explain a.py",
        "--expected-turn-id",
        tid,
        "--request-id",
        "cli-once",
        "--user-id",
        "dev",
        "--acknowledge-uncaptured-state",
        "--json",
    ]


def test_cli_explicit_continuation_then_restart_replay_is_one_post(tmp_path):
    _root, sid, archived = source(tmp_path)
    tid = archived["session"]["turns"][-1]["turn_id"]
    provider = known_provider()
    provider.script(make_response(content="Explained", prompt_tokens=10, completion_tokens=2))
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as api:

        def post(url, **kwargs):
            return api.post(url.removeprefix("http://localhost:8000"), json=kwargs["json"])

        with patch("httpx.post", side_effect=post) as posted, patch("httpx.get") as looked:
            result = runner.invoke(app, args(sid, tid))
        assert result.exit_code == 0, result.output
        record = json.loads(result.output)
        posted.assert_called_once()
        looked.assert_not_called()
        api.portal.call(async_wait, api.app.state.components, record["run_id"])
    restarted = known_provider()
    with TestClient(make_app(tmp_path, restarted, **SETTINGS)) as api:
        with patch("httpx.post", side_effect=post) as posted, patch("httpx.get") as looked:
            result = runner.invoke(app, args(sid, tid))
        assert result.exit_code == 0, result.output
        replayed = json.loads(result.output)
        assert replayed["run_id"] == record["run_id"]
        assert replayed["response"]["completion_status"] == "completed"
        posted.assert_called_once()
        looked.assert_not_called()
    assert len(provider.calls) == 1
    assert restarted.calls == []


def test_cli_requires_scope_ack_before_network():
    with patch("httpx.post") as posted:
        result = runner.invoke(app, [x for x in args() if x != "--acknowledge-uncaptured-state"])
    assert result.exit_code == 2
    assert "excluded" in result.output.lower() or "exclude" in result.output.lower()
    posted.assert_not_called()


def test_cli_continuation_timeout_never_retries_or_leaks_payload():
    with (
        patch("httpx.post", side_effect=httpx.ReadTimeout("private-marker")) as posted,
        patch("httpx.get") as looked,
    ):
        result = runner.invoke(app, args())
    assert result.exit_code == 1
    assert "cli-once" in result.output
    assert "private-marker" not in result.output
    assert "No automatic retry" in result.output
    posted.assert_called_once()
    looked.assert_not_called()


@pytest.mark.parametrize("missing", ["--request-id", "--expected-turn-id"])
def test_cli_requires_explicit_identity(missing):
    invocation = args()
    index = invocation.index(missing)
    del invocation[index : index + 2]
    with patch("httpx.post") as posted:
        assert runner.invoke(app, invocation).exit_code != 0
    posted.assert_not_called()


def test_cli_rejects_wrong_continuation_identity():
    response = httpx.Response(
        202, json={"run_id": "r", "request": {}}, request=httpx.Request("POST", "http://test")
    )
    with patch("httpx.post", return_value=response) as posted, patch("httpx.get") as looked:
        result = runner.invoke(app, args())
    assert result.exit_code == 1
    assert "Submitted fork" not in result.output
    posted.assert_called_once()
    looked.assert_not_called()
