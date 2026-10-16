import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cli.main import app
from harness.admission import turn_admission
from harness.check_suggestions import CheckSuggestions
from tests.api.conftest import ScriptedProvider
from tests.api.test_session_persistence import make_app

runner = CliRunner()


def test_cli_suggestions_after_restart_are_owner_scoped_and_read_only(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = repo / "package.json"
    source = '{"scripts":{"test:run":"private-marker do-not-execute"}}'
    manifest.write_text(source)
    (repo / "pytest.ini").write_text("[pytest]\n")
    provider = ScriptedProvider()
    configured = ["pytest", "-q"]
    with TestClient(make_app(tmp_path, provider, project_check_argv=configured)) as api:
        sid = api.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        saved = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
    with TestClient(make_app(tmp_path, provider, project_check_argv=configured)) as api:

        def get(url, **kwargs):
            assert kwargs["timeout"] == 30
            return api.get(url.removeprefix("http://localhost:8000"), params=kwargs["params"])

        with patch("httpx.get", side_effect=get) as gets, patch("httpx.post") as post:
            args = ["check-suggestions", sid, "--user-id", "dev"]
            human = runner.invoke(app, args)
            assert human.exit_code == 0, human.output
            assert "check_suggestions=inspected" in human.output
            assert '["python", "-m", "pytest"]' in human.output
            assert '["npm", "run", "test:run"]' in human.output
            assert "pytest.ini" in human.output and "package.json" in human.output
            assert "No command ran" in human.output and "PROJECT_CHECK_ARGV" in human.output
            assert "private-marker" not in human.output
            raw = runner.invoke(app, [*args, "--json"])
            assert json.loads(raw.output)["session_id"] == sid
            with turn_admission.claim("held", str(repo.resolve())):
                busy = runner.invoke(app, args)
                assert "check_suggestions=busy" in busy.output
            hidden = runner.invoke(app, ["check-suggestions", sid, "--user-id", "other"])
            assert hidden.exit_code == 1 and "package.json" not in hidden.output
            assert gets.call_count == 4
            assert all(
                call.args[0].endswith(f"/sessions/{sid}/check-suggestions")
                for call in gets.call_args_list
            )
            post.assert_not_called()
        assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
        assert api.app.state.settings.project_check_argv == configured
        assert api.get("/runs", params={"user_id": "dev"}).json() == []
        assert manifest.read_text() == source
    assert provider.calls == []


@pytest.mark.parametrize("status", ["inspected", "partial", "unavailable", "busy", "no_workspace"])
def test_cli_status_issues_and_terminal_escaping(status):
    record = CheckSuggestions(status=status, session_id="sid").model_dump()
    record.update(
        reason="reason\x1b[2J",
        scope_note="scope\x1b[2J",
        issues=[{"path": "package.json", "code": "invalid_manifest"}],
        suggestions=[{"argv": ["example\x1b[2J"], "evidence": ["package.json"]}],
    )
    response = httpx.Response(200, json=record, request=httpx.Request("GET", "http://fixture"))
    with patch("httpx.get", return_value=response) as get:
        result = runner.invoke(app, ["check-suggestions", "sid"])
        assert result.exit_code == 0, result.output
        assert f"check_suggestions={status}" in result.output
        assert "invalid_manifest" in result.output
        assert "\\u001b" in result.output and "\x1b" not in result.output
        assert get.call_count == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_id", "other"),
        ("read_only", False),
        ("commands_executed", True),
        ("configuration_changed", True),
        ("status", "verified"),
        ("issues", [None]),
        ("suggestions", [{"argv": "private-marker", "evidence": []}]),
        ("scope_note", None),
    ],
)
def test_cli_rejects_inconsistent_response_before_output(field, value):
    record = CheckSuggestions(status="inspected", session_id="sid").model_dump()
    record[field] = value
    record["reason"] = "private-marker"
    response = httpx.Response(200, json=record, request=httpx.Request("GET", "http://fixture"))
    with patch("httpx.get", return_value=response) as get:
        result = runner.invoke(app, ["check-suggestions", "sid", "--json"])
        assert result.exit_code == 1 and "private-marker" not in result.output
        assert get.call_count == 1


def test_cli_empty_and_private_transport_errors_never_retry():
    record = CheckSuggestions(status="inspected", session_id="sid").model_dump()
    response = httpx.Response(200, json=record, request=httpx.Request("GET", "http://fixture"))
    with patch("httpx.get", return_value=response):
        result = runner.invoke(app, ["check-suggestions", "sid"])
        assert "supported root-manifest scope" in result.output
    for failure in (httpx.ReadTimeout("private-marker"), ValueError("private-marker")):
        with patch("httpx.get", side_effect=failure) as get:
            result = runner.invoke(app, ["check-suggestions", "sid"])
            assert result.exit_code == 1 and "private-marker" not in result.output
            assert "No automatic retry" in result.output and get.call_count == 1
