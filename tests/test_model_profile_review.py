import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from typer.testing import CliRunner

from cli.main import app, show_profile_selection
from harness.admission import turn_admission
from harness.configuration_review import CurrentModelProfileReview
from tests.api.conftest import ScriptedProvider, make_response
from tests.api.test_session_persistence import make_app
from tests.test_model_profiles import PROFILE

runner = CliRunner()


def settings_for(name):
    return (
        {}
        if name is None
        else {
            "model_profiles": {
                name: {**PROFILE, "max_context_tokens": 8192},
                "unused": {**PROFILE, "model": "private-unused-model"},
            },
            "model_profile": name,
            "ollama_host": "http://private-endpoint:11434",
            "openai_api_key": "private-unused-key",
        }
    )


@pytest.mark.parametrize("name", [None, "local"])
def test_owner_profile_review_and_cli_after_restart_without_work(tmp_path, name):
    provider = ScriptedProvider()
    provider.name = "ollama"
    settings = settings_for(name)
    with TestClient(make_app(tmp_path, provider, **settings)) as api:
        sid = api.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        saved = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
    with TestClient(make_app(tmp_path, provider, **settings)) as api:
        url = f"/sessions/{sid}/configuration-review"
        response = api.post(url, json={"user_id": "dev"})
        selection = response.json()["current_model_profile"]
        assert selection["status"] == ("selected" if name else "direct_settings")
        assert "private-" not in response.text
        if name:
            assert selection["identity"]["name"] == name
            assert selection["identity"]["model"] == "gemma4:12b"
            assert len(selection["identity"]["definition_sha256"]) == 64
        else:
            assert selection["identity"] is None
        assert api.post(url, json={"user_id": "other"}).status_code == 403
        with turn_admission.claim(sid, None):
            busy = api.post(url, json={"user_id": "dev"}).json()
            assert busy["status"] == "busy" and busy["current_model_profile"] is None

        def post(url, **kwargs):
            return api.post(url.removeprefix("http://localhost:8000"), json=kwargs["json"])

        with patch("httpx.post", side_effect=post) as posts, patch("httpx.get") as get:
            result = runner.invoke(app, ["configuration-review", sid, "--user-id", "dev"])
            assert result.exit_code == 0, result.output
            expected = "selected" if name else "direct_settings"
            assert f"current_model_profile={expected}" in result.output
            assert "no switching, inference or settings changes" in result.output
            assert "private-" not in result.output
            if name:
                assert 'name="local"' in result.output and 'model="gemma4:12b"' in result.output
                assert "configured_request_output_cap=512" in result.output
            assert posts.call_count == 1 and posts.call_args.args[0].endswith(url)
            get.assert_not_called()
        assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
        assert api.get("/runs", params={"user_id": "dev"}).json() == []
    assert provider.calls == []


def test_current_selection_does_not_relabel_historical_profile(tmp_path):
    provider = ScriptedProvider()
    provider.name = "ollama"
    provider.script(make_response(content="Inspected", prompt_tokens=10, completion_tokens=5))
    with TestClient(make_app(tmp_path, provider, **settings_for("old"))) as api:
        sid = api.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        assert (
            api.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Inspect"}
            ).status_code
            == 200
        )
        saved = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
    with TestClient(make_app(tmp_path, provider, **settings_for("new"))) as api:
        runs = api.get("/runs", params={"user_id": "dev"}).json()
        report = api.post(f"/sessions/{sid}/configuration-review", json={"user_id": "dev"}).json()
        assert report["current_model_profile"]["identity"]["name"] == "new"
        assert report["configuration"]["changed_fields"] == ["model_profile"]
        assert "historical profile" in report["current_model_profile"]["scope_note"]
        assert saved["responses"][0]["execution_profile"]["model_profile"]["name"] == "old"
        assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
        assert api.get("/runs", params={"user_id": "dev"}).json() == runs
    assert len(provider.calls) == 1


@pytest.mark.parametrize(
    "selection",
    [
        None,
        {"status": "selected", "identity": None},
        {"status": "future", "identity": {}},
        {"status": "direct_settings", "identity": {}},
        {"status": "selected", "identity": {"schema_version": 2}},
    ],
)
def test_cli_legacy_inconsistent_profile_selection_is_not_invented(selection, capsys):
    show_profile_selection(selection)
    output = capsys.readouterr().out
    assert "current_model_profile=selected" not in output
    assert "current_model_profile=direct_settings" not in output
    assert "not_checked_or_legacy" in output or "unknown_or_inconsistent" in output


def test_cli_selection_escapes_terminal_controls_and_does_not_print_other_fields(capsys):
    identity = {
        "schema_version": 1,
        "name": "local\x1b[2J",
        "provider": "ollama",
        "model": "fixture\nmodel",
        "definition_sha256": "a" * 64,
        "unused": "private-marker",
    }
    show_profile_selection({"status": "selected", "identity": identity})
    output = capsys.readouterr().out
    assert "\\u001b" in output and "\x1b" not in output
    assert "private-marker" not in output
    assert json.dumps(identity["model"]) in output


def test_selection_schema_requires_status_identity_consistency():
    with pytest.raises(ValidationError):
        CurrentModelProfileReview(status="selected")
