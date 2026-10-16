import hashlib
import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cli.main import app
from tests.api.conftest import ScriptedProvider
from tests.api.test_session_persistence import make_app

runner = CliRunner()


def test_cli_complete_note_after_restart_escapes_controls_without_writes(tmp_path):
    provider = ScriptedProvider()
    text = "é" * 1000 + "\x1b[2J\nsecond line"
    with TestClient(make_app(tmp_path, provider)) as api:
        api.portal.call(
            lambda: api.app.state.components.fact_store.add("dev", text, workspace_root="/a")
        )
        note = api.get("/memory", params={"user_id": "dev"}).json()["notes"][0]
        assert note["preview_truncated"]
    with TestClient(make_app(tmp_path, provider)) as api:

        def get(url, **kwargs):
            return api.get(url.removeprefix("http://localhost:8000"), params=kwargs["params"])

        with patch("httpx.get", side_effect=get) as gets, patch("httpx.request") as mutation:
            args = ["memory-note", str(note["id"]), "--user-id", "dev"]
            human = runner.invoke(app, args)
            assert human.exit_code == 0, human.output
            assert "\\u001b" in human.output and "\x1b" not in human.output
            assert "second line" in human.output and note["fact_sha256"] in human.output
            raw = runner.invoke(app, [*args, "--json"])
            assert json.loads(raw.output)["text"] == text
            assert gets.call_count == 2
            hidden = runner.invoke(app, ["memory-note", str(note["id"]), "--user-id", "other"])
            assert hidden.exit_code == 1 and "second line" not in hidden.output
            mutation.assert_not_called()
        assert api.get("/runs", params={"user_id": "dev"}).json() == []
    assert provider.calls == []


@pytest.mark.parametrize("change", ["id", "hash", "bytes", "oversize", "missing"])
def test_cli_rejects_inconsistent_detail_without_printing_note(change):
    text = "private-marker"
    note = {
        "id": 1,
        "fact_bytes": len(text),
        "fact_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    record = {"note": note, "text": text}
    if change == "id":
        note["id"] = 2
    if change == "hash":
        note["fact_sha256"] = "a" * 64
    if change == "bytes":
        note["fact_bytes"] = 0
    if change == "oversize":
        record["text"] = "x" * 4097
    if change == "missing":
        record.pop("text")
    response = httpx.Response(
        200, json=record, request=httpx.Request("GET", "http://fixture/memory/1")
    )
    with patch("httpx.get", return_value=response) as get:
        result = runner.invoke(app, ["memory-note", "1", "--json"])
        assert result.exit_code == 1 and "private-marker" not in result.output
        assert get.call_count == 1


def test_cli_detail_timeout_and_invalid_id_never_retry():
    with patch("httpx.get", side_effect=httpx.ReadTimeout("private-marker")) as get:
        result = runner.invoke(app, ["memory-note", "1"])
        assert result.exit_code == 1 and "private-marker" not in result.output
        assert "No automatic retry" in result.output and get.call_count == 1
        assert runner.invoke(app, ["memory-note", "0"]).exit_code != 0
        assert get.call_count == 1
