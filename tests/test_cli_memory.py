import json
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cli.main import app
from tests.api.conftest import ScriptedProvider
from tests.api.test_session_persistence import make_app

runner = CliRunner()


def test_cli_inspects_and_explicitly_deletes_memory_after_restart(tmp_path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as api:

        def seed():
            api.app.state.components.fact_store.add("dev", "bad convention\x1b[2J")
            api.app.state.components.fact_store.add("dev", "keep")

        api.portal.call(seed)
    with TestClient(make_app(tmp_path, provider)) as api:

        def get(url, **kwargs):
            return api.get(url.removeprefix("http://localhost:8000"), params=kwargs["params"])

        def request(method, url, **kwargs):
            return api.request(
                method, url.removeprefix("http://localhost:8000"), json=kwargs["json"]
            )

        with (
            patch("httpx.get", side_effect=get) as got,
            patch("httpx.request", side_effect=request) as deleted,
            patch("httpx.post") as post,
        ):
            listed = runner.invoke(app, ["memory", "--user-id", "dev", "--limit", "1", "--json"])
            assert listed.exit_code == 0, listed.output
            page = json.loads(listed.output)
            note_id = page["next_before_id"]
            next_page = runner.invoke(
                app, ["memory", "--user-id", "dev", "--before-id", str(note_id)]
            )
            assert next_page.exit_code == 0
            assert "\\u001b" in next_page.output and "\x1b" not in next_page.output
            assert "source_turn=null" in next_page.output
            note = api.get("/memory", params={"user_id": "dev", "before_id": note_id}).json()[
                "notes"
            ][0]
            result = runner.invoke(
                app,
                [
                    "memory-delete",
                    str(note["id"]),
                    "--user-id",
                    "dev",
                    "--expected-sha256",
                    note["fact_sha256"],
                ],
            )
            assert result.exit_code == 0, result.output
            assert "Deleted memory note" in result.output
            assert deleted.call_count == 1 and got.call_count == 2
            post.assert_not_called()
            assert (
                api.get("/memory", params={"user_id": "dev"}).json()["notes"][0]["preview"]
                == "keep"
            )
            assert api.get("/runs", params={"user_id": "dev"}).json() == []
    with TestClient(make_app(tmp_path, provider)) as api:
        assert [
            n["preview"] for n in api.get("/memory", params={"user_id": "dev"}).json()["notes"]
        ] == ["keep"]
    assert provider.calls == []


def test_delete_timeout_has_no_retry_lookup_or_private_exception_text():
    with (
        patch("httpx.request", side_effect=httpx.ReadTimeout("private-marker")) as request,
        patch("httpx.get") as get,
        patch("httpx.post") as post,
    ):
        result = runner.invoke(app, ["memory-delete", "3", "--expected-sha256", "a" * 64])
    assert result.exit_code == 1
    assert "Outcome may be unknown" in result.output and "No automatic retry" in result.output
    assert "private-marker" not in result.output
    assert request.call_count == 1
    assert request.call_args.args == ("DELETE", "http://localhost:8000/memory/3")
    assert request.call_args.kwargs["json"] == {
        "user_id": "cli-user",
        "expected_fact_sha256": "a" * 64,
    }
    get.assert_not_called()
    post.assert_not_called()


def test_bad_hash_and_invalid_page_never_dispatch():
    with patch("httpx.request") as request, patch("httpx.get") as get:
        for argv in [
            ["memory-delete", "1", "--expected-sha256", "bad"],
            ["memory-delete", "0", "--expected-sha256", "a" * 64],
            ["memory", "--limit", "51"],
            ["memory", "--before-id", "0"],
        ]:
            assert runner.invoke(app, argv).exit_code != 0
        request.assert_not_called()
        get.assert_not_called()


def test_delete_mismatched_response_is_not_reported_as_success():
    response = httpx.Response(
        200,
        json={"status": "deleted", "fact_id": 99},
        request=httpx.Request("DELETE", "http://fixture/memory/1"),
    )
    with patch("httpx.request", return_value=response) as request:
        result = runner.invoke(app, ["memory-delete", "1", "--expected-sha256", "a" * 64])
    assert result.exit_code == 1
    assert "Outcome may be unknown" in result.output
    assert "Deleted memory note" not in result.output
    assert request.call_count == 1


def test_memory_empty_and_lookup_timeout_are_read_only():
    response = httpx.Response(
        200,
        json={"notes": [], "next_before_id": None, "scope_note": "Future use only"},
        request=httpx.Request("GET", "http://fixture/memory"),
    )
    with patch("httpx.get", return_value=response), patch("httpx.request") as request:
        assert "No remembered notes" in runner.invoke(app, ["memory"]).output
        request.assert_not_called()
    with patch("httpx.get", side_effect=httpx.ReadTimeout("private-marker")) as get:
        result = runner.invoke(app, ["memory"])
    assert result.exit_code == 1 and "private-marker" not in result.output
    assert get.call_count == 1


def test_cli_correction_uses_one_put_and_real_owner_guards(tmp_path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as api:

        def seed():
            api.app.state.components.fact_store.add("dev", "original")

        api.portal.call(seed)
        note = api.get("/memory", params={"user_id": "dev"}).json()["notes"][0]

        def request(method, url, **kwargs):
            return api.request(
                method, url.removeprefix("http://localhost:8000"), json=kwargs["json"]
            )

        with (
            patch("httpx.request", side_effect=request) as put,
            patch("httpx.get") as get,
            patch("httpx.post") as post,
        ):
            result = runner.invoke(
                app,
                [
                    "memory-correct",
                    str(note["id"]),
                    "new note",
                    "--user-id",
                    "dev",
                    "--expected-sha256",
                    note["fact_sha256"],
                ],
            )
            assert result.exit_code == 0, result.output
            assert "Corrected memory note" in result.output
            assert put.call_count == 1 and put.call_args.args[0] == "PUT"
            get.assert_not_called()
            post.assert_not_called()
        assert (
            api.get("/memory", params={"user_id": "dev"}).json()["notes"][0]["preview"]
            == "new note"
        )
    assert provider.calls == []


def test_cli_correction_timeout_never_retries_and_bad_hash_never_dispatches():
    with (
        patch("httpx.request", side_effect=httpx.ReadTimeout("private-marker")) as put,
        patch("httpx.get") as get,
    ):
        result = runner.invoke(app, ["memory-correct", "1", "new", "--expected-sha256", "a" * 64])
        assert result.exit_code == 1
        assert "Outcome may be unknown" in result.output and "No automatic retry" in result.output
        assert "private-marker" not in result.output and put.call_count == 1
        get.assert_not_called()
        invalid = runner.invoke(app, ["memory-correct", "1", "new", "--expected-sha256", "bad"])
        assert invalid.exit_code != 0 and put.call_count == 1
