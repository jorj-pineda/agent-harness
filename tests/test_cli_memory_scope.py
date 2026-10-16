from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cli.main import app
from tests.api.conftest import ScriptedProvider
from tests.api.test_session_persistence import make_app

runner = CliRunner()


def test_cli_scope_move_restart_and_stale_source_are_one_request(tmp_path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as api:
        api.portal.call(
            lambda: api.app.state.components.fact_store.add(
                "dev", "convention", source_turn_id="source"
            )
        )
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
            args = [
                "memory-scope",
                str(note["id"]),
                "--user-id",
                "dev",
                "--expected-sha256",
                note["fact_sha256"],
                "--from-global",
                "--to-workspace",
                "/project/a",
            ]
            result = runner.invoke(app, args)
            assert result.exit_code == 0, result.output
            assert 'scope="/project/a"' in result.output
            assert put.call_count == 1
            assert put.call_args.kwargs["json"] == {
                "user_id": "dev",
                "expected_fact_sha256": note["fact_sha256"],
                "expected_workspace_root": None,
                "workspace_root": "/project/a",
            }
            stale = runner.invoke(app, args)
            assert stale.exit_code == 1 and put.call_count == 2
            get.assert_not_called()
            post.assert_not_called()
    with TestClient(make_app(tmp_path, provider)) as api:
        moved = api.get("/memory", params={"user_id": "dev"}).json()["notes"][0]
        assert moved == {**note, "workspace_root": "/project/a"}
        assert api.get("/runs", params={"user_id": "dev"}).json() == []
    assert provider.calls == []


def test_scope_flags_and_hash_are_required_before_dispatch():
    common = ["memory-scope", "1", "--expected-sha256", "a" * 64]
    with patch("httpx.request") as put:
        for flags in [
            [],
            ["--from-global"],
            ["--to-global"],
            ["--from-global", "--from-workspace", "/a", "--to-global"],
            ["--from-global", "--to-global", "--to-workspace", "/a"],
        ]:
            assert runner.invoke(app, [*common, *flags]).exit_code != 0
        assert (
            runner.invoke(
                app,
                ["memory-scope", "1", "--expected-sha256", "bad", "--from-global", "--to-global"],
            ).exit_code
            != 0
        )
        put.assert_not_called()


@pytest.mark.parametrize("change", ["id", "hash", "root", "status"])
def test_scope_response_identity_mismatch_never_claims_success(change):
    note = {"id": 1, "fact_sha256": "a" * 64, "workspace_root": None}
    record = {"status": "reclassified", "note": note, "scope_note": "ok"}
    if change == "id":
        note["id"] = 2
    if change == "hash":
        note["fact_sha256"] = "b" * 64
    if change == "root":
        note["workspace_root"] = "/unexpected"
    if change == "status":
        record["status"] = "other"
    response = httpx.Response(
        200, json=record, request=httpx.Request("PUT", "http://fixture/memory/1/scope")
    )
    with patch("httpx.request", return_value=response) as put:
        result = runner.invoke(
            app,
            [
                "memory-scope",
                "1",
                "--expected-sha256",
                "a" * 64,
                "--from-workspace",
                "/a",
                "--to-global",
            ],
        )
        assert result.exit_code == 1 and "Outcome may be unknown" in result.output
        assert "Memory note 1 scope=" not in result.output
        assert put.call_count == 1


def test_scope_timeout_has_no_retry_lookup_or_private_error():
    with (
        patch("httpx.request", side_effect=httpx.ReadTimeout("private-marker")) as put,
        patch("httpx.get") as get,
    ):
        result = runner.invoke(
            app,
            [
                "memory-scope",
                "1",
                "--expected-sha256",
                "a" * 64,
                "--from-workspace",
                "/a",
                "--to-global",
            ],
        )
        assert result.exit_code == 1 and "private-marker" not in result.output
        assert "No automatic retry" in result.output and put.call_count == 1
        get.assert_not_called()
