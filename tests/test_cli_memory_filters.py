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


def test_filtered_cli_pages_after_restart_are_read_only(tmp_path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as api:

        def seed():
            store = api.app.state.components.fact_store
            store.add("dev", "global")
            store.add("dev", "older", workspace_root="/a")
            store.add("dev", "foreign", workspace_root="/b")
            store.add("dev", "newer", workspace_root="/a")
            store.add("other", "private", workspace_root="/a")

        api.portal.call(seed)
    with TestClient(make_app(tmp_path, provider)) as api:

        def get(url, **kwargs):
            return api.get(url.removeprefix("http://localhost:8000"), params=kwargs["params"])

        with patch("httpx.get", side_effect=get) as gets, patch("httpx.request") as mutation:
            args = [
                "memory",
                "--user-id",
                "dev",
                "--namespace",
                "workspace",
                "--workspace-root",
                "/a",
                "--limit",
                "1",
            ]
            first = runner.invoke(app, [*args, "--json"])
            assert first.exit_code == 0, first.output
            page = json.loads(first.output)
            assert [n["preview"] for n in page["notes"]] == ["newer"]
            second = runner.invoke(app, [*args, "--before-id", str(page["next_before_id"])])
            assert second.exit_code == 0 and '"older"' in second.output
            assert '"foreign"' not in second.output and '"global"' not in second.output
            assert gets.call_args.kwargs["params"]["workspace_root"] == "/a"
            global_page = runner.invoke(
                app, ["memory", "--user-id", "dev", "--namespace", "global", "--json"]
            )
            assert [n["preview"] for n in json.loads(global_page.output)["notes"]] == ["global"]
            hint = runner.invoke(app, args)
            assert "keep the same namespace/root filters" in hint.output
            mutation.assert_not_called()
        assert api.get("/runs", params={"user_id": "dev"}).json() == []
    assert provider.calls == []


def test_invalid_cli_namespace_combinations_never_dispatch():
    with patch("httpx.get") as get:
        for args in [
            ["--namespace", "unknown"],
            ["--namespace", "workspace"],
            ["--workspace-root", "/a"],
            ["--namespace", "global", "--workspace-root", "/a"],
        ]:
            assert runner.invoke(app, ["memory", *args]).exit_code != 0
        get.assert_not_called()


@pytest.mark.parametrize(
    "ids,cursor", [([4, 4], None), ([3, 4], None), ([5], None), ([4], True), ([4, 3], 4)]
)
def test_cli_memory_rejects_inconsistent_page_ids_and_cursor(ids, cursor):
    page = {"notes": [{"id": item} for item in ids], "next_before_id": cursor}
    response = httpx.Response(200, json=page, request=httpx.Request("GET", "http://fixture"))
    with patch("httpx.get", return_value=response) as get:
        result = CliRunner().invoke(app, ["memory", "--before-id", "5", "--json"])
    assert result.exit_code == 1
    assert "Memory lookup failed" in result.output
    assert '"notes"' not in result.output
    get.assert_called_once()
