from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cli.main import app, show_context_limits
from tests.api.conftest import ScriptedProvider
from tests.api.test_session_persistence import make_app

runner = CliRunner()


@pytest.mark.parametrize(
    "harness_cap,provider_name", [(0, "ollama"), (8192, "ollama"), (8192, "openai")]
)
def test_cli_context_review_after_restart_distinguishes_limits_without_work(
    tmp_path,
    harness_cap,
    provider_name,
):
    provider = ScriptedProvider()
    provider.name = provider_name
    settings = {
        "max_context_tokens": harness_cap,
        "ollama_num_ctx": 4096,
        "prompt_estimate_bytes_per_token": 4.5,
    }
    with TestClient(make_app(tmp_path, provider, **settings)) as api:
        sid = api.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        saved = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
    with TestClient(make_app(tmp_path, provider, **settings)) as api:

        def post(url, **kwargs):
            assert kwargs["timeout"] == 30
            assert url.endswith(f"/sessions/{sid}/configuration-review")
            return api.post(url.removeprefix("http://localhost:8000"), json=kwargs["json"])

        with patch("httpx.post", side_effect=post) as posts, patch("httpx.get") as get:
            result = runner.invoke(
                app,
                [
                    "configuration-review",
                    sid,
                    "--user-id",
                    "dev",
                    "--message",
                    "private-proposed-input",
                ],
            )
            assert result.exit_code == 0, result.output
            cap = 4096 if provider_name == "ollama" else harness_cap
            assert f"effective_context_limit={cap}" in result.output
            assert f"configured_harness_context_limit={harness_cap}" in result.output
            window = "4096" if provider_name == "ollama" else "unrecorded"
            assert f"ollama_requested_context_limit={window}" in result.output
            assert "prompt_estimate_bytes_per_token=4.5" in result.output
            assert "higher ratios estimate fewer tokens" in result.output
            assert "not backend-attested/tokenizer fit or permission to continue" in result.output
            assert "private-proposed-input" not in result.output
            assert posts.call_count == 1
            get.assert_not_called()
        assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
        assert api.get("/runs", params={"user_id": "dev"}).json() == []
    assert provider.calls == []


@pytest.mark.parametrize("policy", [None, "future-policy\x1b[2J"])
def test_cli_missing_or_unsupported_policy_never_invents_effective_limit(policy, capsys):
    show_context_limits({"max_context_tokens": 8192, "context_limit_policy": policy})
    output = capsys.readouterr().out
    assert "effective_context_limit=unknown" in output
    assert "reported_context_limit=8192" in output
    assert "configured_harness_context_limit=unrecorded" in output
    assert "prompt_estimate_bytes_per_token=unrecorded" in output
    assert "\x1b" not in output


@pytest.mark.parametrize(
    "status,allowance",
    [
        ("blocked", "unavailable"),
        ("not_checked", "unavailable"),
        ("unbounded_context", "uncapped_by_harness"),
    ],
)
def test_cli_missing_allowance_keeps_context_status_meaning(status, allowance):
    record = {
        "status": "reviewed",
        "context": {
            "status": status,
            "estimated_prompt_tokens": None,
            "includes_proposed_message": False,
        },
        "scope_note": "Read-only review",
    }
    response = httpx.Response(200, json=record, request=httpx.Request("POST", "http://fixture"))
    with patch("httpx.post", return_value=response) as post:
        result = runner.invoke(app, ["configuration-review", "sid"])
    assert result.exit_code == 0, result.output
    assert f"context={status}" in result.output
    assert f"request_output_allowance={allowance}" in result.output
    assert "effective_context_limit=unknown" in result.output
    assert post.call_count == 1


def test_cli_context_review_private_error_has_no_retry():
    with patch("httpx.post", side_effect=httpx.ReadTimeout("private-marker")) as post:
        result = runner.invoke(app, ["configuration-review", "sid"])
    assert result.exit_code == 1
    assert "private-marker" not in result.output
    assert "No automatic retry" in result.output and post.call_count == 1
