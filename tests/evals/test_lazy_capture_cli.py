"""Scripted CLI + default API lifecycle; no hosted access or trial execution."""

import hashlib
import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from api.server import create_app
from cli.main import app as cli
from evals.projects.duodoro.provider_probe import FirstRequestGuard
from evals.projects.duodoro.request_capture import (
    CapturedProvider,
    CapturedTransport,
    RequestCapture,
)
from harness.config import Settings
from providers import create_chat_provider
from tests.evals.test_compatible_request_capture import completed


@pytest.mark.parametrize("outcome", ["completed", "timeout", "missing_usage"])
def test_cli_selected_provider_and_same_lazy_factory_restart_preserve_inputs(tmp_path, outcome):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "work.py"
    target.write_text("user = 1\n")
    target.chmod(0o640)
    protected = workspace / "preexisting.txt"
    protected.write_text("keep this\n")
    directory = tmp_path / "inputs"
    settings = Settings(
        _env_file=None,
        model_profiles={
            "scripted": {
                "provider": "openai_compatible",
                "model": "SCRIPTED-not-Beam",
                "max_context_tokens": 16384,
                "max_request_output_tokens": 2048,
            }
        },
        model_profile="scripted",
        openai_compatible_base_url="https://fixture.invalid/inference/go/openai/v1",
        openai_compatible_api_key="SCRIPTED-private-key",
        openai_compatible_transport="opencode_go",
        enable_support_tools=False,
        memory_db_path=tmp_path / "facts.db",
        session_db_path=tmp_path / "sessions.db",
    )
    seen = []
    allocations = []
    owned = []
    review_only = False

    def respond(request):
        assert not review_only
        seen.append(request)
        assert request.url.path == "/inference/go/openai/v1/chat/completions"
        assert request.headers["User-Agent"] == "agent-harness/0.1"
        assert (directory / f"request-{len(seen):04d}-wire.json").read_bytes() == request.content
        if len(seen) == 2 and outcome == "timeout":
            raise httpx.ReadTimeout("SCRIPTED-private-error", request=request)
        body = completed().json()
        if len(seen) == 1:
            body["choices"][0] = {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "write-once",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": json.dumps(
                                    {"path": "work.py", "content": "user = 1\nagent = 2\n"}
                                ),
                            },
                        }
                    ],
                },
            }
        elif outcome == "missing_usage":
            body.pop("usage")
        return httpx.Response(200, json=body)

    def construct(kind, **kwargs):
        # Intercept only SDK HTTP, retaining the actual selected settings factory,
        # default Components construction and lifecycle-owned FactStore thread.
        allocations.append((kind, kwargs["provider_name"], kwargs["model"]))
        assert kind == "openai" and kwargs["provider_name"] == "openai_compatible"
        capture = RequestCapture(
            directory, lazy=True, chat_path="/inference/go/openai/v1/chat/completions"
        )
        adapter = create_chat_provider(
            kind,
            **kwargs,
            http_client=httpx.AsyncClient(
                transport=CapturedTransport(httpx.MockTransport(respond), capture)
            ),
        )
        owner = CapturedProvider(adapter, capture)
        owned.append((owner, adapter))
        return FirstRequestGuard(owner, "a" * 64, review_only=True) if review_only else owner

    runner = CliRunner()
    submitted = []
    with patch("harness.providers.create_chat_provider", side_effect=construct):
        with TestClient(create_app(settings=settings), raise_server_exceptions=False) as api:
            assert not directory.exists()

            def post(url, **kwargs):
                submitted.append((url, kwargs["json"]))
                return api.post(url, json=kwargs["json"])

            with patch("httpx.post", side_effect=post):
                result = runner.invoke(
                    cli, ["chat", "--workspace", str(workspace)], input="Bounded edit\n"
                )
            assert result.exit_code == 0, result.output
            saved = api.get("/runs", params={"user_id": "cli-user"}).json()
            assert len(saved) == 1
            saved = saved[0]
            session = api.get(
                f"/sessions/{saved['request']['session_id']}", params={"user_id": "cli-user"}
            ).json()
        assert owned[0][1]._client.is_closed()
        assert len(submitted) == 2 and len(seen) == 2
        assert [r.headers["x-opencode-session"] for r in seen] == [
            saved["request"]["session_id"]
        ] * 2
        assert target.read_text() == "user = 1\nagent = 2\n"
        assert target.stat().st_mode & 0o777 == 0o640
        assert protected.read_text() == "keep this\n"
        assert allocations == [("openai", "openai_compatible", "SCRIPTED-not-Beam")]
        assert saved["execution_profile"]["model_profile"]["name"] == "scripted"
        if outcome == "timeout":
            assert saved["status"] == "failed" and saved["response"] is None
            assert saved["budget_evidence"]["events"][-1]["usage"] is None
        else:
            expected = "completed" if outcome == "completed" else "incomplete"
            assert saved["response"]["completion_status"] == expected
            assert saved["response"]["verification_status"] == "not_run"
            if outcome == "missing_usage":
                assert saved["response"]["token_usage"] == {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                }
        evidence = {p.name: (p.read_bytes(), p.stat().st_mode & 0o777) for p in directory.iterdir()}
        assert len(evidence) == 4
        events = [e for e in saved["budget_evidence"]["events"] if e["kind"] == "request"]
        normalized = [
            json.loads(evidence[f"request-{number:04d}-normalized.json"][0]) for number in [1, 2]
        ]
        assert [record["identity"] for record in normalized] == [e["identity"] for e in events]
        for number, request in enumerate(seen, 1):
            assert evidence[f"request-{number:04d}-wire.json"][0] == request.content
            assert json.loads(request.content)["max_tokens"] == 2048
        raw = b"".join(v[0] for v in evidence.values())
        for private in [
            b"SCRIPTED-private-key",
            b"fixture.invalid",
            b"SCRIPTED-private-error",
            saved["request"]["session_id"].encode(),
        ]:
            assert private not in raw
        hashes = {name: hashlib.sha256(data).hexdigest() for name, (data, _) in evidence.items()}
        target.write_text(target.read_text() + "later_user = 3\n")
        review_only = True
        with TestClient(create_app(settings=settings)) as api:
            assert len(allocations) == 2
            assert (
                api.get(
                    f"/sessions/{saved['request']['session_id']}", params={"user_id": "cli-user"}
                ).json()
                == session
            )
            observed_gets = []

            def get(url, **kwargs):
                observed_gets.append(url)
                return api.get(url, params=kwargs.get("params"))

            with (
                patch("httpx.get", side_effect=get),
                patch("httpx.post", side_effect=AssertionError("Review POST")),
            ):
                review = runner.invoke(cli, ["run", saved["run_id"], "--json"])
            assert review.exit_code == 0, review.output
            assert json.loads(review.output) == saved
            assert len(observed_gets) == 1
            assert len(seen) == 2
            assert {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()
            } == hashes
            assert {
                p.name: (p.read_bytes(), p.stat().st_mode & 0o777) for p in directory.iterdir()
            } == evidence
        assert all(adapter._client.is_closed() for _, adapter in owned)
        assert len(allocations) == 2
    assert target.read_text() == "user = 1\nagent = 2\nlater_user = 3\n"
    assert target.stat().st_mode & 0o777 == 0o640 and protected.read_text() == "keep this\n"
