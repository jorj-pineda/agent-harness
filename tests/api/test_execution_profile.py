from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness.session_store import SessionStore
from providers.base import ToolCall

from .conftest import ScriptedProvider, make_response
from .test_runs import HeldProvider
from .test_session_persistence import make_app


@pytest.mark.parametrize("prompt_profile", ["legacy", "compact"])
def test_profile_is_durable_before_inference_and_survives_finalization_and_restart(
    tmp_path: Path, prompt_profile: str
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("user = 1\n")
    provider = HeldProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="write",
                    name="write_file",
                    arguments={"path": "a.py", "content": "user = 1\nagent = 2\n"},
                )
            ]
        ),
        make_response(content="Done"),
    )
    settings = {
        "max_context_tokens": 16000,
        "max_tool_calls_per_turn": 12,
        "coding_toolset": "whole_file",
        "coding_prompt_profile": prompt_profile,
    }
    with TestClient(make_app(tmp_path, provider, **settings)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        payload = {
            "user_id": "dev",
            "session_id": sid,
            "message": "Write a.py",
            "request_id": "profile",
        }
        run_id = client.post("/runs", json=payload).json()["run_id"]
        try:
            assert provider.started.wait(2)
            record = client.get(f"/runs/{run_id}", params={"user_id": "dev"}).json()
            assert record["status"] == "running"
            profile = record["execution_profile"]
            assert profile["settings"]["max_context_tokens"] == 16000
            assert "write_file" in profile["tool_names"]
            assert "replace_text" not in profile["tool_names"]
            assert profile["prompt_version"] == (
                "coding-v3-compact-whole-file-v1"
                if prompt_profile == "compact"
                else "coding-v2-whole-file-memory-scope-v1"
            )
            assert profile["configured_model"] is None  # Custom provider: unknown.
            assert (repo / "a.py").read_text() == "user = 1\n"
            assert client.post("/runs", json=payload).json()["execution_profile"] == profile
        finally:
            provider.release.set()
        response = client.post("/chat", json=payload).json()
        assert response["completion_status"] == "completed"
        system = provider.calls[0][0][0].content
        assert "Use `write_file` with the full file contents" in system
        assert "use `replace_text`" not in system
        if prompt_profile == "compact":
            assert "ENABLE_SUPPORT_TOOLS" not in system
        archive = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        assert archive["responses"][0]["execution_profile"] == profile
        assert (repo / "a.py").read_text() == "user = 1\nagent = 2\n"
    restarted = ScriptedProvider()
    with TestClient(
        make_app(tmp_path, restarted, max_context_tokens=8000, coding_toolset="full")
    ) as client:
        assert (
            client.get(f"/runs/{run_id}", params={"user_id": "dev"}).json()["execution_profile"]
            == profile
        )
        assert client.post("/chat", json=payload).json() == response
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archive
        assert client.post("/chat", json={**payload, "request_id": "new"}).status_code == 409
        assert restarted.calls == []


def test_profile_storage_failure_stops_before_inference_and_leaves_archive_intact(
    tmp_path: Path, monkeypatch
):
    provider = ScriptedProvider()
    provider.script(make_response(content="Must not run"))

    def fail(*args):
        raise OSError("fixture failure")

    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        before = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        monkeypatch.setattr(SessionStore, "record_execution_profile", fail)
        response = client.post(
            "/chat", json={"user_id": "dev", "session_id": sid, "message": "Explain"}
        )
        assert response.status_code == 503
        assert "no inference or tools started" in response.json()["detail"]
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == before
        record = client.get("/runs", params={"user_id": "dev"}).json()[0]
        assert record["status"] == "failed"
        assert record["execution_profile"] is None
        assert provider.calls == []


def test_failed_execution_keeps_captured_configuration(tmp_path: Path):
    provider = ScriptedProvider()  # No responses: raises after profile capture.
    with TestClient(make_app(tmp_path, provider), raise_server_exceptions=False) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        assert (
            client.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Explain"}
            ).status_code
            == 500
        )
        record = client.get("/runs", params={"user_id": "dev"}).json()[0]
        assert record["status"] == "failed"
        assert record["execution_profile"] is not None
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()["responses"] == []


def test_policy_rejection_has_no_invented_execution_profile(tmp_path: Path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        client.post("/chat", json={"user_id": "dev", "session_id": sid, "message": "rm -rf /"})
        record = client.get("/runs", params={"user_id": "dev"}).json()[0]
        assert record["provider"] == "policy"
        assert record["execution_profile"] is None
        assert provider.calls == []


def test_api_and_sqlite_configuration_payloads_exclude_credentials(tmp_path: Path):
    provider = ScriptedProvider()
    provider.name = "ollama"  # Scripted adapter: records configured identity only.
    provider.script(make_response(content="Done"))
    with TestClient(
        make_app(
            tmp_path,
            provider,
            anthropic_api_key="fixture-private-anthropic",
            openai_api_key="fixture-private-openai",
            ollama_host="http://fixture.example/private-endpoint",
            project_check_argv=["pytest", "fixture-private-check"],
            ollama_num_ctx=8192,
            ollama_think=False,
        )
    ) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        response = client.post(
            "/chat", json={"user_id": "dev", "session_id": sid, "message": "Explain"}
        ).json()
        record = client.get(f"/runs/{response['run_id']}", params={"user_id": "dev"}).json()
        archive = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        profile = record["execution_profile"]
        assert profile["configured_model"] == "gemma4"
        assert profile["ollama_num_ctx"] == 8192
        assert profile["ollama_think"] is False
        assert "fixture-private-check" not in str(profile)
        for payload in (str(record), str(archive)):
            assert "fixture-private-anthropic" not in payload
            assert "fixture-private-openai" not in payload
            assert "fixture.example" not in payload
            assert "private-endpoint" not in payload
        assert archive["responses"][0]["execution_profile"] == profile
    with closing(SessionStore(tmp_path / "sessions.db")) as store:
        assert (
            store.get_run(response["run_id"]).execution_profile.model_dump(mode="json") == profile
        )
        assert "fixture-private-anthropic" not in store.get(sid).model_dump_json()
        assert "fixture-private-openai" not in store.get(sid).model_dump_json()
