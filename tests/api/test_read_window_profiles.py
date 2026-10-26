import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from api.server import Components, create_app
from harness.config import Settings
from harness.grounding import Grounder
from harness.router import ProviderRouter
from memory import FactStore
from providers.base import ToolCall
from tests.api.conftest import make_response
from tests.api.test_continuation import async_wait, body, known_provider
from tests.api.test_stream import _parse_sse

PROFILE = {
    "provider": "ollama",
    "model": "gemma4:12b",
    "max_context_tokens": 8192,
    "max_request_output_tokens": 512,
    "read_file_default_lines": 2,
}


def reply(**kwargs):
    return make_response(prompt_tokens=10, completion_tokens=5, **kwargs)


def application(root, provider, profile):
    settings = Settings(
        _env_file=None,
        model_profiles={"local": profile},
        model_profile="local",
        read_file_default_lines=5,
        memory_db_path=root / "memory.db",
        session_db_path=root / "sessions.db",
    )

    def factory(_):
        providers = {"ollama": provider}
        return Components(
            providers=providers,
            router=ProviderRouter(providers, default="ollama"),
            embedder=None,
            collection=None,
            fact_store=FactStore(settings.memory_db_path),
            grounder=Grounder(escalation_threshold=0.55),
        )

    return create_app(settings=settings, components_factory=factory)


def source(root, transport="http"):
    repo = root / "repo"
    repo.mkdir()
    original = b"value = 1\nline = 2\nuser = 3\n"
    (repo / "a.py").write_bytes(original)
    (repo / "notes.txt").write_text("pre-existing user work\n")
    provider = known_provider()
    provider.script(
        reply(tool_calls=[ToolCall(id="read", name="read_file", arguments={"path": "a.py"})]),
        reply(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="replace_text",
                    arguments={
                        "path": "a.py",
                        "old_text": "value = 1",
                        "new_text": "value = 2",
                        "expected_sha256": hashlib.sha256(original).hexdigest(),
                    },
                )
            ]
        ),
        reply(content="Updated."),
    )
    with TestClient(application(root, provider, PROFILE)) as client:
        sid = client.post(
            "/sessions",
            json={
                "user_id": "dev",
                "workspace_root": str(repo),
                "checkpoint_turns": True,
            },
        ).json()["session_id"]
        payload = {"user_id": "dev", "session_id": sid, "message": "Update a.py"}
        if transport == "http":
            response = client.post("/chat", json=payload)
            assert response.status_code == 200, response.text
            final = response.json()
        else:
            response = client.get("/chat/stream", params=payload)
            assert response.status_code == 200, response.text
            final = json.loads(_parse_sse(response.text)[-1]["data"])["response"]
        assert final["completion_status"] == "completed"
        saved = client.get("/sessions/" + sid, params={"user_id": "dev"}).json()
        run = client.get("/runs/" + final["run_id"], params={"user_id": "dev"}).json()
        assert run["execution_profile"]["settings"]["read_file_default_lines"] == 2
        assert run["before_checkpoint_id"] and run["after_checkpoint_id"]
    assert (repo / "a.py").read_bytes() == b"value = 2\nline = 2\nuser = 3\n"
    read = next(m for m in provider.calls[1][0] if m.role == "tool")
    result = json.loads(read.content)
    assert result["end_line"] == 2 and result["read_window"]["next_start_line"] == 3
    return repo, sid, saved, run


@pytest.mark.parametrize("transport", ["http", "stream"])
def test_profile_window_survives_restart_and_matching_keyed_fork(tmp_path, transport):
    repo, sid, saved, source_run = source(tmp_path, transport)
    provider = known_provider()
    provider.script(
        reply(
            tool_calls=[
                ToolCall(
                    id="fork-read",
                    name="read_file",
                    arguments={"path": "a.py"},
                )
            ]
        ),
        reply(content="Inspected."),
    )
    request = {**body(saved), "message": "Inspect updated a.py"}
    endpoint = "/sessions/" + sid + "/continue"
    with TestClient(application(tmp_path, provider, PROFILE)) as client:
        assert client.get("/sessions/" + sid, params={"user_id": "dev"}).json() == saved
        assert (
            client.get("/runs/" + source_run["run_id"], params={"user_id": "dev"}).json()
            == source_run
        )
        submitted = client.post(endpoint, json=request)
        assert submitted.status_code == 202, submitted.text
        rid = submitted.json()["run_id"]
        client.portal.call(async_wait, client.app.state.components, rid)
        finished = client.get("/runs/" + rid, params={"user_id": "dev"}).json()
        assert finished["response"]["completion_status"] == "completed"
        assert finished["execution_profile"] == source_run["execution_profile"]
        assert finished["request"]["session_id"] != sid
        assert client.get("/sessions/" + sid, params={"user_id": "dev"}).json() == saved
        assert client.post(endpoint, json=request).json() == finished
    reads = [
        json.loads(m.content)
        for m in provider.calls[-1][0]
        if m.role == "tool" and m.tool_name == "read_file"
    ]
    assert reads[-1]["content"] == "value = 2\nline = 2"
    assert reads[-1]["read_window"]["has_more"]
    restarted = known_provider()
    with TestClient(application(tmp_path, restarted, PROFILE)) as client:
        assert client.post(endpoint, json=request).json() == finished
        assert client.get("/sessions/" + sid, params={"user_id": "dev"}).json() == saved
    assert restarted.calls == []
    assert (repo / "notes.txt").read_text() == "pre-existing user work\n"
    assert (repo / "a.py").read_bytes() == b"value = 2\nline = 2\nuser = 3\n"


@pytest.mark.parametrize("change", ["disabled", "larger", "inherit", "workspace"])
def test_changed_profile_window_or_workspace_blocks_fork_before_tools(tmp_path, change):
    repo, sid, saved, _ = source(tmp_path)
    profile = dict(PROFILE)
    if change == "disabled":
        profile["read_file_default_lines"] = 0
    elif change == "larger":
        profile["read_file_default_lines"] = 3
    elif change == "inherit":
        profile.pop("read_file_default_lines")
    else:
        (repo / "a.py").write_text("later user edit\n")
    before = {p.name: p.read_bytes() for p in repo.iterdir()}
    provider = known_provider()
    with TestClient(application(tmp_path, provider, profile)) as client:
        submitted = client.post("/sessions/" + sid + "/continue", json=body(saved))
        assert submitted.status_code == 202, submitted.text
        rid = submitted.json()["run_id"]
        client.portal.call(async_wait, client.app.state.components, rid)
        run = client.get("/runs/" + rid, params={"user_id": "dev"}).json()
        assert run["response"]["completion_status"] == "blocked"
        assert run["execution_profile"] is None and run["before_checkpoint_id"] is None
        assert run["after_checkpoint_id"] is None
        assert client.get("/sessions/" + sid, params={"user_id": "dev"}).json() == saved
        progress = client.get("/runs/" + rid + "/tool-progress", params={"user_id": "dev"})
        assert progress.status_code == 200, progress.text
        assert progress.json()["events"] == []
    assert provider.calls == []
    assert {p.name: p.read_bytes() for p in repo.iterdir()} == before
