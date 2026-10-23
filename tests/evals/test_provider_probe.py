"""Exercise the same manifest selection and component factory used for source launch."""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.server import Components, create_app
from evals.projects.duodoro.provider_probe import FirstRequestGuard, create_probe_app
from harness.config import Settings
from harness.grounding import Grounder
from harness.router import ProviderRouter
from harness.trace import request_identity
from memory import FactStore
from providers.base import ChatMessage, ProviderResponse, TokenUsage, ToolCall, ToolSpec


class Fake:
    name = "ollama"

    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    async def chat(self, messages, *, tools=None, temperature=0.0, max_tokens=None):
        self.calls.append(
            (
                [message.model_copy(deep=True) for message in messages],
                tools,
                temperature,
                max_tokens,
            )
        )
        return self.responses.pop(0)

    async def aclose(self):
        self.closed = True


def reply(*, calls=()):
    return ProviderResponse(
        content="SCRIPTED complete" if not calls else "",
        tool_calls=list(calls),
        finish_reason="tool_use" if calls else "stop",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=20),
        model="SCRIPTED-not-Gemma",
        latency_ms=0,
    )


def builder(fake, allocated):
    def build(settings):
        allocated.append(settings)
        settings.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        providers = {"ollama": fake}
        return Components(
            providers=providers,
            router=ProviderRouter(providers, default="ollama"),
            embedder=None,
            collection=None,
            fact_store=FactStore(settings.memory_db_path),
            grounder=Grounder(escalation_threshold=0.55),
        )

    return build


@pytest.fixture
def prepared(tmp_path):
    tasks = []
    for tid in ["T05", "T09"]:
        directory = tmp_path / "tasks" / tid
        workspace = directory / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "work.py").write_text("user = 1\n")
        settings = Settings(
            _env_file=None,
            default_provider="ollama",
            ollama_model="gemma4:12b",
            coding_toolset="whole_file",
            enable_support_tools=False,
            memory_db_path=directory / "state" / "memory.db",
            session_db_path=directory / "state" / "sessions.db",
        )
        data = settings.model_dump_json().encode()
        (directory / "settings.json").write_bytes(data)
        tasks.append(
            {
                "id": tid,
                "directory": str(directory),
                "workspace_root": str(workspace),
                "settings_sha256": hashlib.sha256(data).hexdigest(),
                "source_initial_identity": {"request_sha256": "a" * 64},
            }
        )
    manifest = {"kind": "unrun-two-task-provider-diagnostic-freeze", "tasks": tasks}
    path = tmp_path / "provider-diagnostic-freeze.json"
    path.write_text(json.dumps(manifest))
    # This is the pre-capture file that caused the real startup failure. It must
    # never supply source factory rows, even when it has a different valid hash.
    raw = [{k: v for k, v in task.items() if k != "source_initial_identity"} for task in tasks]
    (tmp_path / "tasks.json").write_text(json.dumps(raw))
    return tmp_path, manifest


@pytest.mark.parametrize("phase", ["source", "review"])
def test_actual_source_factory_loads_augmented_manifest_before_lifecycle(prepared, phase):
    root, _ = prepared
    fake = Fake()
    allocated = []
    app = create_probe_app(root, "T05", phase, components_builder=builder(fake, allocated))
    assert allocated == []  # Resources are still lifecycle-owned.
    with TestClient(app) as client:
        guard = app.state.components.providers["ollama"]
        assert isinstance(guard, FirstRequestGuard)
        assert guard.expected == "a" * 64
        assert guard.review_only is (phase == "review")
        assert client.get("/runs", params={"user_id": "owner"}).json() == []
        assert fake.calls == []
    assert len(allocated) == 1 and fake.closed


@pytest.mark.parametrize("identity", [None, {}, {"request_sha256": "invalid"}])
@pytest.mark.parametrize("phase", ["source", "review"])
def test_missing_capture_identity_rejects_before_allocating_resources(prepared, identity, phase):
    root, manifest = prepared
    manifest["tasks"][0]["source_initial_identity"] = identity
    (root / "provider-diagnostic-freeze.json").write_text(json.dumps(manifest))
    allocated = []
    with pytest.raises(ValidationError):
        create_probe_app(root, "T05", phase, components_builder=builder(Fake(), allocated))
    assert allocated == []
    assert not (root / "tasks/T05/state").exists()


@pytest.mark.parametrize("change", ["settings", "paths", "order", "raw_rows"])
def test_manifest_drift_and_raw_rows_cannot_start_factory(prepared, change):
    root, manifest = prepared
    if change == "settings":
        (root / "tasks/T05/settings.json").write_text("{}")
    elif change == "paths":
        manifest["tasks"][0]["directory"] = str(root)
    elif change == "order":
        manifest["tasks"].reverse()
    else:
        manifest["tasks"] = json.loads((root / "tasks.json").read_text())
    (root / "provider-diagnostic-freeze.json").write_text(json.dumps(manifest))
    allocated = []
    with pytest.raises(ValueError):
        create_probe_app(root, "T05", "source", components_builder=builder(Fake(), allocated))
    assert allocated == []


def test_real_file_source_submission_then_restart_review_uses_same_factory(prepared):
    root, manifest = prepared
    task = manifest["tasks"][0]
    settings = Settings(
        _env_file=None, **json.loads((Path(task["directory"]) / "settings.json").read_text())
    )
    workspace = Path(task["workspace_root"])
    capture = Fake([reply()])
    capture_settings = settings.model_copy(
        update={
            "memory_db_path": root / "capture-memory.db",
            "session_db_path": root / "capture-sessions.db",
        }
    )
    with TestClient(
        create_app(settings=capture_settings, components_factory=builder(capture, []))
    ) as client:
        sid = client.post(
            "/sessions", json={"user_id": "owner", "workspace_root": str(workspace)}
        ).json()["session_id"]
        assert (
            client.post(
                "/chat",
                json={
                    "user_id": "owner",
                    "session_id": sid,
                    "message": "Bounded edit",
                    "provider": "ollama",
                },
            ).status_code
            == 200
        )
    messages, tools, _, allowance = capture.calls[0]
    identity = request_identity(messages, tools or [], allowance)
    task["source_initial_identity"] = identity.model_dump()
    (root / "provider-diagnostic-freeze.json").write_text(json.dumps(manifest))
    fake = Fake(
        [
            reply(
                calls=[
                    ToolCall(
                        id="write",
                        name="write_file",
                        arguments={"path": "work.py", "content": "user = 1\nagent = 2\n"},
                    )
                ]
            ),
            reply(),
        ]
    )
    app = create_probe_app(root, "T05", "source", components_builder=builder(fake, []))
    with TestClient(app) as client:
        sid = client.post(
            "/sessions", json={"user_id": "owner", "workspace_root": str(workspace)}
        ).json()["session_id"]
        result = client.post(
            "/chat",
            json={
                "user_id": "owner",
                "session_id": sid,
                "message": "Bounded edit",
                "provider": "ollama",
                "request_id": "one",
            },
        )
        assert result.status_code == 200
        saved = client.get("/runs", params={"user_id": "owner"}).json()[0]
        assert (workspace / "work.py").read_text() == "user = 1\nagent = 2\n"
    assert len(fake.calls) == 2 and fake.closed
    review = Fake()
    app = create_probe_app(root, "T05", "review", components_builder=builder(review, []))
    with TestClient(app) as client:
        assert client.get(f"/runs/{saved['run_id']}", params={"user_id": "owner"}).json() == saved
        assert review.calls == []
    assert review.closed


@pytest.mark.parametrize("change", ["messages", "tools", "allowance", "temperature", "review"])
def test_first_input_drift_or_review_never_forwards(change):
    messages = [ChatMessage(role="user", content="task")]
    tools = [ToolSpec(name="read_file", description="read", parameters_schema={})]
    expected = request_identity(messages, tools, 2048).request_sha256
    fake = Fake([reply()])
    guard = FirstRequestGuard(fake, expected, review_only=change == "review")
    allowance = 2048
    temperature = 0.0
    if change == "messages":
        messages[0].content = "changed"
    elif change == "tools":
        tools = []
    elif change == "allowance":
        allowance = 2047
    elif change == "temperature":
        temperature = 0.1
    with pytest.raises(ValueError):
        asyncio.run(
            guard.chat(messages, tools=tools, max_tokens=allowance, temperature=temperature)
        )
    assert fake.calls == []
    asyncio.run(guard.aclose())
    assert fake.closed
