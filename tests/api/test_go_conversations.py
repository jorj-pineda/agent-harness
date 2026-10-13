"""Configured transport contract with HTTP interception; no live inference."""

import json

import httpx
from fastapi.testclient import TestClient

from api.server import Components, create_app
from harness.config import Settings
from harness.grounding import Grounder
from harness.router import ProviderRouter
from memory import FactStore
from providers import compatible_transport_options, create_chat_provider


def test_api_binds_session_headers_and_keyed_replay_does_not_dispatch(tmp_path):
    requests = []
    settings = Settings(
        _env_file=None,
        default_provider="openai_compatible",
        openai_compatible_base_url="https://fixture.example/v1",
        openai_compatible_model="fixture",
        openai_compatible_api_key="fixture-private-key",
        openai_compatible_transport="opencode_go",
        memory_db_path=tmp_path / "facts.db",
        session_db_path=tmp_path / "sessions.db",
    )
    root = tmp_path / "repo"
    root.mkdir()
    (root / "user.txt").write_text("pre-existing work")

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Explanation"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            },
        )

    def factory(_):
        provider = create_chat_provider(
            "openai",
            api_key=settings.openai_compatible_api_key,
            model="fixture",
            provider_name="openai_compatible",
            base_url=settings.openai_compatible_base_url,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            **compatible_transport_options(settings.openai_compatible_transport),
        )
        providers = {"openai_compatible": provider}
        return Components(
            providers=providers,
            router=ProviderRouter(providers, default="openai_compatible"),
            embedder=None,
            collection=None,
            fact_store=FactStore(settings.memory_db_path),
            grounder=Grounder(escalation_threshold=0.55),
        )

    def app():
        return create_app(settings=settings, components_factory=factory)

    with TestClient(app()) as client:
        sessions = [
            client.post(
                "/sessions",
                json={
                    "user_id": "dev",
                    "workspace_root": str(root),
                },
            ).json()["session_id"]
            for _ in range(2)
        ]
        first = {
            "user_id": "dev",
            "session_id": sessions[0],
            "message": "Explain the code",
            "request_id": "first",
        }
        response = client.post("/chat", json=first)
        assert response.status_code == 200, response.text
        original = response.json()
        assert client.post("/chat", json=first).json() == original
        assert client.post("/chat", json={**first, "request_id": "second"}).status_code == 200
        assert (
            client.post(
                "/chat", json={**first, "session_id": sessions[1], "request_id": "third"}
            ).status_code
            == 200
        )
        assert [request.headers["x-opencode-session"] for request in requests] == [
            sessions[0],
            sessions[0],
            sessions[1],
        ]
        for request in requests:
            assert request.headers["user-agent"] == "agent-harness/0.1"
            assert json.loads(request.content)["model"] == "fixture"
        profile = client.get(f"/runs/{original['run_id']}", params={"user_id": "dev"}).json()[
            "execution_profile"
        ]
        assert profile["provider_options_sha256"]
        assert "fixture-private-key" not in str(profile)
        assert sessions[0] not in str(profile)
    with TestClient(app()) as client:
        assert client.post("/chat", json=first).json() == original
        assert len(requests) == 3
    assert (root / "user.txt").read_text() == "pre-existing work"
