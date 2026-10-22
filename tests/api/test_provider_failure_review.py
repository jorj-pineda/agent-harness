"""Real edits followed by mocked provider failure, durable review and keyed replay."""

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from providers.base import ProviderError, ToolCall
from providers.ollama import OllamaProvider

from .conftest import ScriptedProvider, make_response
from .test_session_persistence import make_app
from .test_stream import _parse_sse


class FailingProvider(ScriptedProvider):
    def __init__(self, failure):
        super().__init__()
        self.failure = failure

        def respond(request):
            if failure == "read_timeout":
                raise httpx.ReadTimeout("private-token private-source", request=request)
            return httpx.Response(503, text="private-response-body", request=request)

        self.upstream = OllamaProvider(
            host="http://private-endpoint",
            model="local",
            embed_model="unused",
            transport=httpx.MockTransport(respond),
        )

    async def chat(self, messages, **kwargs):
        if self._queue:
            return await super().chat(messages, **kwargs)
        self.calls.append((messages, kwargs.get("tools")))
        if self.failure == "unknown":
            raise ProviderError("private-token private-source")
        if self.failure == "non_provider":
            raise RuntimeError("private-token private-source")
        return await self.upstream.chat(messages, **kwargs)

    async def aclose(self):
        await self.upstream.aclose()


@pytest.mark.parametrize("transport", ["http", "stream"])
@pytest.mark.parametrize("failure", ["read_timeout", "http_status", "unknown", "non_provider"])
def test_failed_run_retains_safe_observation_after_edit_and_restart(
    tmp_path: Path, transport, failure
):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "work.py"
    target.write_text("user_work = 1\n")
    provider = FailingProvider(failure)
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "work.py", "content": "user_work = 1\nagent_work = 2\n"},
                )
            ]
        )
    )
    with TestClient(make_app(tmp_path, provider), raise_server_exceptions=False) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        payload = {
            "user_id": "dev",
            "session_id": sid,
            "message": "Make the bounded edit",
            "request_id": "failure-once",
        }
        if transport == "http":
            assert client.post("/chat", json=payload).status_code == 500
        else:
            response = client.get("/chat/stream", params=payload)
            events = _parse_sse(response.text)
            assert [e["event"] for e in events] == ["tool_start", "tool_end", "error"]
            assert "private" not in events[-1]["data"]
        record = client.get("/runs", params={"user_id": "dev"}).json()[0]
        run_id = record["run_id"]
        assert record["status"] == "failed"
        assert record["response"] is None  # No invented completion/check/usage/diff evidence.
        expected = (
            None
            if failure == "non_provider"
            else {
                "kind": failure,
                "http_status": 503 if failure == "http_status" else None,
                "call_observation": record["provider_failure"]["call_observation"],
            }
        )
        assert record["provider_failure"] == expected
        if failure in {"read_timeout", "http_status"}:
            observation = record["provider_failure"]["call_observation"]
            assert observation["duration_ms"] >= 0
            assert all(value is None for key, value in observation.items() if key != "duration_ms")
        elif expected is not None:
            assert expected["call_observation"] is None
        if expected is not None:
            assert f"Provider failure: {failure}" in record["error"]
        assert "private" not in json.dumps(record)
        assert client.get(f"/runs/{run_id}", params={"user_id": "other"}).status_code == 403
        assert client.post("/runs", json=payload).json() == record
        assert client.post("/chat", json=payload).status_code == 409
        assert len(provider.calls) == 2
        assert target.read_text() == "user_work = 1\nagent_work = 2\n"
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted)) as client:
        assert client.get(f"/runs/{run_id}", params={"user_id": "dev"}).json() == record
        assert client.post("/runs", json=payload).json() == record
        assert client.post("/chat", json=payload).status_code == 409
        assert restarted.calls == []
        assert target.read_text() == "user_work = 1\nagent_work = 2\n"
