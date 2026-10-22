"""Real-file scripted HTTP observations stay bounded and reviewable after restart."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from providers.ollama import OllamaProvider

from .conftest import ScriptedProvider
from .test_session_persistence import make_app
from .test_stream import _parse_sse


@pytest.mark.parametrize("transport", ["http", "stream"])
@pytest.mark.parametrize("terminal", ["missing_usage", "read_timeout"])
def test_observations_preserve_edits_and_unknown_usage_across_restart(
    tmp_path, transport, terminal
):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "work.py"
    target.write_text("user = 1\n")
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "write_file",
                                    "arguments": {
                                        "path": "work.py",
                                        "content": "user = 1\nagent = 2\n",
                                    },
                                }
                            }
                        ],
                    },
                    "done_reason": "stop",
                    "prompt_eval_count": 100,
                    "eval_count": 20,
                },
            )
        assert len(calls) == 2
        if terminal == "read_timeout":
            raise httpx.ReadTimeout("private-transport-detail", request=request)
        return httpx.Response(
            200,
            json={
                "message": {"content": ""},
                "prompt_eval_count": None,
                "done_reason": "private-native-reason",
            },
        )

    provider = OllamaProvider(
        host="http://private-endpoint",
        model="fixture",
        embed_model="unused",
        transport=httpx.MockTransport(respond),
    )
    with TestClient(
        make_app(tmp_path, provider, max_request_output_tokens=2048),
        raise_server_exceptions=False,
    ) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        payload = {
            "user_id": "dev",
            "session_id": sid,
            "message": "Bounded edit",
            "request_id": "one",
        }
        if transport == "http":
            result = client.post("/chat", json=payload)
            assert result.status_code == (500 if terminal == "read_timeout" else 200)
        else:
            events = _parse_sse(client.get("/chat/stream", params=payload).text)
            assert events[-1]["event"] == ("error" if terminal == "read_timeout" else "turn_done")
        saved = client.get("/runs", params={"user_id": "dev"}).json()[0]
        evidence = saved["budget_evidence"]
        first = evidence["events"][1]
        assert first["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}
        observation = first["call_observation"]
        assert observation["duration_ms"] >= 0
        assert (
            observation["prompt_usage_field"] == observation["completion_usage_field"] == "present"
        )
        assert observation["terminal_reason"] == "stop"
        assert observation["content_empty"] is True and observation["tool_call_count"] == 1
        assert evidence["events"][0]["call_observation"] is None
        final = evidence["events"][3]
        assert final["call_observation"]["duration_ms"] >= 0
        if terminal == "read_timeout":
            assert final["kind"] == "request_failure" and final["usage"] is None
            assert final["prompt_tokens_used"] is None
            assert final["call_observation"] == saved["provider_failure"]["call_observation"]
            assert final["call_observation"]["prompt_usage_field"] is None
            assert saved["response"] is None
        else:
            assert final["kind"] == "response"
            assert final["call_observation"]["prompt_usage_field"] == "null"
            assert final["call_observation"]["completion_usage_field"] == "absent"
            assert final["call_observation"]["terminal_reason"] == "other"
            assert final["call_observation"]["tool_call_count"] == 0
            assert evidence["events"][-1]["kind"] == "post_response_stop"
            assert saved["response"]["completion_status"] == "incomplete"
            assert saved["response"]["token_usage"] == {
                "prompt_tokens": None,
                "completion_tokens": None,
            }
        assert "private" not in json.dumps(evidence)
        assert '"arguments"' not in json.dumps(evidence)
        assert target.read_text() == "user = 1\nagent = 2\n"
        assert len(calls) == 2 and all(call["options"]["num_predict"] == 2048 for call in calls)
        assert (
            client.get(f"/runs/{saved['run_id']}", params={"user_id": "other"}).status_code == 403
        )
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted, max_request_output_tokens=2048)) as client:
        assert client.get(f"/runs/{saved['run_id']}", params={"user_id": "dev"}).json() == saved
        assert client.post("/runs", json=payload).json() == saved
        replay = client.post("/chat", json=payload)
        if terminal == "read_timeout":
            assert replay.status_code == 409
        else:
            assert replay.status_code == 200 and replay.json() == saved["response"]
        assert restarted.calls == [] and target.read_text() == "user = 1\nagent = 2\n"
