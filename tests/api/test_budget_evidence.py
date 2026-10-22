"""Compact provider-boundary observations survive failures/restart without replay."""

import json
from contextlib import closing

import pytest
from fastapi.testclient import TestClient

from harness.budget_evidence import MAX_SAVED_BUDGET_EVENTS, BudgetObservation
from harness.session_store import SessionStore
from providers.base import ProviderCallObservation, ProviderError, ToolCall

from .conftest import ScriptedProvider, make_response
from .test_session_persistence import make_app
from .test_stream import _parse_sse


@pytest.mark.parametrize("transport", ["http", "stream"])
def test_budget_stop_keeps_reported_usage_and_request_identity_after_restart(tmp_path, transport):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = "private-source" * 1000
    (repo / "a.py").write_text(source)
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[ToolCall(id="private-call", name="read_file", arguments={"path": "a.py"})],
            prompt_tokens=100,
            completion_tokens=5,
        )
    )
    with TestClient(make_app(tmp_path, provider, max_total_tokens_per_turn=7000)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        payload = {"user_id": "dev", "session_id": sid, "message": "Inspect", "request_id": "one"}
        if transport == "http":
            result = client.post("/chat", json=payload).json()
        else:
            events = _parse_sse(client.get("/chat/stream", params=payload).text)
            assert [e["event"] for e in events] == ["tool_start", "tool_end", "turn_done"]
            result = json.loads(events[-1]["data"])["response"]
        assert result["completion_status"] == "budget_exhausted"
        endpoint = f"/runs/{result['run_id']}"
        saved = client.get(endpoint, params={"user_id": "dev"}).json()
        evidence = saved["budget_evidence"]
        request, response, stop = evidence["events"]
        assert [e["kind"] for e in evidence["events"]] == [
            "request",
            "response",
            "pre_request_stop",
        ]
        assert request["identity"]["format"] == "normalized-chat-request-v1"
        assert request["max_tokens"] > 0
        assert response["usage"] == {"prompt_tokens": 100, "completion_tokens": 5}
        assert stop["prompt_tokens_used"] == 100 and stop["completion_tokens_used"] == 5
        assert stop["estimated_prompt_tokens"] + 256 > 7000 - 105
        assert stop["estimated_prompt_tokens"] == (
            stop["heuristic_prompt_tokens"] + stop["anchor_adjustment_tokens"]
        )
        assert evidence["observed_events"] == 3 and not evidence["events_truncated"]
        for private in ("private-source", "private-call", str(repo), '"arguments"', '"content"'):
            assert private not in json.dumps(evidence)
        assert client.get(endpoint, params={"user_id": "other"}).status_code == 403
        assert len(provider.calls) == 1
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted)) as client:
        assert client.get(endpoint, params={"user_id": "dev"}).json() == saved
        assert client.post("/chat", json=payload).json() == result
        assert restarted.calls == []
        assert (repo / "a.py").read_text() == source


def test_provider_failure_keeps_known_prefix_and_unknown_failed_usage(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("user = 1\n")

    class Failing(ScriptedProvider):
        async def chat(self, *args, **kwargs):
            if self.calls:
                raise ProviderError("private provider detail")
            return await super().chat(*args, **kwargs)

    provider = Failing()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "a.py", "content": "user = 1\nagent = 2\n"},
                )
            ],
            prompt_tokens=111,
            completion_tokens=7,
        )
    )
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        payload = {"user_id": "dev", "session_id": sid, "message": "Fix", "request_id": "one"}
        with pytest.raises(ProviderError, match="private provider detail"):
            client.post("/chat", json=payload)
        saved = client.get("/runs", params={"user_id": "dev"}).json()[0]
        assert saved["status"] == "failed" and saved["response"] is None
        evidence = saved["budget_evidence"]
        assert [e["kind"] for e in evidence["events"]] == [
            "request",
            "response",
            "request",
            "request_failure",
        ]
        assert evidence["events"][1]["usage"] == {"prompt_tokens": 111, "completion_tokens": 7}
        assert evidence["events"][-1]["usage"] is None
        assert evidence["events"][-1]["prompt_tokens_used"] is None
        assert "private provider" not in json.dumps(evidence)
    with TestClient(make_app(tmp_path, ScriptedProvider())) as client:
        assert client.get(f"/runs/{saved['run_id']}", params={"user_id": "dev"}).json() == saved
        assert client.post("/chat", json=payload).status_code == 409
        assert (repo / "a.py").read_text() == "user = 1\nagent = 2\n"


@pytest.mark.parametrize("failure_kind", ["request", "response"])
def test_evidence_storage_failure_prevents_following_inference_or_tools(
    tmp_path, monkeypatch, failure_kind
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("user = 1\n")
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(id="edit", name="write_file", arguments={"path": "a.py", "content": "bad"})
            ]
        )
    )
    original = SessionStore.record_budget_observation

    def fail(store, run_id, event):
        if event.kind == failure_kind:
            raise OSError("private storage detail")
        return original(store, run_id, event)

    monkeypatch.setattr(SessionStore, "record_budget_observation", fail)
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        response = client.post(
            "/chat", json={"user_id": "dev", "session_id": sid, "message": "Fix"}
        )
        assert response.status_code == 503 and "private storage" not in response.text
        saved = client.get("/runs", params={"user_id": "dev"}).json()[0]
        assert saved["status"] == "failed"
        assert len(saved["budget_evidence"]["events"]) == (0 if failure_kind == "request" else 1)
        assert len(provider.calls) == (0 if failure_kind == "request" else 1)
        assert (repo / "a.py").read_text() == "user = 1\n"


def test_budget_prevention_does_not_invent_a_request(tmp_path):
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider, max_context_tokens=1)) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        result = client.post(
            "/chat", json={"user_id": "dev", "session_id": sid, "message": "Inspect"}
        ).json()
        evidence = client.get(f"/runs/{result['run_id']}", params={"user_id": "dev"}).json()[
            "budget_evidence"
        ]
        assert [e["kind"] for e in evidence["events"]] == ["pre_request_stop"]
        assert evidence["events"][0]["prompt_tokens_used"] == 0
        assert provider.calls == []


def test_storage_bound_and_legacy_unknown(tmp_path):
    from harness.config import Settings
    from harness.execution_profile import capture_execution_profile
    from harness.runs import RunInput, RunRecord
    from tools import ToolRegistry

    with closing(SessionStore(tmp_path / "store.db")) as store:
        record, _ = store.create_run(
            RunInput(user_id="dev", session_id="s", message="Inspect", request_id="one"),
            provider="ollama",
            configured_model="gemma4",
        )
        assert record.budget_evidence is None
        store.record_execution_profile(
            record.run_id,
            capture_execution_profile(Settings(_env_file=None), "ollama", ToolRegistry()),
        )
        event = BudgetObservation(
            kind="request_failure",
            iteration=0,
            message_count=1,
            call_observation=ProviderCallObservation(duration_ms=12.5),
        )
        for _ in range(MAX_SAVED_BUDGET_EVENTS + 1):
            store.record_budget_observation(record.run_id, event)
        evidence = store.get_run(record.run_id).budget_evidence
        assert len(evidence.events) == MAX_SAVED_BUDGET_EVENTS
        assert evidence.observed_events == MAX_SAVED_BUDGET_EVENTS + 1 and evidence.events_truncated
        assert evidence.events[-1].call_observation.duration_ms == 12.5
        store.fail_run(record.run_id)
        with pytest.raises(ValueError):
            store.record_budget_observation(record.run_id, event)
        payload = store.get_run(record.run_id).model_dump()
        del payload["budget_evidence"]
        assert RunRecord.model_validate(payload).budget_evidence is None
        legacy_event = event.model_dump()
        del legacy_event["call_observation"]
        assert BudgetObservation.model_validate(legacy_event).call_observation is None


def test_missing_usage_remains_unknown_at_post_response_stop(tmp_path):
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[ToolCall(id="private", name="remember_fact", arguments={"text": "never"})],
            prompt_tokens=None,
            completion_tokens=7,
        )
    )
    with TestClient(make_app(tmp_path, provider, max_total_tokens_per_turn=7000)) as client:
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        result = client.post(
            "/chat", json={"user_id": "dev", "session_id": sid, "message": "Inspect"}
        ).json()
        assert result["completion_status"] == "incomplete"
        saved = client.get(f"/runs/{result['run_id']}", params={"user_id": "dev"}).json()
        evidence = saved["budget_evidence"]
        assert [e["kind"] for e in evidence["events"]] == [
            "request",
            "response",
            "post_response_stop",
        ]
        assert evidence["events"][1]["usage"] == {"prompt_tokens": None, "completion_tokens": 7}
        assert evidence["events"][-1]["prompt_tokens_used"] is None
        assert evidence["events"][-1]["completion_tokens_used"] == 7
        assert all(call["result"] is None for call in result["tool_calls"])
