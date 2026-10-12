"""Durable metadata survives interrupted/failed tools without enabling replay."""

import json
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

from harness.session_store import SessionStore
from providers.base import ToolCall

from .conftest import ScriptedProvider, make_response
from .test_session_persistence import make_app
from .test_stream import _parse_sse


@pytest.mark.parametrize("transport", ["http", "stream"])
def test_progress_is_durable_owner_scoped_and_retries_do_not_duplicate_events(tmp_path, transport):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("user = 1\n")
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="private-id",
                    name="write_file",
                    arguments={"path": "a.py", "content": "user = 1\nagent = 2\n"},
                ),
                ToolCall(
                    id="unknown",
                    name="model-supplied-private-name",
                    arguments={"secret": "fixture"},
                ),
            ]
        ),
        make_response(content="Done"),
    )
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        payload = {"user_id": "dev", "session_id": sid, "message": "Fix", "request_id": "one"}
        if transport == "http":
            result = client.post("/chat", json=payload).json()
        else:
            events = _parse_sse(client.get("/chat/stream", params=payload).text)
            result = json.loads(events[-1]["data"])["response"]
            assert [event["event"] for event in events] == [
                "tool_start",
                "tool_end",
                "tool_start",
                "tool_end",
                "turn_done",
            ]
        run_id = result["run_id"]
        endpoint = f"/runs/{run_id}/tool-progress"
        response = client.get(endpoint, params={"user_id": "dev"})
        progress = response.json()
        assert progress["coverage"] == "recorded"
        assert progress["observed_events"] == 4
        assert [event["tool"] for event in progress["events"]] == [
            "write_file",
            "write_file",
            "unknown_tool",
            "unknown_tool",
        ]
        assert progress["events"][1]["error_observed"] is False
        assert progress["events"][3]["error_observed"] is True
        assert all(
            word not in response.text
            for word in (
                "private-id",
                "model-supplied-private-name",
                "agent = 2",
                "secret",
                '"arguments"',
                '"result_snippet"',
            )
        )
        assert client.get(endpoint, params={"user_id": "other"}).status_code == 403
        assert (
            client.get("/runs/missing/tool-progress", params={"user_id": "dev"}).status_code == 404
        )
        assert client.get(endpoint, params={"user_id": ""}).status_code == 422
        archive = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted)) as client:
        assert client.post("/chat", json=payload).json() == result
        assert client.get(endpoint, params={"user_id": "dev"}).json() == progress
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archive
        assert (repo / "a.py").read_text() == "user = 1\nagent = 2\n"
        assert restarted.calls == []


@pytest.mark.parametrize("failure_at", ["tool_start", "tool_end"])
def test_storage_failure_stops_following_work_and_keeps_prior_snapshot(
    tmp_path, monkeypatch, failure_at
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("user = 1\n")
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="first",
                    name="write_file",
                    arguments={"path": "a.py", "content": "user = 1\nagent = 2\n"},
                ),
                ToolCall(
                    id="second", name="write_file", arguments={"path": "b.py", "content": "no"}
                ),
            ]
        ),
        make_response(content="Unreached"),
    )
    original = SessionStore.record_tool_event

    def fail(store, run_id, event):
        if event.type == failure_at:
            raise OSError("private fixture detail must not escape")
        return original(store, run_id, event)

    monkeypatch.setattr(SessionStore, "record_tool_event", fail)
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        archive = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        result = client.post(
            "/chat",
            json={"user_id": "dev", "session_id": sid, "message": "Fix", "request_id": "one"},
        )
        assert result.status_code == 503
        assert "private fixture" not in result.text
        runs = client.get("/runs", params={"user_id": "dev"}).json()
        assert runs[0]["status"] == "failed" and runs[0]["response"] is None
        progress = client.get(
            f"/runs/{runs[0]['run_id']}/tool-progress", params={"user_id": "dev"}
        ).json()
        assert progress["observed_events"] == (0 if failure_at == "tool_start" else 1)
        assert len(provider.calls) == 1
        assert not (repo / "b.py").exists()
        assert (repo / "a.py").read_text() == (
            "user = 1\n" if failure_at == "tool_start" else "user = 1\nagent = 2\n"
        )
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archive
        assert (
            client.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Continue"}
            ).status_code
            == 409
        )


def test_forced_process_death_retains_real_edit_metadata_without_final_response(tmp_path):
    script = r"""
import asyncio, json, sys, time
from pathlib import Path
from fastapi.testclient import TestClient
from tests.api.conftest import ScriptedProvider, make_response
from tests.api.test_session_persistence import make_app
from providers.base import ToolCall
root=Path(sys.argv[1]); repo=root/'repo'; repo.mkdir(); (repo/'a.py').write_text('user = 1\n')
class Held(ScriptedProvider):
    async def chat(self, messages, **kwargs):
        if self.calls:
            (root/'ready').write_text('durable edit completed')
            await asyncio.Event().wait()
        return await super().chat(messages, **kwargs)
provider=Held(); provider.script(make_response(tool_calls=[ToolCall(id='edit',name='write_file',arguments={'path':'a.py','content':'user = 1\nagent = 2\n'})]))
with TestClient(make_app(root,provider)) as client:
    sid=client.post('/sessions',json={'user_id':'dev','workspace_root':str(repo)}).json()['session_id']
    record=client.post('/runs',json={'user_id':'dev','session_id':sid,'message':'Fix','request_id':'one'}).json()
    (root/'identity.json').write_text(json.dumps(record))
    time.sleep(30)
"""
    with subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) as process:
        try:
            until = time.monotonic() + 10
            while (
                not (tmp_path / "ready").exists() or not (tmp_path / "identity.json").exists()
            ) and time.monotonic() < until:
                assert process.poll() is None, "Fixture server exited before edit"
                time.sleep(0.01)
            assert (tmp_path / "ready").exists()
            record = json.loads((tmp_path / "identity.json").read_text())
            process.kill()
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted)) as client:
        run_id = record["run_id"]
        saved = client.get(f"/runs/{run_id}", params={"user_id": "dev"}).json()
        assert saved["status"] == "interrupted" and saved["response"] is None
        progress = client.get(f"/runs/{run_id}/tool-progress", params={"user_id": "dev"}).json()
        assert progress["observed_events"] == 2
        assert [event["type"] for event in progress["events"]] == ["tool_start", "tool_end"]
        assert progress["events"][1]["error_observed"] is False
        archive = client.get(
            f"/sessions/{record['request']['session_id']}", params={"user_id": "dev"}
        ).json()
        assert archive["responses"] == []
        assert (tmp_path / "repo" / "a.py").read_text() == "user = 1\nagent = 2\n"
        assert client.post("/chat", json=record["request"]).status_code == 409
        assert restarted.calls == []


def test_active_empty_journal_and_legacy_coverage_are_distinct(tmp_path):
    from harness.runs import RunInput

    from .test_runs import HeldProvider

    store = SessionStore(tmp_path / "sessions.db")
    legacy, _ = store.create_run(
        RunInput(user_id="dev", session_id="legacy", message="Old", request_id="legacy"),
        provider="scripted",
        configured_model=None,
    )
    store.close()
    provider = HeldProvider()
    provider.script(make_response(content="Explained"))
    with TestClient(make_app(tmp_path, provider)) as client:
        endpoint = f"/runs/{legacy.run_id}/tool-progress"
        assert client.get(endpoint, params={"user_id": "dev"}).json()["coverage"] == "not_recorded"
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        payload = {"user_id": "dev", "session_id": sid, "message": "Explain", "request_id": "one"}
        record = client.post("/runs", json=payload).json()
        try:
            assert provider.started.wait(2)
            progress = client.get(
                f"/runs/{record['run_id']}/tool-progress", params={"user_id": "dev"}
            ).json()
            assert progress["coverage"] == "recorded" and progress["events"] == []
            assert progress["observed_events"] == 0
        finally:
            provider.release.set()
        assert client.post("/chat", json=payload).json()["answer"] == "Explained"
        assert len(provider.calls) == 1
        assert (
            client.get(f"/runs/{record['run_id']}/tool-progress", params={"user_id": "dev"}).json()
            == progress
        )
