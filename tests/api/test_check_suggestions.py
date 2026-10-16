import asyncio
import threading

import httpx
from fastapi.testclient import TestClient

from harness.admission import turn_admission
from harness.check_suggestions import CheckSuggestions

from .conftest import ScriptedProvider
from .test_session_persistence import make_app


def test_owner_restart_read_only_suggestions_and_busy(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = repo / "package.json"
    manifest.write_text('{"scripts":{"test:run":"do-not-execute private-marker"}}')
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        saved = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
    with TestClient(make_app(tmp_path, provider)) as client:
        url = f"/sessions/{sid}/check-suggestions"
        assert client.get(url, params={"user_id": "other"}).status_code == 403
        response = client.get(url, params={"user_id": "dev"})
        assert response.status_code == 200
        report = response.json()
        assert report["session_id"] == sid and report["status"] == "inspected"
        assert report["suggestions"] == [
            {"argv": ["npm", "run", "test:run"], "evidence": ["package.json"]}
        ]
        assert "private-marker" not in response.text
        with turn_admission.claim("other-session", str(repo.resolve())) as busy:
            assert busy is None
            assert client.get(url, params={"user_id": "dev"}).json()["status"] == "busy"
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
        assert client.get("/runs", params={"user_id": "dev"}).json() == []
        assert manifest.read_text().endswith('private-marker"}}')
    assert provider.calls == []


def test_disconnected_inspection_keeps_admission_until_worker_settles(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    provider = ScriptedProvider()
    started, release = threading.Event(), threading.Event()

    def held(root):
        started.set()
        if not release.wait(5):
            raise TimeoutError("test worker did not settle")
        return CheckSuggestions(status="inspected")

    monkeypatch.setattr("api.server.suggest_project_checks", held)
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]

        async def request():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=client.app), base_url="http://fixture"
            ) as api:
                return await api.get(
                    f"/sessions/{sid}/check-suggestions", params={"user_id": "dev"}
                )

        future = client.portal.start_task_soon(request)
        try:
            assert started.wait(2)
            future.cancel()
            with turn_admission.claim("other", str(repo.resolve())) as busy:
                assert busy is not None
        finally:
            release.set()

        async def settle():
            await asyncio.gather(*list(client.app.state.components.tasks), return_exceptions=True)

        client.portal.call(settle)
        with turn_admission.claim("other", str(repo.resolve())) as busy:
            assert busy is None
    assert provider.calls == []
