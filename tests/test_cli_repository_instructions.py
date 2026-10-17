import hashlib
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cli.main import app, show_repository_instructions
from harness.admission import turn_admission
from tests.api.conftest import ScriptedProvider, make_response
from tests.api.test_session_persistence import make_app

runner = CliRunner()


def metadata(status="loaded", **kwargs):
    return {
        "policy": "root-agents-v1",
        "path": "AGENTS.md",
        "max_bytes": 8192,
        "status": status,
        "content_bytes": 3 if status == "loaded" else None,
        "content_sha256": "a" * 64 if status == "loaded" else None,
        "issue": "too_large" if status == "unavailable" else None,
        **kwargs,
    }


@pytest.mark.parametrize("status", ["disabled", "no_workspace", "missing", "loaded", "unavailable"])
def test_known_metadata_keeps_status_and_omits_content(status, capsys):
    show_repository_instructions(
        metadata(status, text="private-content", scope_note="\x1b[2Jprivate")
    )
    output = capsys.readouterr().out
    assert f"current_repository_instructions={status}" in output
    assert "max_bytes=8192" in output and "no parent/nested discovery" in output
    assert ("content_sha256=" in output) == (status == "loaded")
    assert ("issue=too_large" in output) == (status == "unavailable")
    assert "\x1b" not in output and "private" not in output
    assert "No commands, inference or continuation permission" in output


@pytest.mark.parametrize(
    "change",
    [
        {"policy": "future"},
        {"path": "\x1b[2JAGENTS.md"},
        {"max_bytes": 99999},
        {"content_bytes": True},
        {"content_bytes": -1},
        {"content_bytes": 8193},
        {"content_sha256": "a" * 63},
        {"content_sha256": "\x1b" * 64},
        {"status": ["loaded"]},
        {"status": "disabled"},
        {"issue": "changed"},
    ],
)
def test_unsupported_or_inconsistent_evidence_never_claims_loaded(change, capsys):
    show_repository_instructions({**metadata(), **change})
    output = capsys.readouterr().out
    assert "current_repository_instructions=unknown_or_inconsistent" in output
    assert "current_repository_instructions=loaded" not in output
    assert "\x1b" not in output


def test_absent_current_and_saved_evidence_are_not_disabled(capsys):
    show_repository_instructions(None)
    show_repository_instructions(None, saved=True)
    output = capsys.readouterr().out
    assert "current_repository_instructions=not_checked_or_legacy_response" in output
    assert "saved_repository_instructions=unrecorded" in output
    assert "=disabled" not in output


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_owner_review_after_restart_is_one_request_and_preserves_archive(tmp_path, enabled):
    root = tmp_path / "repo"
    root.mkdir()
    path = root / "AGENTS.md"
    path.write_text("private-instruction-content")
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider, load_root_agents=enabled)) as api:
        sid = api.post("/sessions", json={"user_id": "dev", "workspace_root": str(root)}).json()[
            "session_id"
        ]
        saved = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
    with TestClient(make_app(tmp_path, provider, load_root_agents=enabled)) as api:

        def post(url, **kwargs):
            assert url.endswith(f"/sessions/{sid}/configuration-review")
            assert kwargs["json"] == {"user_id": "dev", "message": "", "provider": None}
            return api.post(url.removeprefix("http://localhost:8000"), json=kwargs["json"])

        with patch("httpx.post", side_effect=post) as posts, patch("httpx.get") as get:
            result = runner.invoke(app, ["configuration-review", sid, "--user-id", "dev"])
            assert result.exit_code == 0, result.output
            assert (
                f"current_repository_instructions={'loaded' if enabled else 'disabled'}"
                in result.output
            )
            if enabled:
                assert hashlib.sha256(path.read_bytes()).hexdigest() in result.output
            assert "private-instruction-content" not in result.output
            assert posts.call_count == 1
            get.assert_not_called()
            with turn_admission.claim(sid, str(root)):
                busy = runner.invoke(app, ["configuration-review", sid, "--user-id", "dev"])
                assert (
                    "current_repository_instructions=not_checked_or_legacy_response" in busy.output
                )
        assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
        assert api.get("/runs", params={"user_id": "dev"}).json() == []
    assert provider.calls == []
    assert path.read_text() == "private-instruction-content"


def test_cli_saved_run_inspection_preserves_historical_instruction_identity(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    path = root / "AGENTS.md"
    path.write_text("Historical convention")
    provider = ScriptedProvider()
    provider.script(make_response(content="Reviewed"))
    with TestClient(make_app(tmp_path, provider, load_root_agents=True)) as api:
        sid = api.post("/sessions", json={"user_id": "dev", "workspace_root": str(root)}).json()[
            "session_id"
        ]
        response = api.post(
            "/chat", json={"user_id": "dev", "session_id": sid, "message": "Inspect"}
        ).json()
        rid = response["run_id"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text("Later user convention")
    with TestClient(make_app(tmp_path, provider, load_root_agents=False)) as api:
        before = api.get(f"/runs/{rid}", params={"user_id": "dev"}).json()

        def get(url, **kwargs):
            return api.get(url.removeprefix("http://localhost:8000"), params=kwargs["params"])

        with patch("httpx.get", side_effect=get) as gets, patch("httpx.post") as post:
            result = runner.invoke(app, ["run", rid, "--user-id", "dev"])
            assert result.exit_code == 0, result.output
            assert "saved_repository_instructions=loaded" in result.output
            assert digest in result.output and "Historical convention" not in result.output
            assert "current_repository_instructions=" not in result.output
            assert gets.call_count == 1
            post.assert_not_called()
        assert api.get(f"/runs/{rid}", params={"user_id": "dev"}).json() == before
    assert len(provider.calls) == 1
    assert path.read_text() == "Later user convention"
