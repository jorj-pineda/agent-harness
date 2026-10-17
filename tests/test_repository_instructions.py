import hashlib
import os
from dataclasses import replace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from harness.admission import turn_admission
from harness.config import Settings
from harness.configuration_review import compare_profiles
from harness.execution_profile import capture_execution_profile
from harness.repository_instructions import INSTRUCTION_LIMIT_BYTES, load_repository_instructions
from harness.runtime import run_configured_turn
from harness.state import Session
from memory import FactStore
from tests.api.conftest import ScriptedProvider, make_response
from tests.api.test_continuation import SETTINGS, body, known_provider, submit_and_finish
from tests.api.test_session_persistence import make_app
from tools import ToolRegistry


def test_disabled_does_not_touch_filesystem_and_missing_scope_is_explicit(tmp_path):
    with patch("harness.repository_instructions.os.open", side_effect=AssertionError):
        assert (
            load_repository_instructions(str(tmp_path), enabled=False).review.status == "disabled"
        )
        assert load_repository_instructions(None, enabled=True).review.status == "no_workspace"
    assert load_repository_instructions(str(tmp_path), enabled=True).review.status == "missing"
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "AGENTS.md").write_text("Nested ignored")
    assert load_repository_instructions(str(tmp_path), enabled=True).review.status == "missing"


@pytest.mark.parametrize("size", [0, INSTRUCTION_LIMIT_BYTES, INSTRUCTION_LIMIT_BYTES + 1])
def test_complete_utf8_byte_bound_and_hash(tmp_path, size):
    path = tmp_path / "AGENTS.md"
    data = b"x" * size
    path.write_bytes(data)
    result = load_repository_instructions(str(tmp_path), enabled=True)
    if size > INSTRUCTION_LIMIT_BYTES:
        assert result.review.status == "unavailable" and result.review.issue == "too_large"
        assert result.text is None and result.review.content_sha256 is None
    else:
        assert result.text == data.decode()
        assert result.review.content_bytes == size
        assert result.review.content_sha256 == hashlib.sha256(data).hexdigest()
        assert result.review.status == "loaded"
        assert "user requests" in result.prompt_block()


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo", "invalid_utf8", "symlink_root"])
def test_unsafe_inputs_fail_closed_without_content_or_private_errors(tmp_path, kind):
    path = tmp_path / "AGENTS.md"
    if kind == "symlink":
        target = tmp_path / "private-marker"
        target.write_text("private-marker")
        path.symlink_to(target)
    elif kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "invalid_utf8":
        path.write_bytes(b"\xff")
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        tmp_path = alias
    result = load_repository_instructions(str(tmp_path), enabled=True)
    assert result.review.status == "unavailable" and result.text is None
    assert "private-marker" not in result.review.model_dump_json()


def test_replaced_instruction_identity_is_rejected(tmp_path):
    path = tmp_path / "AGENTS.md"
    path.write_text("old")
    real_stat = os.stat

    def changed(name, **kwargs):
        if name == "AGENTS.md":
            path.unlink()
            path.write_text("new")
        return real_stat(name, **kwargs)

    with patch("harness.repository_instructions.os.stat", side_effect=changed):
        result = load_repository_instructions(str(tmp_path), enabled=True)
    assert result.review.issue == "changed" and result.text is None


@pytest.mark.parametrize(
    "enabled,oversize,context",
    [(False, True, 0), (True, False, 0), (True, True, 0), (True, False, 256)],
)
async def test_runtime_injection_and_pre_inference_guards(tmp_path, enabled, oversize, context):
    text = "root convention ñ\n" * (1000 if oversize else 100)
    (tmp_path / "AGENTS.md").write_text(text)
    session = Session(user_id="dev", workspace_root=str(tmp_path))
    original = session.model_dump_json()
    provider = ScriptedProvider()
    provider.script(make_response(content="Reviewed"))
    profiles = []
    with FactStore(tmp_path / "facts.db") as store:
        response = await run_configured_turn(
            settings=Settings(_env_file=None, load_root_agents=enabled, max_context_tokens=context),
            session=session,
            user_id="dev",
            message="Explain the repository",
            provider=provider,
            fact_store=store,
            registry=ToolRegistry(),
            on_execution_profile=profiles.append,
        )
    if enabled and oversize:
        assert response.completion_status == "blocked" and provider.calls == []
        assert session.model_dump_json() == original and profiles == []
    elif context:
        assert response.completion_status == "budget_exhausted" and provider.calls == []
    else:
        assert response.completion_status == "completed" and len(provider.calls) == 1
        prompt = provider.calls[0][0][0].content
        assert ("Repository conventions from workspace-root" in prompt) == enabled
        assert ("root convention" in prompt) == enabled
        assert (profiles[0].repository_instructions is not None) == enabled
    assert (tmp_path / "AGENTS.md").read_text() == text


def test_owner_review_after_restart_includes_instructions_in_context_without_archive_changes(
    tmp_path,
):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "AGENTS.md").write_text("Use bounded checks.\n" * 100)
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider, load_root_agents=True)) as api:
        sid = api.post("/sessions", json={"user_id": "dev", "workspace_root": str(root)}).json()[
            "session_id"
        ]
        saved = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
    with TestClient(make_app(tmp_path, provider, load_root_agents=True)) as api:
        route = f"/sessions/{sid}/configuration-review"
        result = api.post(route, json={"user_id": "dev"}).json()
        assert result["repository_instructions"]["status"] == "loaded"
        assert "Use bounded checks" not in str(result)
        assert result["context"]["estimated_prompt_tokens"] > 500
        assert api.post(route, json={"user_id": "other"}).status_code == 403
        with turn_admission.claim(sid, str(root)):
            assert (
                api.post(route, json={"user_id": "dev"}).json()["repository_instructions"] is None
            )
        (root / "AGENTS.md").write_text("x" * (INSTRUCTION_LIMIT_BYTES + 1))
        unavailable = api.post(route, json={"user_id": "dev"}).json()
        assert unavailable["repository_instructions"]["issue"] == "too_large"
        assert unavailable["context"]["status"] == "not_checked"
        assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
        assert api.get("/runs", params={"user_id": "dev"}).json() == []
    assert provider.calls == []


@pytest.mark.parametrize("change", ["same", "text", "enabled"])
def test_restart_capture_and_continuation_guards(tmp_path, change):
    root = tmp_path / "repo"
    root.mkdir()
    path = root / "AGENTS.md"
    path.write_text("Repository convention")
    settings = {**SETTINGS, "load_root_agents": True}
    provider = known_provider()
    provider.script(make_response(content="Reviewed"))
    with TestClient(make_app(tmp_path, provider, **settings)) as api:
        sid = api.post(
            "/sessions",
            json={"user_id": "dev", "workspace_root": str(root), "checkpoint_turns": True},
        ).json()["session_id"]
        assert (
            api.post(
                "/chat", json={"user_id": "dev", "session_id": sid, "message": "Inspect"}
            ).json()["completion_status"]
            == "completed"
        )
        saved = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        evidence = saved["responses"][0]["execution_profile"]["repository_instructions"]
        assert evidence["content_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    later = known_provider()
    later.script(make_response(content="Reviewed again"))
    if change == "text":
        path.write_text("Later user convention")
    elif change == "enabled":
        settings["load_root_agents"] = False
    with TestClient(make_app(tmp_path, later, **settings)) as api:
        review = api.post(f"/sessions/{sid}/configuration-review", json={"user_id": "dev"}).json()
        assert review["configuration"]["status"] == ("matching" if change == "same" else "changed")
        run = submit_and_finish(api, sid, body(saved))
        assert run["response"]["completion_status"] == (
            "completed" if change == "same" else "blocked"
        ), run
        assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
    assert len(later.calls) == (1 if change == "same" else 0)


def test_legacy_disabled_profile_is_known_but_enabling_changes_it(tmp_path):
    settings = Settings(_env_file=None)
    profile = capture_execution_profile(settings, "ollama", ToolRegistry())
    old = profile.model_dump()
    del old["settings"]["load_root_agents"]
    del old["repository_instructions"]
    legacy = type(profile).model_validate(old)
    assert compare_profiles(legacy, profile).status == "matching"
    enabled = capture_execution_profile(
        Settings(_env_file=None, load_root_agents=True),
        "ollama",
        ToolRegistry(),
        repository_instructions=load_repository_instructions(str(tmp_path), enabled=True).review,
    )
    assert compare_profiles(legacy, enabled).status == "changed"


async def test_disposable_evaluator_uses_same_instructions_and_records_identity(tmp_path):
    import shutil

    from evals.real_run import load_tasks, run_task

    original = load_tasks()[0]
    fixture = tmp_path / "fixture"
    shutil.copytree(original.fixture, fixture)
    (fixture / "AGENTS.md").write_text("Use the visible checks.")
    provider = ScriptedProvider()
    provider.script(make_response(content="No patch"))
    result = await run_task(
        replace(original, fixture=fixture),
        provider=provider,
        settings=Settings(_env_file=None, load_root_agents=True),
        mode="harness",
    )
    assert "Use the visible checks." in provider.calls[0][0][0].content
    assert result.runtime_config["load_root_agents"] is True
    assert result.runtime_config["repository_instructions"]["status"] == "loaded"
    assert not result.passed  # Independent acceptance rejects unchanged fixture.
