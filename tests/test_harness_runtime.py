from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.config import Settings
from harness.loop import MAX_ITERATIONS_STUB
from harness.prompts import BASE_SYSTEM_PROMPT
from harness.runtime import build_registry, run_configured_turn
from harness.state import Session
from memory import FactStore
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response


def test_project_check_settings_accept_json_env_and_reject_non_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROJECT_CHECK_ARGV", '["pytest", "-q"]')
    assert Settings(_env_file=None).project_check_argv == ["pytest", "-q"]
    monkeypatch.delenv("PROJECT_CHECK_ARGV")
    for argv in ([], ["pytest", "--version"], ["bash", "-c", "pytest"]):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, project_check_argv=argv)


async def test_configured_check_is_shown_in_shared_system_prompt(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, project_check_argv=["pytest", "-q", "test_calc.py"])
    provider = ScriptedProvider()
    provider.script(make_response(content="hello"))
    with FactStore(tmp_path / "memory.db") as store:
        await run_configured_turn(
            settings=settings,
            session=Session(workspace_root=str(tmp_path)),
            user_id="dev",
            message="Inspect the code",
            provider=provider,
            fact_store=store,
            registry=build_registry(fact_store=store, user_id="dev", workspace_root=str(tmp_path)),
        )
    assert '["pytest", "-q", "test_calc.py"]' in provider.calls[0][0][0].content


async def test_runtime_refreshes_memory_without_duplicating_system_message(tmp_path: Path) -> None:
    settings = Settings(_env_file=None)
    session = Session(user_id="dev", workspace_root=str(tmp_path))
    provider = ScriptedProvider()
    provider.script(make_response(content="first"), make_response(content="second"))
    with FactStore(tmp_path / "memory.db") as store:
        registry = build_registry(fact_store=store, user_id="dev", workspace_root=str(tmp_path))
        for message in ("Inspect the code", "Continue"):
            await run_configured_turn(
                settings=settings,
                session=session,
                user_id="dev",
                message=message,
                provider=provider,
                fact_store=store,
                registry=registry,
            )
            store.add("dev", "Use focused changes")
    first, second = [call[0] for call in provider.calls]
    assert first[0].content.startswith(BASE_SYSTEM_PROMPT)
    assert "Use focused changes" not in first[0].content
    assert "Use focused changes" in second[0].content
    assert f"Workspace root: {tmp_path}" in second[0].content
    assert sum(message.role == "system" for message in second) == 1
    assert [message.role for message in second] == ["system", "user", "assistant", "user"]


@pytest.mark.parametrize(
    "policy",
    [
        {"require_plan_before_edit": True},
        {"require_verification_before_finish": True},
        {"max_files_touched_per_turn": 1},
    ],
)
async def test_runtime_applies_configured_policy(tmp_path: Path, policy: dict) -> None:
    settings = Settings(_env_file=None, **policy)
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(id="a", name="write_file", arguments={"path": "a.py", "content": "a=1"}),
                ToolCall(id="b", name="write_file", arguments={"path": "b.py", "content": "b=2"}),
            ]
        ),
        make_response(content="Done"),
        make_response(content="Still done"),
    )
    with FactStore(tmp_path / "memory.db") as store:
        response = await run_configured_turn(
            settings=settings,
            session=Session(workspace_root=str(tmp_path)),
            user_id="dev",
            message="Add two modules",
            provider=provider,
            fact_store=store,
            registry=build_registry(fact_store=store, user_id="dev", workspace_root=str(tmp_path)),
        )
    assert response.escalated is True
    assert response.completion_status == "incomplete"
    if policy.get("require_plan_before_edit"):
        assert response.files_touched == []
        assert not (tmp_path / "a.py").exists()
        assert not (tmp_path / "b.py").exists()
    elif policy.get("max_files_touched_per_turn"):
        assert response.files_touched == ["a.py"]
        assert (tmp_path / "a.py").exists()
        assert not (tmp_path / "b.py").exists()
    else:
        assert response.files_touched == ["a.py", "b.py"]
        assert response.verification_status == "not_run"


async def test_runtime_uses_iteration_budget_and_scope_policy(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    provider.script(make_response(tool_calls=[ToolCall(id="a", name="recall", arguments={})]))
    with FactStore(tmp_path / "memory.db") as store:
        registry = build_registry(fact_store=store, user_id="dev", workspace_root=None)
        kwargs = {
            "settings": Settings(_env_file=None, max_tool_iterations=1),
            "session": Session(),
            "user_id": "dev",
            "provider": provider,
            "fact_store": store,
            "registry": registry,
        }
        refused = await run_configured_turn(message="delete .git", **kwargs)
        assert refused.provider == "policy"
        assert refused.completion_status == "blocked"
        assert provider.calls == []
        response = await run_configured_turn(message="Recall my preferences", **kwargs)
    assert response.answer == MAX_ITERATIONS_STUB
    assert len(provider.calls) == 1


async def test_runtime_applies_output_budget_when_provider_omits_usage(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "a.py", "content": "a=1"},
                )
            ]
        )
    )
    with FactStore(tmp_path / "memory.db") as store:
        response = await run_configured_turn(
            settings=Settings(_env_file=None, max_completion_tokens_per_turn=5),
            session=Session(workspace_root=str(tmp_path)),
            user_id="dev",
            message="Edit a.py",
            provider=provider,
            fact_store=store,
            registry=build_registry(fact_store=store, user_id="dev", workspace_root=str(tmp_path)),
        )
    assert response.completion_status == "incomplete"
    assert "did not report" in (response.completion_reason or "")
    assert not (tmp_path / "a.py").exists()
