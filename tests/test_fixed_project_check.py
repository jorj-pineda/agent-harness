from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.config import Settings
from harness.prompts import coding_prompt, coding_prompt_version
from harness.runtime import build_registry, run_configured_turn
from harness.state import Session
from memory import FactStore
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tools import ToolError

CHECK = ["python", "-m", "pytest", "-q", "test_value.py"]


@pytest.mark.parametrize("toolset", ["no_plan_check", "whole_file_check"])
def test_fixed_check_refuses_missing_command_or_required_plan(toolset):
    with pytest.raises(ValidationError, match="requires project_check_argv"):
        Settings(_env_file=None, coding_toolset=toolset)
    with pytest.raises(ValidationError, match="cannot require_plan_before_edit"):
        Settings(
            _env_file=None,
            coding_toolset=toolset,
            project_check_argv=CHECK,
            require_plan_before_edit=True,
        )


async def test_fixed_check_copies_tokens_and_refuses_model_overrides(tmp_path):
    (tmp_path / "test_value.py").write_text("def test_value():\n    assert True\n")
    selected = list(CHECK)
    with FactStore(tmp_path / "facts.db") as store:
        registry = build_registry(
            fact_store=store,
            user_id="dev",
            workspace_root=str(tmp_path),
            coding_toolset="no_plan_check",
            project_check_argv=selected,
        )
        selected[:] = ["bash", "-c", "private"]
        with pytest.raises(ToolError, match="Invalid arguments"):
            await registry.invoke("run_project_check", {"argv": ["python", "--version"]})
        assert not (tmp_path / ".pytest_cache").exists()
        with pytest.raises(ToolError, match="Unknown tool"):
            await registry.invoke("run_command", {"argv": CHECK})
        result = await registry.invoke("run_project_check", {})
        assert result["argv"] == CHECK and result["success"] is True
        names = {spec.name for spec in registry.as_tool_specs()}
        assert "emit_plan" not in names and "run_command" not in names
        assert {"write_file", "replace_text", "run_project_check"} <= names


@pytest.mark.parametrize("toolset", ["no_plan_check", "whole_file_check"])
@pytest.mark.parametrize("python_root", ["python", "python3"])
@pytest.mark.parametrize("check_after_edit", [True, False])
async def test_fixed_check_is_actual_verification_only_after_latest_edit(
    tmp_path: Path, check_after_edit, python_root, toolset
):
    (tmp_path / "value.py").write_text("value = 1\n")
    (tmp_path / "test_value.py").write_text(
        "from value import value\ndef test_value():\n    assert value in (1, 2)\n"
    )
    selected_check = [python_root, *CHECK[1:]]
    settings = Settings(
        _env_file=None,
        coding_prompt_profile="direct",
        coding_toolset=toolset,
        project_check_argv=selected_check,
        require_verification_before_finish=True,
        max_completion_retries=0,
    )
    check = ToolCall(id="check", name="run_project_check", arguments={})
    edit = ToolCall(
        id="edit", name="write_file", arguments={"path": "value.py", "content": "value = 2\n"}
    )
    provider = ScriptedProvider()
    provider.script(
        make_response(tool_calls=[edit, check] if check_after_edit else [check, edit]),
        make_response(content="Changed value.py"),
    )
    with FactStore(tmp_path / "facts.db") as store:
        response = await run_configured_turn(
            settings=settings,
            session=Session(workspace_root=str(tmp_path)),
            user_id="dev",
            message="Set value to 2",
            provider=provider,
            fact_store=store,
            registry=build_registry(
                fact_store=store,
                user_id="dev",
                workspace_root=str(tmp_path),
                coding_toolset=settings.coding_toolset,
                project_check_argv=selected_check,
            ),
        )
    assert (tmp_path / "value.py").read_text() == "value = 2\n"
    assert response.verification_status == ("passed" if check_after_edit else "stale")
    assert response.completion_status == ("completed" if check_after_edit else "incomplete")
    assert "run_project_check with no arguments" in provider.calls[0][0][0].content
    assert "run_command" not in provider.calls[0][0][0].content


@pytest.mark.parametrize("profile", ["legacy", "compact", "direct"])
def test_fixed_check_prompt_identity_is_separate_without_changing_prior_versions(profile):
    assert coding_prompt_version("no_plan_check", profile) != coding_prompt_version(
        "no_plan", profile
    )
    assert "run_command" not in coding_prompt("no_plan_check", profile)


def test_api_uses_shared_fixed_check_and_persists_verification(tmp_path):
    from fastapi.testclient import TestClient

    from tests.api.test_session_persistence import make_app

    root = tmp_path / "repo"
    root.mkdir()
    (root / "test_value.py").write_text("def test_value():\n    assert True\n")
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="write",
                    name="write_file",
                    arguments={"path": "value.py", "content": "value = 2\n"},
                ),
                ToolCall(id="check", name="run_project_check", arguments={}),
            ]
        ),
        make_response(content="Changed value.py"),
    )
    with TestClient(
        make_app(
            tmp_path,
            provider,
            coding_toolset="no_plan_check",
            coding_prompt_profile="direct",
            project_check_argv=list(CHECK),
            require_verification_before_finish=True,
        )
    ) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(root)}).json()[
            "session_id"
        ]
        response = client.post(
            "/chat",
            json={"user_id": "dev", "session_id": sid, "message": "Create value.py and check it"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["completion_status"] == "completed" and data["verification_status"] == "passed"
        assert data["check_attempts"][-1]["argv"] == CHECK
        saved = client.get(f"/runs/{data['run_id']}", params={"user_id": "dev"}).json()
        assert saved["response"]["check_attempts"] == data["check_attempts"]
    assert "run_project_check" in {s.name for s in provider.calls[0][1]}
    assert "run_command" not in {s.name for s in provider.calls[0][1]}


def test_fixed_python3_check_error_is_retained_as_relevant_unavailable_evidence():
    from harness.outcome import harvest_checks
    from harness.state import ToolCallRecord

    selected = ["python3", *CHECK[1:]]
    checks = harvest_checks(
        [ToolCallRecord(name="run_project_check", arguments={}, error="Command unavailable")],
        required_check=selected,
    )
    assert len(checks) == 1 and checks[0].relevant and checks[0].status == "unavailable"


async def test_whole_file_check_omits_fragile_edit_interface(tmp_path):
    with FactStore(tmp_path / "facts.db") as store:
        registry = build_registry(
            fact_store=store,
            user_id="dev",
            workspace_root=str(tmp_path),
            coding_toolset="whole_file_check",
            project_check_argv=CHECK,
        )
        names = {spec.name for spec in registry.as_tool_specs()}
        assert {"write_file", "run_project_check"} <= names
        assert not {"replace_text", "run_command", "emit_plan"} & names
        with pytest.raises(ToolError, match="Unknown tool"):
            await registry.invoke("replace_text", {})
        assert not (tmp_path / "value.py").exists()


@pytest.mark.parametrize("profile", ["legacy", "compact", "direct"])
def test_whole_file_check_prompt_matches_offered_tools(profile):
    prompt = coding_prompt("whole_file_check", profile)
    assert "run_command" not in prompt and "replace_text" not in prompt
    assert "write_file" in prompt
    if profile != "legacy":
        assert "run_project_check" in prompt
    assert coding_prompt_version("whole_file_check", profile) != coding_prompt_version(
        "no_plan_check", profile
    )
