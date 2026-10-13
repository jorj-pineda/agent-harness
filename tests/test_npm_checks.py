import json
import shutil
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from harness.config import Settings
from harness.outcome import is_verification_command
from harness.runtime import build_registry, run_configured_turn
from harness.state import Session
from memory import FactStore
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tools import ToolError, ToolRegistry
from tools.code import register_code_tools
from workspace import Workspace, WorkspaceError
from workspace.npm import npm_test_prefix


@pytest.mark.parametrize(
    "argv,prefix",
    [
        (["npm", "run", "test:run"], "."),
        (
            ["npm", "--prefix", "client", "run", "test:run", "--", "src/lib/format.test.ts"],
            "client",
        ),
        (["npm", "--prefix", ".", "run", "test:run"], "."),
    ],
)
def test_npm_check_shape_is_shared_by_configuration_and_outcomes(argv, prefix):
    assert npm_test_prefix(argv) == prefix
    assert is_verification_command(argv)
    assert Settings(_env_file=None, project_check_argv=argv).project_check_argv == argv


@pytest.mark.parametrize(
    "argv",
    [
        ["npm", "install"],
        ["npx", "vitest", "run"],
        ["npm", "run", "start"],
        ["npm", "run", "test"],
        ["npm", "--prefix", "../outside", "run", "test:run"],
        ["npm", "--prefix", "/tmp", "run", "test:run"],
        ["npm", "--prefix", "client/../server", "run", "test:run"],
        ["npm", "--prefix", "client//nested", "run", "test:run"],
        ["npm", "--prefix", "C:\\external", "run", "test:run"],
        ["npm", "run", "test:run", "--prefix", "outside"],
        ["npm", "run", "test:run", "--", "--watch=true"],
        ["npm", "run", "test:run", "--", "--list"],
        ["npm", "run", "test:run", "--", "--passWithNoTests"],
        ["npm", "run", "test:run", "--", "--help"],
    ],
)
async def test_unapproved_npm_shapes_never_dispatch_or_count_as_checks(tmp_path, monkeypatch, argv):
    assert npm_test_prefix(argv) is None
    assert not is_verification_command(argv)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, project_check_argv=argv)
    registry = ToolRegistry()
    register_code_tools(registry, workspace=Workspace(tmp_path))
    dispatch = AsyncMock()
    monkeypatch.setattr("tools.code.run_process", dispatch)
    with pytest.raises(ToolError):
        await registry.invoke("run_command", {"argv": argv})
    dispatch.assert_not_called()


@pytest.mark.parametrize("prefix", ["linked", "node_modules", "missing", "file"])
async def test_npm_prefix_is_checked_against_actual_workspace_before_dispatch(
    tmp_path, monkeypatch, prefix
):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    (root / "node_modules").mkdir()
    (root / "file").write_text("not a directory")
    registry = ToolRegistry()
    register_code_tools(registry, workspace=Workspace(root))
    dispatch = AsyncMock()
    monkeypatch.setattr("tools.code.run_process", dispatch)
    with pytest.raises((ToolError, WorkspaceError)):
        await registry.invoke(
            "run_command", {"argv": ["npm", "--prefix", prefix, "run", "test:run"]}
        )
    dispatch.assert_not_called()


@pytest.mark.skipif(
    shutil.which("npm") is None or shutil.which("node") is None, reason="Node/npm not installed"
)
async def test_real_npm_script_failure_then_edit_and_exact_check_through_shared_runtime(tmp_path):
    root = tmp_path / "repo"
    package = root / "client"
    package.mkdir(parents=True)
    (root / "user.txt").write_text("pre-existing work\n")
    (package / "package.json").write_text(json.dumps({"scripts": {"test:run": "node check.cjs"}}))
    (package / "check.cjs").write_text(
        "const assert = require('node:assert/strict'); assert.equal(require('./value.cjs'), 2);\n"
    )
    (package / "value.cjs").write_text("module.exports = 1;\n")
    argv = ["npm", "--prefix", "client", "run", "test:run"]
    settings = Settings(_env_file=None, project_check_argv=argv)
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[ToolCall(id="first", name="run_command", arguments={"argv": argv})]
        ),
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "client/value.cjs", "content": "module.exports = 2;\n"},
                )
            ]
        ),
        make_response(
            tool_calls=[ToolCall(id="check", name="run_command", arguments={"argv": argv})]
        ),
        make_response(content="Done"),
    )
    with FactStore(tmp_path / "memory.db") as store:
        registry = build_registry(fact_store=store, user_id="dev", workspace_root=str(root))
        response = await run_configured_turn(
            settings=settings,
            session=Session(workspace_root=str(root)),
            user_id="dev",
            message="Fix the value",
            provider=provider,
            fact_store=store,
            registry=registry,
        )
    assert response.completion_status == "completed"
    assert response.verification_status == "passed"
    assert [check.exit_code for check in response.check_attempts] == [1, 0]
    assert json.dumps(argv) in provider.calls[0][0][0].content
    assert (root / "user.txt").read_text() == "pre-existing work\n"
    assert (package / "value.cjs").read_text() == "module.exports = 2;\n"
