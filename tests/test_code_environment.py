import json
import os
import sys

import pytest

from harness.runtime import build_registry
from memory import FactStore
from tools import code
from tools.process import ProcessResult
from tools.registry import ToolRegistry
from workspace import Workspace


def registry(root, environment):
    result = ToolRegistry()
    code.register_code_tools(
        result, workspace=Workspace(root=root), command_env=environment, inherit_environment=False
    )
    return result


@pytest.mark.parametrize(
    "environment", [None, {}, {"PATH": ""}, {"PATH": "relative"}, {"PATH": os.pathsep + "/usr/bin"}]
)
def test_replacement_tools_require_explicit_absolute_search_path(tmp_path, environment):
    with pytest.raises(ValueError):
        registry(tmp_path, environment)


async def test_command_uses_copied_selected_path_without_ambient_values(tmp_path, monkeypatch):
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    python = binary_dir / "python"
    python.write_text(
        f'#!{sys.executable}\nimport os,json\nprint(json.dumps({{k:os.environ.get(k) for k in ["LANG","TMPDIR","EXPLICIT"]}}))\n'
    )
    python.chmod(0o700)
    selected = {"PATH": str(binary_dir), "EXPLICIT": "selected", "LC_ALL": "C"}
    store = FactStore(tmp_path / "facts.db")
    reg = build_registry(
        fact_store=store,
        user_id="test",
        workspace_root=str(tmp_path),
        command_env=selected,
        inherit_environment=False,
    )
    selected["PATH"] = "/changed-after-construction"
    monkeypatch.setenv("PATH", "/unavailable")
    monkeypatch.setenv("LANG", "ambient")
    monkeypatch.setenv("TMPDIR", "ambient")
    try:
        result = await reg.invoke("run_command", {"argv": ["python", "-m", "pytest", "-q"]})
    finally:
        store.close()
    assert result["success"]
    assert json.loads(result["stdout"]) == {"LANG": None, "TMPDIR": None, "EXPLICIT": "selected"}


async def test_git_and_ripgrep_receive_same_replacement_policy(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    calls = []
    searches = []
    environment = {"PATH": "/selected/bin", "LC_ALL": "C"}

    async def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return ProcessResult(list(argv), 0, "", "", False, False)

    def which(executable, path=None):
        searches.append((executable, path))
        return "/selected/bin/" + executable

    monkeypatch.setattr(code, "run_process", run)
    monkeypatch.setattr(code.shutil, "which", which)
    reg = registry(tmp_path, environment)
    await reg.invoke("grep_repo", {"pattern": "needle", "path": "."})
    await reg.invoke("git_status", {})
    await reg.invoke("git_diff", {})
    assert len(calls) == 3
    assert all(
        kwargs["env"] == environment and kwargs["inherit_environment"] is False
        for _, kwargs in calls
    )
    assert searches == [("rg", "/selected/bin")]
