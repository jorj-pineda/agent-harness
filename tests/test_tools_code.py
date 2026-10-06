from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tools import ToolError, ToolRegistry
from tools.code import build_code_tools, register_code_tools
from workspace import Workspace

FIXTURE_REPO = Path(__file__).resolve().parent / "fixtures" / "tiny_repo"


@pytest.fixture
def workspace() -> Workspace:
    return Workspace(root=FIXTURE_REPO)


@pytest.fixture
def registry(workspace: Workspace) -> ToolRegistry:
    reg = ToolRegistry()
    register_code_tools(reg, workspace=workspace)
    return reg


async def test_read_file_returns_line_range(registry: ToolRegistry) -> None:
    result = await registry.invoke(
        "read_file", {"path": "calc.py", "start_line": 1, "end_line": 10}
    )
    assert result["path"] == "calc.py"
    assert result["start_line"] == 1
    assert "def add" in result["content"]


async def test_read_file_rejects_escape(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError, match="escapes workspace"):
        await registry.invoke("read_file", {"path": "../../../etc/passwd"})


async def test_grep_repo_finds_pattern(registry: ToolRegistry) -> None:
    hits = await registry.invoke("grep_repo", {"pattern": "def add", "path": ".", "glob": "*.py"})
    assert any(h["path"] == "calc.py" and h["line"] >= 1 for h in hits)


async def test_grep_repo_handles_single_file_option_pattern_and_colon_path(tmp_path: Path) -> None:
    if not shutil.which("rg"):
        pytest.skip("ripgrep is required for this regression")
    target = tmp_path / "module:one.py"
    target.write_text("first\n-needle: found\n", encoding="utf-8")
    reg = ToolRegistry()
    register_code_tools(reg, workspace=Workspace(root=tmp_path))
    hits = await reg.invoke("grep_repo", {"pattern": "-needle", "path": "module:one.py"})
    assert hits == [{"path": "module:one.py", "line": 2, "text": "-needle: found"}]


async def test_read_file_large_file_in_bounded_slices_and_hash(tmp_path: Path) -> None:
    target = tmp_path / "large.py"
    target.write_bytes(b"line\n" * 60_000)
    reg = ToolRegistry()
    register_code_tools(reg, workspace=Workspace(root=tmp_path))
    result = await reg.invoke(
        "read_file", {"path": "large.py", "start_line": 59_999, "end_line": 60_000}
    )
    assert result == {
        "path": "large.py",
        "start_line": 59_999,
        "end_line": 60_000,
        "content": "line\nline",
        "total_lines": 60_000,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    with pytest.raises(ToolError, match="Line range exceeds"):
        await reg.invoke("read_file", {"path": "large.py"})


async def test_read_file_empty_and_oversized_selected_line(tmp_path: Path) -> None:
    (tmp_path / "empty.py").write_bytes(b"")
    (tmp_path / "long.py").write_bytes(b"a" * 256_001)
    reg = ToolRegistry()
    register_code_tools(reg, workspace=Workspace(root=tmp_path))
    empty = await reg.invoke("read_file", {"path": "empty.py"})
    assert (empty["content"], empty["total_lines"], empty["end_line"]) == ("", 0, 0)
    with pytest.raises(ToolError, match="Selected range exceeds"):
        await reg.invoke("read_file", {"path": "long.py", "start_line": 1, "end_line": 1})


async def test_replace_text_returns_diff_and_refuses_stale_or_ambiguous_edits(
    tmp_path: Path,
) -> None:
    target = tmp_path / "calc.py"
    target.write_text("a = 1\nb = 1\n", encoding="utf-8")
    reg = ToolRegistry()
    register_code_tools(reg, workspace=Workspace(root=tmp_path))
    read = await reg.invoke("read_file", {"path": "calc.py"})
    with pytest.raises(ToolError, match="found 2"):
        await reg.invoke(
            "replace_text",
            {
                "path": "calc.py",
                "old_text": "= 1",
                "new_text": "= 2",
                "expected_sha256": read["sha256"],
            },
        )
    assert target.read_text(encoding="utf-8") == "a = 1\nb = 1\n"
    result = await reg.invoke(
        "replace_text",
        {
            "path": "calc.py",
            "old_text": "a = 1",
            "new_text": "a = 2",
            "expected_sha256": read["sha256"],
        },
    )
    assert target.read_text(encoding="utf-8") == "a = 2\nb = 1\n"
    assert "-a = 1" in result["diff"] and "+a = 2" in result["diff"]
    assert result["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(ToolError, match="File changed since read_file"):
        await reg.invoke(
            "replace_text",
            {
                "path": "calc.py",
                "old_text": "b = 1",
                "new_text": "b = 2",
                "expected_sha256": read["sha256"],
            },
        )


async def test_replace_text_mismatch_errors_say_how_to_fix_the_call(tmp_path: Path) -> None:
    target = tmp_path / "dedupe.py"
    source = "def dedupe(items):\n    return list(set(items))\n"
    target.write_text(source, encoding="utf-8")
    reg = ToolRegistry()
    register_code_tools(reg, workspace=Workspace(root=tmp_path))
    sha = (await reg.invoke("read_file", {"path": "dedupe.py"}))["sha256"]

    async def replace(old_text: str) -> None:
        await reg.invoke(
            "replace_text",
            {"path": "dedupe.py", "old_text": old_text, "new_text": "x", "expected_sha256": sha},
        )

    with pytest.raises(ToolError, match="backslash and n where the file has line breaks"):
        await replace("def dedupe(items):\\n    return list(set(items))")
    with pytest.raises(ToolError, match="Copy old_text exactly"):
        await replace("def dedupe(values):")
    with pytest.raises(ToolError, match="Copy old_text exactly"):
        await replace("missing\\nlines")
    with pytest.raises(ToolError, match="Include more surrounding lines"):
        await replace("e")
    assert target.read_text(encoding="utf-8") == source


async def test_replace_text_preserves_literal_escapes_and_does_not_misdiagnose_tabs(
    tmp_path: Path,
) -> None:
    target = tmp_path / "escapes.py"
    target.write_text('value = "\\n"\n\tpass\n', encoding="utf-8")
    reg = ToolRegistry()
    register_code_tools(reg, workspace=Workspace(root=tmp_path))
    sha = (await reg.invoke("read_file", {"path": "escapes.py"}))["sha256"]
    with pytest.raises(ToolError, match="Copy old_text exactly"):
        await reg.invoke(
            "replace_text",
            {"path": "escapes.py", "old_text": "\\tpass", "new_text": "x", "expected_sha256": sha},
        )
    await reg.invoke(
        "replace_text",
        {"path": "escapes.py", "old_text": '"\\n"', "new_text": '"x"', "expected_sha256": sha},
    )
    assert target.read_text(encoding="utf-8") == 'value = "x"\n\tpass\n'


async def test_replace_text_rejects_escape_non_utf8_and_noop(tmp_path: Path) -> None:
    target = tmp_path / "binary.py"
    target.write_bytes(b"\xff")
    reg = ToolRegistry()
    register_code_tools(reg, workspace=Workspace(root=tmp_path))
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(ToolError, match="not valid UTF-8"):
        await reg.invoke(
            "replace_text",
            {
                "path": "binary.py",
                "old_text": "x",
                "new_text": "y",
                "expected_sha256": digest,
            },
        )
    with pytest.raises(ToolError, match="escapes workspace"):
        await reg.invoke(
            "replace_text",
            {
                "path": "../escape.py",
                "old_text": "x",
                "new_text": "y",
                "expected_sha256": digest,
            },
        )
    target.write_text("x", encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(ToolError, match="would not change"):
        await reg.invoke(
            "replace_text",
            {
                "path": "binary.py",
                "old_text": "x",
                "new_text": "x",
                "expected_sha256": digest,
            },
        )


async def test_list_dir_lists_fixture_files(registry: ToolRegistry) -> None:
    entries = await registry.invoke("list_dir", {"path": "."})
    names = {e["path"] for e in entries}
    assert "calc.py" in names
    assert "test_calc.py" in names


async def test_tree_respects_depth(registry: ToolRegistry) -> None:
    rows = await registry.invoke("tree", {"path": ".", "depth": 1})
    paths = {r["path"] for r in rows}
    assert "calc.py" in paths
    assert all(int(r["depth"]) <= 1 for r in rows)


async def test_git_status_requires_git_repo(tmp_path: Path) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "file.txt").write_text("x", encoding="utf-8")
    reg = ToolRegistry()
    register_code_tools(reg, workspace=Workspace(root=bare))
    with pytest.raises(ToolError, match="Not a git repository"):
        await reg.invoke("git_status", {})


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "tracked.txt").write_text("hello", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True)
    return repo


async def test_git_status_on_initialized_repo(git_repo: Path) -> None:
    reg = ToolRegistry()
    register_code_tools(reg, workspace=Workspace(root=git_repo))
    result = await reg.invoke("git_status", {})
    assert "tracked.txt" in result["stdout"] or result["stdout"] == ""


async def test_write_file_jailed_to_workspace(workspace: Workspace, registry: ToolRegistry) -> None:
    result = await registry.invoke(
        "write_file",
        {"path": "new_module.py", "content": "VALUE = 1\n"},
    )
    assert result["path"] == "new_module.py"
    assert (workspace.root / "new_module.py").read_text(encoding="utf-8") == "VALUE = 1\n"


async def test_write_file_rejects_escape(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError, match="escapes workspace"):
        await registry.invoke(
            "write_file",
            {"path": "../../../tmp/evil.py", "content": "x"},
        )


async def test_run_command_pytest_on_fixture(workspace: Workspace, registry: ToolRegistry) -> None:
    result = await registry.invoke(
        "run_command",
        {"argv": ["pytest", "test_calc.py", "-q"]},
    )
    assert result["argv"] == ["pytest", "test_calc.py", "-q"]
    assert result["success"] is False  # divide bug in fixture


async def test_run_command_rejects_disallowed_executable(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError, match="not allowlisted"):
        await registry.invoke("run_command", {"argv": ["bash", "-c", "echo hi"]})


@pytest.mark.parametrize("root", ["python", "python3"])
@pytest.mark.parametrize("args", [[], ["-c", "print(1)"], ["script.py"], ["-m", "unittest"]])
async def test_python_commands_retain_pytest_module_restriction(
    registry: ToolRegistry, root: str, args: list[str]
) -> None:
    with pytest.raises(ToolError, match=f"{root} is only allowed as"):
        await registry.invoke("run_command", {"argv": [root, *args]})


async def test_python3_alias_uses_python_environment_and_preserves_requested_argv(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # A different python3 executable must not be selected by the alias.
    (bin_dir / "python3").write_text("not an executable interpreter", encoding="utf-8")
    (bin_dir / "python3").chmod(0o755)
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    reg = ToolRegistry()
    register_code_tools(
        reg,
        workspace=Workspace(root=tmp_path),
        command_env={
            "PATH": os.pathsep.join((str(bin_dir), os.environ.get("PATH", ""))),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
    )
    requested = ["python3", "-m", "pytest", "-q", "test_ok.py"]
    result = await reg.invoke("run_command", {"argv": requested})
    assert result["success"] is True
    assert result["argv"] == ["python", *requested[1:]]
    assert requested[0] == "python3"


def test_build_code_tools_exposes_every_tool(workspace: Workspace) -> None:
    names = {t.name for t in build_code_tools(workspace)}
    assert names == {
        "read_file",
        "grep_repo",
        "list_dir",
        "tree",
        "git_status",
        "git_diff",
        "emit_plan",
        "write_file",
        "replace_text",
        "run_command",
    }


async def test_emit_plan_records_steps_in_tool_trace(registry: ToolRegistry) -> None:
    result = await registry.invoke(
        "emit_plan",
        {
            "steps": ["Read calc.py", "Fix divide", "Run pytest"],
            "summary": "Bugfix divide test",
        },
    )
    assert result["step_count"] == 3
    assert result["steps"] == ["Read calc.py", "Fix divide", "Run pytest"]
    assert result["summary"] == "Bugfix divide test"


async def test_emit_plan_rejects_empty_steps(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError, match="non-empty step"):
        await registry.invoke("emit_plan", {"steps": ["  ", ""]})
