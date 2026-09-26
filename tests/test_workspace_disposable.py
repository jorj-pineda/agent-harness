from __future__ import annotations

from pathlib import Path

import pytest

from tools import ToolRegistry
from tools.code import register_code_tools
from workspace import WorkspaceError, disposable_workspace


def test_disposable_workspace_preserves_source_and_tracks_changes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "existing.txt").write_text("user edit", encoding="utf-8")
    (source / "delete.txt").write_text("remove", encoding="utf-8")
    (source / ".env").write_text("API_KEY=secret", encoding="utf-8")
    with disposable_workspace(source) as copy:
        assert copy.changes().added == ()
        assert not (copy.workspace.root / ".env").exists()
        (copy.workspace.root / "existing.txt").write_text("agent edit", encoding="utf-8")
        (copy.workspace.root / "added.txt").write_text("generated", encoding="utf-8")
        (copy.workspace.root / "delete.txt").unlink()
        assert copy.changes().added == ("added.txt",)
        assert copy.changes().modified == ("existing.txt",)
        assert copy.changes().deleted == ("delete.txt",)
        copied_root = copy.workspace.root
    assert (source / "existing.txt").read_text(encoding="utf-8") == "user edit"
    assert (source / "delete.txt").exists()
    assert not (source / "added.txt").exists()
    assert not copied_root.exists()


async def test_disposable_workspace_counts_command_created_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "test_generation.py").write_text(
        "from pathlib import Path\n"
        "def test_generate():\n"
        "    Path('generated.txt').write_text('from pytest')\n",
        encoding="utf-8",
    )
    with disposable_workspace(source) as copy:
        registry = ToolRegistry()
        register_code_tools(
            registry,
            workspace=copy.workspace,
            command_env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        result = await registry.invoke("run_command", {"argv": ["pytest", "-q"]})
        assert result["success"] is True
        assert copy.changes().added == ("generated.txt",)
    assert not (source / "generated.txt").exists()


def test_disposable_workspace_rejects_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "outside").symlink_to(tmp_path)
    with pytest.raises(WorkspaceError, match="Symlinks"), disposable_workspace(source):
        pass
