"""Real ripgrep/fallback searches enforce literal workspace exclusions."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tools import ToolError, ToolRegistry
from tools.code import register_code_tools
from workspace import Workspace


def registry(root: Path, ignore: tuple[str, ...]) -> ToolRegistry:
    result = ToolRegistry()
    register_code_tools(result, workspace=Workspace(root=root, ignore_globs=ignore))
    return result


def select_backend(backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if backend == "python":
        monkeypatch.setattr("tools.code.shutil.which", lambda _name: None)
    elif not shutil.which("rg"):
        pytest.skip("ripgrep required for native search regression")


@pytest.mark.parametrize("backend", ["rg", "python"])
async def test_search_excludes_ignored_components_even_with_positive_glob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    select_backend(backend, monkeypatch)
    for path in ["visible.py", "generated/hidden.py", "nested/generated/hidden.py", "blocked.py"]:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("needle\n")
    reg = registry(tmp_path, ("generated", "blocked.py"))
    assert await reg.invoke("grep_repo", {"pattern": "needle", "glob": "*.py"}) == [
        {"path": "visible.py", "line": 1, "text": "needle"}
    ]
    with pytest.raises(ToolError, match="ignored"):
        await reg.invoke("grep_repo", {"pattern": "needle", "path": "generated"})


@pytest.mark.parametrize("backend", ["rg", "python"])
async def test_ignore_names_are_literal_and_do_not_hide_similar_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    select_backend(backend, monkeypatch)
    for name in ["cache[1]", "cache1"]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "module.py").write_text("needle\n")
    assert await registry(tmp_path, ("cache[1]",)).invoke("grep_repo", {"pattern": "needle"}) == [
        {"path": "cache1/module.py", "line": 1, "text": "needle"}
    ]


async def test_fallback_does_not_read_outside_symlink_or_walk_ignored_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    select_backend("python", monkeypatch)
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("outside needle\n")
    (root / "external.py").symlink_to(outside)
    (root / "ignored").mkdir()
    (root / "ignored" / "hidden.py").write_text("ignored needle\n")
    (root / "visible.py").write_text("needle\n")
    original_read = Path.read_text

    def guarded_read(path: Path, *args: object, **kwargs: object) -> str:
        assert path != outside
        assert "ignored" not in path.parts
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    assert await registry(root, ("ignored",)).invoke("grep_repo", {"pattern": "needle"}) == [
        {"path": "visible.py", "line": 1, "text": "needle"}
    ]
