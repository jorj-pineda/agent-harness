"""Real ripgrep/fallback searches enforce literal workspace exclusions."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tools import ToolError, ToolRegistry
from tools.code import MAX_GREP_HITS, register_code_tools
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
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object) -> object:
        assert path != outside
        assert "ignored" not in path.parts
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    assert await registry(root, ("ignored",)).invoke("grep_repo", {"pattern": "needle"}) == [
        {"path": "visible.py", "line": 1, "text": "needle"}
    ]


@pytest.mark.parametrize("backend", ["rg", "python"])
@pytest.mark.parametrize("count", [0, MAX_GREP_HITS, MAX_GREP_HITS + 1])
async def test_exact_limit_returns_hits_but_overflow_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, count: int
) -> None:
    select_backend(backend, monkeypatch)
    (tmp_path / "module.py").write_text("needle\n" * count)
    reg = registry(tmp_path, ())
    if count > MAX_GREP_HITS:
        with pytest.raises(ToolError, match=r"results are incomplete.*Narrow"):
            await reg.invoke("grep_repo", {"pattern": "needle", "path": "module.py"})
    else:
        hits = await reg.invoke("grep_repo", {"pattern": "needle", "path": "module.py"})
        assert hits == [
            {"path": "module.py", "line": line, "text": "needle"} for line in range(1, count + 1)
        ]


@pytest.mark.parametrize("backend", ["rg", "python"])
async def test_limit_is_global_and_ignored_matches_do_not_consume_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    select_backend(backend, monkeypatch)
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "noise.py").write_text("needle\n" * (MAX_GREP_HITS + 1))
    for name in ["first.py", "second.py"]:
        (tmp_path / name).write_text("needle\n" * (MAX_GREP_HITS // 2))
    reg = registry(tmp_path, ("ignored",))
    hits = await reg.invoke("grep_repo", {"pattern": "needle"})
    assert len(hits) == MAX_GREP_HITS
    assert {hit["path"] for hit in hits} == {"first.py", "second.py"}
    (tmp_path / "third.py").write_text("needle\n")
    with pytest.raises(ToolError, match=r"results are incomplete.*Narrow"):
        await reg.invoke("grep_repo", {"pattern": "needle"})
    narrowed = await reg.invoke("grep_repo", {"pattern": "needle", "path": "third.py"})
    assert narrowed == [{"path": "third.py", "line": 1, "text": "needle"}]
