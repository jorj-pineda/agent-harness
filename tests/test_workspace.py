from __future__ import annotations

from pathlib import Path

import pytest

from workspace import Workspace, WorkspaceError

FIXTURE_REPO = Path(__file__).resolve().parent / "fixtures" / "tiny_repo"


def test_workspace_resolves_relative_paths() -> None:
    ws = Workspace(root=FIXTURE_REPO)
    assert ws.resolve("calc.py") == (FIXTURE_REPO / "calc.py").resolve()


def test_workspace_rejects_escape_via_parent_segments() -> None:
    ws = Workspace(root=FIXTURE_REPO)
    with pytest.raises(WorkspaceError, match="escapes workspace"):
        ws.resolve("../../etc/passwd", must_exist=False)


def test_workspace_rejects_absolute_path_outside_root(tmp_path: Path) -> None:
    ws = Workspace(root=FIXTURE_REPO)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="escapes workspace"):
        ws.resolve(str(outside), must_exist=False)


def test_workspace_rejects_missing_path() -> None:
    ws = Workspace(root=FIXTURE_REPO)
    with pytest.raises(WorkspaceError, match="not found"):
        ws.resolve("missing.py")


def test_workspace_rejects_ignored_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    ignored = repo / "node_modules" / "pkg"
    ignored.mkdir(parents=True)
    ws = Workspace(root=repo)
    with pytest.raises(WorkspaceError, match="ignored"):
        ws.resolve("node_modules/pkg", must_exist=True)


def test_relative_str_returns_posix_path() -> None:
    ws = Workspace(root=FIXTURE_REPO)
    assert ws.relative_str(ws.resolve("calc.py")) == "calc.py"
