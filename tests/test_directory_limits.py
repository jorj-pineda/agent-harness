from pathlib import Path

import pytest

from tests.test_workspace_search import registry
from tools import ToolError


def seed(root: Path, count: int) -> None:
    for number in range(count):
        (root / f"file{number}.py").write_bytes(b"")
    (root / "ignored").mkdir()
    (root / "ignored" / "hidden.py").write_bytes(b"")


@pytest.mark.parametrize("count", [0, 3, 4])
async def test_list_exact_limit_and_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    seed(tmp_path, count)
    monkeypatch.setattr("tools.code.MAX_LIST_ENTRIES", 3)
    reg = registry(tmp_path, ("ignored",))
    if count > 3:
        with pytest.raises(ToolError, match=r"list_dir exceeded.*incomplete"):
            await reg.invoke("list_dir", {})
    else:
        assert await reg.invoke("list_dir", {}) == [
            {"path": f"file{number}.py", "kind": "file"} for number in range(count)
        ]


@pytest.mark.parametrize("count", [0, 2, 3])
async def test_tree_root_counts_toward_exact_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    seed(tmp_path, count)
    monkeypatch.setattr("tools.code.MAX_TREE_ENTRIES", 3)
    reg = registry(tmp_path, ("ignored",))
    if count + 1 > 3:
        with pytest.raises(ToolError, match=r"tree exceeded.*incomplete"):
            await reg.invoke("tree", {"depth": 1})
    else:
        assert await reg.invoke("tree", {"depth": 1}) == [
            {"path": ".", "kind": "dir", "depth": "0"},
            *[
                {"path": f"file{number}.py", "kind": "file", "depth": "1"}
                for number in range(count)
            ],
        ]


async def test_symlink_directories_are_visible_leaves_without_external_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "external.py").write_bytes(b"")
    (root / "external").symlink_to(outside, target_is_directory=True)
    (root / "loop").symlink_to(root, target_is_directory=True)
    (root / "broken").symlink_to("absent")
    original_iterdir = Path.iterdir

    def guarded_iterdir(path: Path):
        assert path == root
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    reg = registry(root, ())
    entries = await reg.invoke("list_dir", {})
    assert entries == [{"path": name, "kind": "symlink"} for name in ["broken", "external", "loop"]]
    rows = await reg.invoke("tree", {"depth": 6})
    assert rows == [
        {"path": ".", "kind": "dir", "depth": "0"},
        *[{**entry, "depth": "1"} for entry in entries],
    ]


async def test_tree_narrowing_recovers_after_global_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ["one", "two"]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "module.py").write_text("user content")
    monkeypatch.setattr("tools.code.MAX_TREE_ENTRIES", 3)
    reg = registry(tmp_path, ())
    with pytest.raises(ToolError, match="results are incomplete"):
        await reg.invoke("tree", {"depth": 2})
    assert await reg.invoke("tree", {"path": "one", "depth": 1}) == [
        {"path": "one", "kind": "dir", "depth": "0"},
        {"path": "one/module.py", "kind": "file", "depth": "1"},
    ]
    assert (tmp_path / "one/module.py").read_text() == "user content"
