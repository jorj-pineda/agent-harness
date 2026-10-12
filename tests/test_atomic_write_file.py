"""Real-file regressions for whole-file publication and failure preservation."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest

from tools import ToolError, ToolRegistry
from tools.code import MAX_WRITE_BYTES, register_code_tools
from workspace import Workspace


def entries(root: Path) -> list[Path]:
    return list(root.iterdir())


def registry(root: Path) -> ToolRegistry:
    result = ToolRegistry()
    register_code_tools(result, workspace=Workspace(root=root))
    return result


@pytest.mark.parametrize("mode", [0o640, 0o755])
async def test_atomic_write_preserves_existing_permissions_and_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: int
) -> None:
    target = tmp_path / "script.py"
    target.write_bytes(b"user's original\n")
    target.chmod(mode)
    replace = os.replace
    observed = []

    def publish(source: Path, destination: Path) -> None:
        assert destination == target
        assert source.parent == target.parent
        assert target.read_bytes() == b"user's original\n"
        assert source.read_bytes() == "new café\n".encode()
        assert stat.S_IMODE(source.stat().st_mode) == mode
        observed.append(True)
        replace(source, destination)

    monkeypatch.setattr("tools.code.os.replace", publish)
    result = await registry(tmp_path).invoke(
        "write_file", {"path": "script.py", "content": "new café\n"}
    )
    assert result == {"path": "script.py", "bytes_written": len("new café\n".encode())}
    assert observed == [True]
    assert target.read_text() == "new café\n"
    assert stat.S_IMODE(target.stat().st_mode) == mode
    assert entries(tmp_path) == [target]


async def test_new_nested_file_retains_normal_creation_permissions(tmp_path: Path) -> None:
    control = tmp_path / "control.py"
    control.write_bytes(b"control")
    result = await registry(tmp_path).invoke(
        "write_file", {"path": "nested/empty.py", "content": ""}
    )
    target = tmp_path / "nested/empty.py"
    assert result == {"path": "nested/empty.py", "bytes_written": 0}
    assert target.read_bytes() == b""
    assert stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(control.stat().st_mode)
    assert entries(target.parent) == [target]


async def test_atomic_replacement_preserves_hard_link_alias_content(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    alias = tmp_path / "alias.py"
    target.write_bytes(b"original")
    os.link(target, alias)
    await registry(tmp_path).invoke("write_file", {"path": "module.py", "content": "new"})
    assert target.read_bytes() == b"new"
    assert alias.read_bytes() == b"original"
    assert target.stat().st_ino != alias.stat().st_ino


async def test_in_workspace_symlink_still_resolves_to_its_target(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    alias = tmp_path / "alias.py"
    target.write_bytes(b"original")
    alias.symlink_to(target.name)
    result = await registry(tmp_path).invoke("write_file", {"path": "alias.py", "content": "new"})
    assert result == {"path": "module.py", "bytes_written": 3}
    assert alias.is_symlink()
    assert target.read_bytes() == alias.read_bytes() == b"new"


@pytest.mark.parametrize("existing", [True, False])
@pytest.mark.parametrize("phase", ["partial_write", "rename"])
async def test_failed_write_preserves_destination_and_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool, phase: str
) -> None:
    target = tmp_path / "module.py"
    if existing:
        target.write_bytes(b"pre-existing user edit\n")
        target.chmod(0o640)

    if phase == "rename":

        def fail_replace(source: Path, destination: Path) -> None:
            assert source.read_bytes() == b"replacement\n"
            raise OSError("injected rename failure")

        monkeypatch.setattr("tools.code.os.replace", fail_replace)
    else:
        fdopen = os.fdopen

        class FailingWriter:
            def __init__(self, descriptor: int, mode: str) -> None:
                self.stream = fdopen(descriptor, mode)

            def __enter__(self) -> FailingWriter:
                return self

            def __exit__(self, *_args: Any) -> None:
                self.stream.close()

            def write(self, content: bytes) -> None:
                self.stream.write(content[:3])
                self.stream.flush()
                raise OSError("injected partial write failure")

        monkeypatch.setattr("tools.code.os.fdopen", FailingWriter)

    with pytest.raises(ToolError, match="injected"):
        await registry(tmp_path).invoke(
            "write_file", {"path": "module.py", "content": "replacement\n"}
        )
    if existing:
        assert target.read_bytes() == b"pre-existing user edit\n"
        assert stat.S_IMODE(target.stat().st_mode) == 0o640
        assert entries(tmp_path) == [target]
    else:
        assert entries(tmp_path) == []


async def test_permission_copy_failure_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "module.py"
    target.write_bytes(b"user edit")

    def fail_chmod(_descriptor: int, _mode: int) -> None:
        raise OSError("injected chmod failure")

    monkeypatch.setattr("tools.code.os.fchmod", fail_chmod)
    with pytest.raises(ToolError, match="injected chmod"):
        await registry(tmp_path).invoke("write_file", {"path": "module.py", "content": "new"})
    assert target.read_bytes() == b"user edit"
    assert entries(tmp_path) == [target]


async def test_oversized_content_does_not_create_parent(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="Content exceeds"):
        await registry(tmp_path).invoke(
            "write_file", {"path": "new/module.py", "content": "x" * (MAX_WRITE_BYTES + 1)}
        )
    assert entries(tmp_path) == []


async def test_special_file_is_rejected_without_opening_it(tmp_path: Path) -> None:
    target = tmp_path / "pipe"
    os.mkfifo(target)
    with pytest.raises(ToolError, match="non-regular file"):
        await registry(tmp_path).invoke("write_file", {"path": "pipe", "content": "new"})
    assert stat.S_ISFIFO(target.stat().st_mode)
    assert entries(tmp_path) == [target]
