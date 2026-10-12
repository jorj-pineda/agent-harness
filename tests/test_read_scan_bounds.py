import hashlib
import os
from pathlib import Path

import pytest

from tests.test_workspace_search import registry
from tools import ToolError


@pytest.mark.parametrize("size", [7, 64, 65])
async def test_range_keeps_complete_hash_within_scan_limit_and_rejects_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: int
) -> None:
    raw = b"header\n" + b"x" * (size - 7)
    (tmp_path / "module.py").write_bytes(raw)
    monkeypatch.setattr("tools.code.MAX_READ_SCAN_BYTES", 64)
    reg = registry(tmp_path, ())
    args = {"path": "module.py", "start_line": 1, "end_line": 1}
    if size > 64:
        with pytest.raises(ToolError, match=r"no complete hash.*narrowing"):
            await reg.invoke("read_file", args)
    else:
        result = await reg.invoke("read_file", args)
        assert result["content"] == "header"
        assert result["total_lines"] == (1 if size == 7 else 2)
        assert result["sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("replacement", ["symlink", "fifo"])
async def test_final_component_swap_is_rejected_without_reading_referent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "module.py"
    target.write_bytes(b"original")
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"outside")
    original_open = os.open

    def swapped_open(path: Path, flags: int, *args: object, **kwargs: object) -> int:
        target.unlink()
        if replacement == "symlink":
            target.symlink_to(outside)
        else:
            os.mkfifo(target)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr("tools.code.os.open", swapped_open)
    with pytest.raises(ToolError):
        await registry(root, ()).invoke("read_file", {"path": "module.py"})
    assert outside.read_bytes() == b"outside"
