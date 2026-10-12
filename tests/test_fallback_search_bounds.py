from pathlib import Path

import pytest

from tests.test_workspace_search import registry, select_backend
from tools import ToolError


@pytest.mark.parametrize("size", [0, 32, 33])
async def test_fallback_file_limit_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: int
) -> None:
    (tmp_path / "module.py").write_bytes(b"x" * size)
    select_backend("python", monkeypatch)
    monkeypatch.setattr("tools.code.MAX_GREP_FILE_BYTES", 32)
    reg = registry(tmp_path, ())
    if size > 32:
        with pytest.raises(ToolError, match=r"byte limit.*incomplete"):
            await reg.invoke("grep_repo", {"pattern": "x"})
    else:
        hits = await reg.invoke("grep_repo", {"pattern": "x"})
        assert hits == ([{"path": "module.py", "line": 1, "text": "x" * size}] if size else [])


async def test_total_scan_limit_counts_nonmatching_files_and_narrowing_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.py").write_bytes(b"no match" * 4)
    (tmp_path / "b.py").write_bytes(b"needle")
    select_backend("python", monkeypatch)
    monkeypatch.setattr("tools.code.MAX_GREP_SCAN_BYTES", 32)
    reg = registry(tmp_path, ())
    with pytest.raises(ToolError, match=r"byte limit.*incomplete"):
        await reg.invoke("grep_repo", {"pattern": "needle"})
    assert await reg.invoke("grep_repo", {"pattern": "needle", "path": "b.py"}) == [
        {"path": "b.py", "line": 1, "text": "needle"}
    ]


async def test_unreadable_file_does_not_silently_return_no_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "module.py").write_text("needle")
    select_backend("python", monkeypatch)

    def deny_open(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("injected")

    monkeypatch.setattr(Path, "open", deny_open)
    with pytest.raises(ToolError, match=r"could not read.*incomplete"):
        await registry(tmp_path, ()).invoke("grep_repo", {"pattern": "needle"})
