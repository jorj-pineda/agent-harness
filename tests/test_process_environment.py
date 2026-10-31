import json
import sys

import pytest

from tools.process import command_environment, run_process


def test_default_environment_keeps_allowlisted_merge_contract(monkeypatch):
    monkeypatch.setenv("PATH", "/ambient/path")
    monkeypatch.setenv("LANG", "ambient-locale")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-copy")
    result = command_environment({"LANG": "override"})
    assert result["PATH"] == "/ambient/path" and result["LANG"] == "override"
    assert "UNRELATED_SECRET" not in result


def test_replacement_requires_explicit_mapping_and_copies_it(monkeypatch):
    monkeypatch.setenv("PATH", "/ambient/path")
    with pytest.raises(ValueError, match="must be explicit"):
        command_environment(inherit_environment=False)
    assert command_environment({}, inherit_environment=False) == {}
    selected = {"LANG": "C"}
    result = command_environment(selected, inherit_environment=False)
    selected["LANG"] = "later"
    assert result == {"LANG": "C"}


async def test_real_child_does_not_receive_ambient_runtime_values(tmp_path, monkeypatch):
    for key in ("PATH", "VIRTUAL_ENV", "LANG", "LC_ALL", "TMPDIR"):
        monkeypatch.setenv(key, "ambient-sentinel")
    result = await run_process(
        [
            sys.executable,
            "-c",
            "import os,json; print(json.dumps({k:os.environ.get(k) for k in ['PATH','VIRTUAL_ENV','LANG','LC_ALL','TMPDIR','EXPLICIT']}))",
        ],
        cwd=tmp_path,
        timeout_seconds=10,
        output_limit=2000,
        env={"EXPLICIT": "selected", "LC_ALL": "C"},
        inherit_environment=False,
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "PATH": None,
        "VIRTUAL_ENV": None,
        "LANG": None,
        "LC_ALL": "C",
        "TMPDIR": None,
        "EXPLICIT": "selected",
    }


async def test_missing_replacement_refuses_before_subprocess_creation(tmp_path, monkeypatch):
    import tools.process as process

    async def forbidden(*args, **kwargs):
        pytest.fail("spawned")

    monkeypatch.setattr(process.asyncio, "create_subprocess_exec", forbidden)
    with pytest.raises(ValueError, match="must be explicit"):
        await run_process(
            [sys.executable],
            cwd=tmp_path,
            timeout_seconds=1,
            output_limit=100,
            inherit_environment=False,
        )
