from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

from tools.process import ProcessTimeoutError, command_environment, run_process


def _wait_for_file(path: Path) -> None:
    deadline = time.monotonic() + 0.4
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists(), "child process did not start"


async def test_capture_is_bounded_and_drains_output(tmp_path: Path) -> None:
    result = await run_process(
        [sys.executable, "-c", "import sys; print('x'*100000); print('e'*100000, file=sys.stderr)"],
        cwd=tmp_path,
        timeout_seconds=5,
        output_limit=100,
    )
    assert result.exit_code == 0
    assert result.stdout_truncated and result.stderr_truncated
    assert result.stdout.startswith("x" * 100)
    assert result.stderr.startswith("e" * 100)


@pytest.mark.skipif(sys.platform == "win32", reason="process group termination is POSIX-only")
@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_or_cancellation_kills_child_processes(
    tmp_path: Path,
    cancel: bool,
) -> None:
    started = tmp_path / "child-started"
    marker = tmp_path / "child-survived"
    child = tmp_path / "child.py"
    child.write_text(
        "import pathlib, sys, time\n"
        "pathlib.Path(sys.argv[1]).write_text('started')\n"
        "time.sleep(0.8)\n"
        "pathlib.Path(sys.argv[2]).write_text('bad')\n",
        encoding="utf-8",
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, *sys.argv[1:]])\n"
        "time.sleep(20)\n",
        encoding="utf-8",
    )
    task = asyncio.create_task(
        run_process(
            [sys.executable, str(parent), str(child), str(started), str(marker)],
            cwd=tmp_path,
            timeout_seconds=0.5 if not cancel else 10,
            output_limit=1024,
        )
    )
    await asyncio.to_thread(_wait_for_file, started)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(ProcessTimeoutError, match="timed out"):
            await task
    await asyncio.sleep(0.85)
    assert not marker.exists()


def test_command_environment_requires_explicit_extra_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    assert "ANTHROPIC_API_KEY" not in command_environment()
    assert command_environment({"PROJECT_FLAG": "yes"})["PROJECT_FLAG"] == "yes"


async def test_subprocess_receives_only_selected_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    result = await run_process(
        [
            sys.executable, "-c",
            "import os; print(os.getenv('ANTHROPIC_API_KEY'), os.getenv('PROJECT_FLAG'))",
        ],
        cwd=tmp_path,
        timeout_seconds=5,
        output_limit=100,
        env={"PROJECT_FLAG": "yes"},
    )
    assert result.stdout.strip() == "None yes"
