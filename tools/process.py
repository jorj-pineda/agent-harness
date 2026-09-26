"""Bounded asynchronous subprocess execution with process-tree cleanup."""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path


class ProcessTimeoutError(TimeoutError):
    """A subprocess and its output readers did not finish before the deadline."""


@dataclass(frozen=True)
class ProcessResult:
    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool


def command_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Pass a small runtime environment, plus explicitly configured additions."""
    keys = ("PATH", "VIRTUAL_ENV", "LANG", "LC_ALL", "SYSTEMROOT", "TMPDIR")
    env = {key: os.environ[key] for key in keys if key in os.environ}
    if extra:
        env.update(extra)
    return env


async def _read_limited(stream: asyncio.StreamReader, limit: int) -> tuple[str, bool]:
    captured = bytearray()
    truncated = False
    while chunk := await stream.read(8192):
        available = max(0, limit - len(captured))
        captured.extend(chunk[:available])
        truncated |= len(chunk) > available
    text = captured.decode("utf-8", errors="replace")
    if truncated:
        text += "\n...(truncated)"
    return text, truncated


def _signal_tree(process: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        elif process.returncode is None:
            process.kill()
    except ProcessLookupError:
        pass


async def _stop_tree(process: asyncio.subprocess.Process) -> None:
    _signal_tree(process, signal.SIGTERM)
    with suppress(TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=0.2)
    # Descendants may keep pipes open after their parent exits.
    _signal_tree(process, signal.SIGKILL)
    await process.wait()


async def run_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    output_limit: int,
    env: Mapping[str, str] | None = None,
) -> ProcessResult:
    """Run argv without a shell; terminate its process group on timeout/cancel."""
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        env=command_environment(env),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=os.name == "posix",
    )
    assert process.stdout is not None and process.stderr is not None
    stdout_task = asyncio.create_task(_read_limited(process.stdout, output_limit))
    stderr_task = asyncio.create_task(_read_limited(process.stderr, output_limit))
    try:
        async with asyncio.timeout(timeout_seconds):
            exit_code, stdout_result, stderr_result = await asyncio.gather(
                process.wait(), stdout_task, stderr_task
            )
    except (TimeoutError, asyncio.CancelledError) as exc:
        await _stop_tree(process)
        for task in (stdout_task, stderr_task):
            task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        if isinstance(exc, TimeoutError):
            raise ProcessTimeoutError(f"Command timed out after {timeout_seconds:g}s") from exc
        raise
    stdout, stdout_truncated = stdout_result
    stderr, stderr_truncated = stderr_result
    return ProcessResult(
        argv=list(argv),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )
