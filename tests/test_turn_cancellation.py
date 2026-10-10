from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from pydantic import BaseModel

from harness.cancellation import TurnCancellation
from harness.state import Session
from memory import FactStore
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tests.test_turn_admission import HoldingProvider, turn
from tools import Tool, ToolRegistry


async def test_cancel_before_model_request(tmp_path: Path) -> None:
    signal = TurnCancellation()
    signal.request()
    provider = ScriptedProvider()
    session = Session()
    with FactStore(tmp_path / "memory.db") as store:
        result = await turn(session, store, provider, cancellation=signal)
    assert result.completion_status == "cancelled"
    assert provider.calls == []
    assert session.turns[-1].finished_at is not None
    assert signal.closed
    assert not signal.request()


async def test_cancel_model_request_stops_inference(tmp_path: Path) -> None:
    signal = TurnCancellation()
    provider = HoldingProvider()
    session = Session()
    with FactStore(tmp_path / "memory.db") as store:
        active = asyncio.create_task(turn(session, store, provider, cancellation=signal))
        await asyncio.wait_for(provider.started.wait(), 2)
        assert signal.request()
        result = await asyncio.wait_for(active, 2)
    assert result.completion_status == "cancelled"
    assert provider.calls == []
    assert result.token_usage.completion_tokens is None


async def test_cancel_sync_worker_holds_admission_and_blocks_remaining_batch(
    tmp_path: Path,
) -> None:
    started, release = threading.Event(), threading.Event()
    signal = TurnCancellation()

    class Args(BaseModel):
        path: str
        content: str

    def write(args: Args):
        started.set()
        assert release.wait(3)
        (tmp_path / args.path).write_text(args.content)
        return {"path": args.path, "bytes_written": len(args.content)}

    class CommandArgs(BaseModel):
        argv: list[str]

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="run_command",
            description="scripted passing check",
            input_model=CommandArgs,
            fn=lambda _: {"success": True, "exit_code": 0},
        )
    )
    registry.register(Tool(name="write_file", description="slow write", input_model=Args, fn=write))
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(id="check", name="run_command", arguments={"argv": ["pytest"]}),
                ToolCall(
                    id="first", name="write_file", arguments={"path": "a.py", "content": "a = 1\n"}
                ),
                ToolCall(
                    id="second", name="write_file", arguments={"path": "b.py", "content": "b = 1\n"}
                ),
            ]
        )
    )
    session = Session(workspace_root=str(tmp_path))
    with FactStore(tmp_path / "memory.db") as store:
        active = asyncio.create_task(
            turn(session, store, provider, registry=registry, cancellation=signal)
        )
        try:
            assert await asyncio.to_thread(started.wait, 2)
            signal.request()
            await asyncio.sleep(0.02)
            assert signal.request()  # repeated requests do not abandon the worker
            assert not active.done()
            other = ScriptedProvider()
            blocked = await turn(Session(workspace_root=str(tmp_path)), store, other)
            assert blocked.completion_status == "blocked"
            assert other.calls == []
        finally:
            release.set()
        result = await asyncio.wait_for(active, 2)
    assert result.completion_status == "cancelled"
    assert result.workspace_changes.added == ["a.py"]
    assert not (tmp_path / "b.py").exists()
    assert len(result.tool_calls) == 3
    assert result.verification_status == "stale"
    assert result.check_attempts[0].superseded_by_edit
    assert result.tool_calls[1].interrupted
    assert "result unavailable" in result.tool_calls[1].error
    assert "not executed" in result.tool_calls[2].error
    assert [m.tool_call_id for m in session.messages if m.role == "tool"] == [
        "check",
        "first",
        "second",
    ]


async def test_cancel_real_command_kills_descendant_before_final_review(tmp_path: Path) -> None:
    import sys

    import pytest

    if sys.platform == "win32":
        pytest.skip("Process-group cleanup is POSIX-only")
    # Real allowlisted pytest starts a child that would otherwise mutate later.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_wait.py").write_text(
        "import subprocess, sys, time\n"
        "def test_wait():\n"
        "    subprocess.Popen([sys.executable, '-c', \"from pathlib import Path; import time; Path('started').write_text('yes'); time.sleep(1); Path('late').write_text('bad')\"])\n"
        "    time.sleep(20)\n"
    )
    signal = TurnCancellation()
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="check",
                    name="run_command",
                    arguments={"argv": ["python", "-m", "pytest", "test_wait.py", "-q"]},
                ),
                ToolCall(
                    id="late", name="write_file", arguments={"path": "b.py", "content": "bad = 1\n"}
                ),
            ]
        )
    )
    with FactStore(tmp_path / "memory.db") as store:
        active = asyncio.create_task(
            turn(Session(workspace_root=str(repo)), store, provider, cancellation=signal)
        )
        try:
            async with asyncio.timeout(5):
                while not (repo / "started").exists():  # noqa: ASYNC110 — child signals via file
                    await asyncio.sleep(0.01)
            signal.request()
            result = await asyncio.wait_for(active, 3)
        finally:
            if not active.done():
                active.cancel()
                await asyncio.gather(active, return_exceptions=True)
    assert result.completion_status == "cancelled"
    assert result.tool_calls[0].interrupted
    assert result.check_attempts[0].status == "unavailable"
    assert result.workspace_changes.added == ["started"]
    assert not (repo / "b.py").exists()
    await asyncio.sleep(1.1)
    assert not (repo / "late").exists()
