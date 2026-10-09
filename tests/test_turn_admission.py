from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from pydantic import BaseModel

from harness.admission import TurnAdmission
from harness.config import Settings
from harness.runtime import build_registry, run_configured_turn
from harness.state import Session
from memory import FactStore
from providers.base import ChatMessage, ProviderResponse, ToolCall, ToolSpec
from tests.api.conftest import ScriptedProvider, make_response
from tools import Tool, ToolRegistry


class HoldingProvider(ScriptedProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        self.started.set()
        await self.release.wait()
        return await super().chat(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )


async def turn(session: Session, store: FactStore, provider: ScriptedProvider, **kwargs):
    return await run_configured_turn(
        settings=kwargs.pop("settings", Settings(_env_file=None)),
        session=session,
        user_id="dev",
        message="Update the module",
        provider=provider,
        fact_store=store,
        registry=kwargs.pop("registry", None)
        or build_registry(
            fact_store=store,
            user_id="dev",
            workspace_root=session.workspace_root,
        ),
        **kwargs,
    )


@pytest.mark.parametrize("root_mode", ["same", "child", "parent", "symlink", "tracking_off"])
async def test_conflict_is_rejected_before_context_inference_or_mutation(
    tmp_path: Path,
    root_mode: str,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    child = root / "child"
    child.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root)
    first_root = child if root_mode == "parent" else root
    second_root = {"child": child, "symlink": alias}.get(root_mode, root)
    first_session = Session(workspace_root=str(first_root))
    second_session = Session(workspace_root=str(second_root))
    first = HoldingProvider()
    first.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "first.py", "content": "first = 1\n"},
                )
            ]
        ),
        make_response(content="Finished"),
    )
    second = ScriptedProvider()
    second.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "second.py", "content": "second = 2\n"},
                )
            ]
        ),
        make_response(content="Finished"),
    )
    settings = Settings(_env_file=None, track_workspace_changes=root_mode != "tracking_off")
    with FactStore(tmp_path / "memory.db") as store:
        active = asyncio.create_task(turn(first_session, store, first, settings=settings))
        try:
            await asyncio.wait_for(first.started.wait(), 2)
            trace = []
            events = []

            async def event(item):
                events.append(item)

            blocked = await turn(
                second_session, store, second, trace=trace, on_event=event, settings=settings
            )
            assert blocked.completion_status == "blocked"
            assert "workspace overlaps" in blocked.answer
            assert blocked.tool_calls == [] and blocked.workspace_changes.status == "not_tracked"
            assert (
                second.calls == [] and second_session.messages == [] and second_session.turns == []
            )
            assert trace == events == []
            assert not (second_root / "second.py").exists()
        finally:
            first.release.set()
            await active
        accepted = await turn(second_session, store, second, settings=settings)
    assert accepted.completion_status == "completed"
    assert (first_root / "first.py").exists() and (second_root / "second.py").exists()
    if root_mode != "tracking_off":
        assert accepted.workspace_changes.added == ["second.py"]
        assert all(d.path != "first.py" for d in accepted.workspace_changes.diffs)


async def test_no_workspace_session_is_still_guarded(tmp_path: Path) -> None:
    session = Session()
    first = HoldingProvider()
    first.script(make_response(content="first"))
    second = ScriptedProvider()
    second.script(make_response(content="second"))
    with FactStore(tmp_path / "memory.db") as store:
        active = asyncio.create_task(turn(session, store, first))
        try:
            await asyncio.wait_for(first.started.wait(), 2)
            messages = [m.model_copy(deep=True) for m in session.messages]
            blocked = await turn(session, store, second)
            assert "session already" in blocked.answer
            assert session.messages == messages and len(session.turns) == 1
            assert second.calls == []
        finally:
            first.release.set()
            await active
        await turn(session, store, second)
    assert len(session.turns) == 2
    assert [m.content for m in session.messages if m.role == "assistant"] == ["first", "second"]


async def test_separate_workspaces_can_progress_concurrently(tmp_path: Path) -> None:
    roots = [tmp_path / "repo", tmp_path / "repo-other"]
    for root in roots:
        root.mkdir()
    providers = [HoldingProvider(), HoldingProvider()]
    for provider in providers:
        provider.script(make_response(content="done"))
    with FactStore(tmp_path / "memory.db") as store:
        tasks = [
            asyncio.create_task(turn(Session(workspace_root=str(root)), store, provider))
            for root, provider in zip(roots, providers, strict=True)
        ]
        try:
            await asyncio.wait_for(asyncio.gather(*(p.started.wait() for p in providers)), 2)
        finally:
            for provider in providers:
                provider.release.set()
            results = await asyncio.gather(*tasks)
    assert all(r.completion_status == "completed" for r in results)


@pytest.mark.parametrize("stop", ["cancel", "failure"])
async def test_reservation_released_after_provider_exit(tmp_path: Path, stop: str) -> None:
    session = Session()
    provider = HoldingProvider()
    # No scripted response causes an exception after the hold is released.
    with FactStore(tmp_path / "memory.db") as store:
        active = asyncio.create_task(turn(session, store, provider))
        await asyncio.wait_for(provider.started.wait(), 2)
        if stop == "cancel":
            active.cancel()
            with pytest.raises(asyncio.CancelledError):
                await active
        else:
            provider.release.set()
            with pytest.raises(AssertionError, match="ran out"):
                await active
        retry = ScriptedProvider()
        retry.script(make_response(content="retry"))
        assert (await turn(session, store, retry)).completion_status == "completed"


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
async def test_reservation_stays_held_until_sync_write_worker_finishes(
    tmp_path: Path,
    stop: str,
) -> None:
    started, release = threading.Event(), threading.Event()

    class Args(BaseModel):
        path: str
        content: str

    def slow_write(args: Args):
        started.set()
        assert release.wait(3), "test did not release worker"
        (tmp_path / args.path).write_text(args.content)
        return {"path": args.path, "bytes_written": len(args.content)}

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="write_file",
            description="test write",
            input_model=Args,
            fn=slow_write,
            timeout_seconds=0.02 if stop == "timeout" else 30,
        )
    )
    session = Session(workspace_root=str(tmp_path))
    first = ScriptedProvider()
    first.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="write",
                    name="write_file",
                    arguments={"path": "a.py", "content": "a = 1\n"},
                )
            ]
        ),
        make_response(content="done"),
    )
    second = ScriptedProvider()
    second.script(make_response(content="retry"))
    with FactStore(tmp_path / "memory.db") as store:
        active = asyncio.create_task(turn(session, store, first, registry=registry))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            if stop == "cancel":
                active.cancel()
            await asyncio.sleep(0.05)
            assert not active.done()
            assert (
                await turn(Session(workspace_root=str(tmp_path)), store, second)
            ).completion_status == "blocked"
            if stop == "cancel":
                active.cancel()  # repeated cancellation must not abandon the write
                await asyncio.sleep(0)
                assert not active.done()
        finally:
            release.set()
            if stop == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await active
            else:
                result = await active
                assert "timed out" in result.tool_errors[0]
                assert result.workspace_changes.added == ["a.py"]
        assert (tmp_path / "a.py").read_text() == "a = 1\n"
        assert (await turn(session, store, second)).completion_status == "completed"


def test_admission_is_shared_across_threads_and_releases_rejected_claim(tmp_path: Path) -> None:
    admission = TurnAdmission()
    reasons = []
    with admission.claim("first", str(tmp_path)) as reason:
        assert reason is None
        with admission.claim("first", None) as busy:
            assert busy is not None  # session identity is independent of the workspace

        def conflict():
            with admission.claim("second", str(tmp_path)) as busy:
                reasons.append(busy)

        thread = threading.Thread(target=conflict)
        thread.start()
        thread.join(2)
        assert not thread.is_alive()
        assert reasons and reasons[0] is not None
        with admission.claim("third", str(tmp_path)) as busy:
            assert busy is not None  # rejected second did not release first
    with admission.claim("second", str(tmp_path)) as reason:
        assert reason is None


def test_case_aliases_cannot_bypass_root_or_parent_reservations(tmp_path: Path) -> None:
    root = tmp_path / "mixed_case_repo"
    root.mkdir()
    child = root / "child"
    child.mkdir()
    alias = tmp_path / root.name.upper()
    if not alias.exists():
        pytest.skip("filesystem is case sensitive")
    admission = TurnAdmission()
    with admission.claim("first", str(root)):
        with admission.claim("same", str(alias)) as busy:
            assert busy is not None
        with admission.claim("child", str(alias / "child")) as busy:
            assert busy is not None


@pytest.mark.parametrize("phase", ["before", "after"])
async def test_reservation_covers_workspace_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    import harness.runtime as runtime

    started, release = asyncio.Event(), asyncio.Event()
    original = runtime._snapshot_or_reason
    calls = 0

    async def snapshot(root, settings):
        nonlocal calls
        calls += 1
        if calls == (1 if phase == "before" else 2):
            started.set()
            await release.wait()
        return await original(root, settings)

    monkeypatch.setattr(runtime, "_snapshot_or_reason", snapshot)
    first = ScriptedProvider()
    first.script(make_response(content="done"))
    second = ScriptedProvider()
    second.script(make_response(content="second"))
    with FactStore(tmp_path / "memory.db") as store:
        active = asyncio.create_task(turn(Session(workspace_root=str(tmp_path)), store, first))
        try:
            await asyncio.wait_for(started.wait(), 2)
            blocked = await turn(Session(workspace_root=str(tmp_path)), store, second)
            assert blocked.completion_status == "blocked"
            assert second.calls == []
        finally:
            release.set()
            await active
        assert (
            await turn(Session(workspace_root=str(tmp_path)), store, second)
        ).completion_status == "completed"
