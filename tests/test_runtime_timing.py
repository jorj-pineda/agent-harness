"""Review time consumes dispatch budget without abandoning final evidence."""

import asyncio
import threading

import pytest

from harness.admission import turn_admission
from harness.config import Settings
from harness.loop import run_turn
from harness.runtime import build_registry, run_configured_turn
from harness.state import Session
from memory import FactStore
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tools import ToolRegistry


async def test_initial_snapshot_consumes_budget_and_holds_admission(tmp_path, monkeypatch):
    import harness.runtime as runtime

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("user edit\n")
    entered, release = threading.Event(), threading.Event()
    original = runtime.snapshot
    calls = 0

    def held(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "snapshot", held)
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="w",
                    name="write_file",
                    arguments={"path": "a.py", "content": "should never run"},
                )
            ]
        )
    )
    session = Session(user_id="dev", workspace_root=str(repo))
    finalized = []
    with FactStore(tmp_path / "facts.db") as store:
        task = asyncio.create_task(
            run_configured_turn(
                settings=Settings(_env_file=None, max_turn_wall_seconds=0.01),
                session=session,
                user_id="dev",
                message="Fix a.py",
                provider=provider,
                fact_store=store,
                registry=build_registry(fact_store=store, user_id="dev", workspace_root=str(repo)),
                on_finalized=lambda _, response: finalized.append(response.model_dump()),
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            await asyncio.sleep(0.03)
            assert not task.done()
            with turn_admission.claim("other", str(repo)) as busy:
                assert busy is not None
            assert provider.calls == []
        finally:
            release.set()
        response = await task
    assert response.completion_status == "budget_exhausted"
    assert "wall-time" in response.completion_reason
    assert provider.calls == [] and response.tool_calls == []
    assert (repo / "a.py").read_text() == "user edit\n"
    assert response.workspace_changes.status == "tracked"
    assert response.workspace_review_ms >= 30
    assert finalized[0]["workspace_review_ms"] == response.workspace_review_ms
    with turn_admission.claim("other", str(repo)) as busy:
        assert busy is None


async def test_final_review_settles_after_deadline_without_relabeling_completed_work(
    tmp_path, monkeypatch
):
    import harness.runtime as runtime

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("user = 1\n")
    entered, release = threading.Event(), threading.Event()
    original = runtime.snapshot
    calls = 0

    def held(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "snapshot", held)
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="w",
                    name="write_file",
                    arguments={"path": "a.py", "content": "user = 1\nagent = 2\n"},
                )
            ]
        ),
        make_response(content="Done"),
    )
    session = Session(user_id="dev", workspace_root=str(repo))
    finalized = []
    with FactStore(tmp_path / "facts.db") as store:
        task = asyncio.create_task(
            run_configured_turn(
                settings=Settings(_env_file=None, max_turn_wall_seconds=1),
                session=session,
                user_id="dev",
                message="Fix a.py",
                provider=provider,
                fact_store=store,
                registry=build_registry(fact_store=store, user_id="dev", workspace_root=str(repo)),
                on_finalized=lambda _, response: finalized.append(response.model_dump()),
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            await asyncio.sleep(1.05)
            assert not task.done() and not finalized
            with turn_admission.claim("other", str(repo)) as busy:
                assert busy is not None
            assert len(provider.calls) == 2
        finally:
            release.set()
        response = await task
    assert response.completion_status == "completed"
    assert response.workspace_review_ms >= 1000
    assert response.workspace_review_ms > response.latency_ms
    assert "+agent = 2" in response.workspace_changes.diffs[0].diff
    assert (repo / "a.py").read_text() == "user = 1\nagent = 2\n"
    assert finalized[0]["workspace_review_ms"] == response.workspace_review_ms


@pytest.mark.parametrize("mode", ["disabled", "no_workspace", "failed"])
async def test_skipped_or_failed_tracking_has_honest_timing(tmp_path, mode):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a").write_text("one")
    (repo / "b").write_text("two")
    provider = ScriptedProvider()
    provider.script(make_response(content="Explained"))
    root = None if mode == "no_workspace" else str(repo)
    with FactStore(tmp_path / "facts.db") as store:
        response = await run_configured_turn(
            settings=Settings(
                _env_file=None,
                track_workspace_changes=mode != "disabled",
                max_tracked_files=1 if mode == "failed" else 100,
            ),
            session=Session(workspace_root=root),
            user_id="dev",
            message="Explain",
            provider=provider,
            fact_store=store,
            registry=build_registry(fact_store=store, user_id="dev", workspace_root=root),
        )
    assert response.workspace_review_ms is not None
    if mode == "failed":
        assert response.workspace_changes.status == "unavailable"
        assert response.workspace_review_ms > 0
    else:
        assert response.workspace_changes.status == "not_tracked"
        assert response.workspace_review_ms == 0


async def test_low_level_deadline_can_use_earlier_start_and_timing_remains_unmeasured():
    provider = ScriptedProvider()
    provider.script(make_response(content="Unreached"))
    response = await run_turn(
        session=Session(),
        user_input="Explain",
        provider=provider,
        registry=ToolRegistry(),
        max_turn_wall_seconds=1,
        wall_started_at=asyncio.get_running_loop().time() - 2,
    )
    assert response.completion_status == "budget_exhausted"
    assert provider.calls == []
    assert response.workspace_review_ms is None


async def test_configuration_capture_time_also_consumes_dispatch_budget(tmp_path):
    import time

    provider = ScriptedProvider()
    provider.script(make_response(content="Unreached"))
    captured = []

    def capture(profile):
        captured.append(profile)
        time.sleep(0.03)

    with FactStore(tmp_path / "facts.db") as store:
        response = await run_configured_turn(
            settings=Settings(_env_file=None, max_turn_wall_seconds=0.01),
            session=Session(),
            user_id="dev",
            message="Explain",
            provider=provider,
            fact_store=store,
            registry=ToolRegistry(),
            on_execution_profile=capture,
        )
    assert len(captured) == 1
    assert provider.calls == []
    assert response.completion_status == "budget_exhausted"
    assert response.workspace_review_ms == 0  # Setup is in the deadline, not review duration.
