import pytest

from evals.public_feedback_runner import run_feedback_schedule as run_schedule


def test_fixed_cells_run_serially_once_with_every_state_retained(tmp_path):
    calls, snapshots = [], []
    marker = tmp_path / "started.json"
    result = run_schedule(marker, execute_and_settle=calls.append, persist=snapshots.append)
    assert result.status == "finished"
    assert [cell.ordinal for cell in calls] == list(range(1, 9))
    assert [cell.repeat for cell in calls] == [1, 1, 2, 2, 3, 3, 4, 4]
    assert [cell.arm for cell in calls] == [
        "control",
        "bool_check",
        "bool_check",
        "control",
        "control",
        "bool_check",
        "bool_check",
        "control",
    ]
    assert result.format == "public-feedback-ledger-v1"
    assert all(state.status == "unrun" for state in snapshots[0].cells)
    assert all(state.status == "recorded" for state in result.cells)
    with pytest.raises(FileExistsError):
        run_schedule(marker, execute_and_settle=calls.append, persist=snapshots.append)
    assert len(calls) == 8


@pytest.mark.parametrize("ordinal", [1, 2, 7, 8])
def test_any_cell_failure_stops_public_schedule_without_retry(tmp_path, ordinal):
    calls, snapshots = [], []

    def execute(cell):
        calls.append(cell.ordinal)
        if cell.ordinal == ordinal:
            raise ValueError("unknown usage or unsettled cell")

    with pytest.raises(ValueError):
        run_schedule(
            tmp_path / "started.json", execute_and_settle=execute, persist=snapshots.append
        )
    assert calls == list(range(1, ordinal + 1))
    final = snapshots[-1]
    assert final.status == "stopped"
    assert [s.status for s in final.cells] == (
        ["recorded"] * (ordinal - 1) + ["started"] + ["unrun"] * (8 - ordinal)
    )


def test_recording_failure_does_not_dispatch_and_preserves_primary_error(tmp_path):
    calls = []

    def persist(ledger):
        raise OSError("report unavailable")

    with pytest.raises(OSError, match="report unavailable") as error:
        run_schedule(tmp_path / "started.json", execute_and_settle=calls.append, persist=persist)
    assert calls == []
    assert "Final feedback ledger also failed" in error.value.__notes__[0]
    assert (tmp_path / "started.json").exists()


def test_callback_and_snapshot_mutations_do_not_rewrite_schedule(tmp_path):
    calls = []
    snapshots = []

    def execute(cell):
        calls.append(cell.ordinal)
        object.__setattr__(cell, "ordinal", 6)

    def persist(ledger):
        snapshots.append(ledger)
        object.__setattr__(ledger.cells[0].cell, "task_id", "callback-mutated")
        ledger.cells.clear()

    result = run_schedule(tmp_path / "started.json", execute_and_settle=execute, persist=persist)
    assert calls == list(range(1, 9))
    assert len(result.cells) == 8
    assert result.cells[0].cell.task_id == "paginate_items"


def test_protocol_stop_is_recorded_without_supplying_authority(tmp_path):
    from evals.heldout_rehearsal import ProtocolStop

    snapshots = []

    def refuse(cell):
        raise ProtocolStop("missing_usage")

    with pytest.raises(ProtocolStop):
        run_schedule(tmp_path / "started.json", execute_and_settle=refuse, persist=snapshots.append)
    assert snapshots[-1].stop_reason == "missing_usage"
    assert snapshots[-1].error_type == "ProtocolStop"


def test_started_state_must_persist_before_callback(tmp_path):
    calls = []
    snapshots = []

    def persist(ledger):
        snapshots.append(ledger)
        if ledger.cells[0].status == "started":
            raise OSError("cannot retain start")

    with pytest.raises(OSError):
        run_schedule(tmp_path / "started.json", execute_and_settle=calls.append, persist=persist)
    assert not calls and snapshots[-1].status == "stopped"


def test_competing_schedules_consume_one_marker(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    ready = Barrier(2)
    calls = []

    def run():
        ready.wait(timeout=10)
        return run_schedule(
            tmp_path / "started.json", execute_and_settle=calls.append, persist=lambda ledger: None
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run) for _ in range(2)]
        results, failures = [], []
        for future in futures:
            try:
                results.append(future.result(timeout=30))
            except FileExistsError as error:
                failures.append(error)
    assert len(results) == len(failures) == 1
    assert [cell.ordinal for cell in calls] == list(range(1, 9))


def test_cancellation_retains_started_cell_and_stops_later_dispatch(tmp_path):
    import asyncio

    snapshots = []
    calls = []

    def cancel(cell):
        calls.append(cell.ordinal)
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        run_schedule(tmp_path / "started.json", execute_and_settle=cancel, persist=snapshots.append)
    assert calls == [1] and snapshots[-1].status == "stopped"
    assert snapshots[-1].error_type == "CancelledError"
    assert [s.status for s in snapshots[-1].cells] == ["started"] + ["unrun"] * 7
