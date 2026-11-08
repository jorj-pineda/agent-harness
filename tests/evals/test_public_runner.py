import pytest

from evals.public_runner import run_public_schedule as run_schedule


def test_fixed_cells_run_serially_once_with_every_state_retained(tmp_path):
    calls, snapshots = [], []
    marker = tmp_path / "started.json"
    result = run_schedule(marker, execute_and_settle=calls.append, persist=snapshots.append)
    assert result.status == "finished"
    assert [cell.ordinal for cell in calls] == list(range(1, 7))
    assert [cell.repeat for cell in calls] == [1] * 3 + [2] * 3
    assert all(state.status == "unrun" for state in snapshots[0].cells)
    assert all(state.status == "recorded" for state in result.cells)
    with pytest.raises(FileExistsError):
        run_schedule(marker, execute_and_settle=calls.append, persist=snapshots.append)
    assert len(calls) == 6


@pytest.mark.parametrize("ordinal", [1, 2, 5, 6])
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
        ["recorded"] * (ordinal - 1) + ["started"] + ["unrun"] * (6 - ordinal)
    )


def test_recording_failure_does_not_dispatch_and_preserves_primary_error(tmp_path):
    calls = []

    def persist(ledger):
        raise OSError("report unavailable")

    with pytest.raises(OSError, match="report unavailable") as error:
        run_schedule(tmp_path / "started.json", execute_and_settle=calls.append, persist=persist)
    assert calls == []
    assert "Final public ledger also failed" in error.value.__notes__[0]
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
    assert calls == list(range(1, 7))
    assert len(result.cells) == 6
    assert result.cells[0].cell.task_id == "divide_zero"


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
