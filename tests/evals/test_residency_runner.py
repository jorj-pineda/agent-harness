import pytest

from evals.residency_runner import run_schedule


def test_fixed_cells_run_serially_once_with_every_state_retained(tmp_path):
    calls, snapshots = [], []
    marker = tmp_path / "started.json"
    result = run_schedule(marker, execute_and_settle=calls.append, persist=snapshots.append)
    assert result.status == "finished"
    assert [cell.ordinal for cell in calls] == list(range(1, 17))
    assert [cell.keep_alive_seconds for cell in calls] == [0, 300] * 4 + [300, 0] * 4
    assert all(state.status == "unrun" for state in snapshots[0].cells)
    assert all(state.status == "recorded" for state in result.cells)
    with pytest.raises(FileExistsError):
        run_schedule(marker, execute_and_settle=calls.append, persist=snapshots.append)
    assert len(calls) == 16


@pytest.mark.parametrize("ordinal", [1, 2, 9, 16])
def test_any_cell_failure_stops_both_arms_without_retry(tmp_path, ordinal):
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
        ["recorded"] * (ordinal - 1) + ["started"] + ["unrun"] * (16 - ordinal)
    )


def test_recording_failure_does_not_dispatch_and_preserves_primary_error(tmp_path):
    calls = []

    def persist(ledger):
        raise OSError("report unavailable")

    with pytest.raises(OSError, match="report unavailable") as error:
        run_schedule(tmp_path / "started.json", execute_and_settle=calls.append, persist=persist)
    assert calls == []
    assert "Final residency ledger also failed" in error.value.__notes__[0]
    assert (tmp_path / "started.json").exists()
