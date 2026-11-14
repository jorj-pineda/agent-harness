"""Serial fresh-visible orchestration seam; caller owns admission, execution and settlement."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import Field

from evals.fresh_visible_workspaces import FreshCell, selected_cell
from evals.heldout_comparison import ProtocolModel
from evals.heldout_inputs import _save
from evals.heldout_rehearsal import ProtocolStop


class CellState(ProtocolModel):
    cell: FreshCell
    status: Literal["unrun", "started", "recorded"] = "unrun"


class FreshLedger(ProtocolModel):
    format: Literal["fresh-visible-ledger-v1"] = "fresh-visible-ledger-v1"
    status: Literal["planned", "running", "finished", "stopped"] = "planned"
    cells: list[CellState] = Field(min_length=12, max_length=12)
    error_type: str | None = None
    stop_reason: str | None = None


def run_fresh_schedule(
    marker: Path,
    *,
    execute_and_settle: Callable[[FreshCell], None],
    persist: Callable[[FreshLedger], None],
) -> FreshLedger:
    """Enforce one schedule and stop on any cell/settlement/recording failure.

    Not a launch API. Trusted caller supplies an evaluation-owned marker path,
    validates all selections and persists reports safely. The callback must record
    its result and settle owned resources before returning; it must raise for all
    protocol stop conditions. Neither a callback nor this ledger proves settlement
    or approval. Single-use marker and partial evidence are never removed here.
    """
    cells = [selected_cell(i)[0] for i in range(1, 13)]
    ledger = FreshLedger(cells=[CellState(cell=cell) for cell in cells])
    _save(marker, b'{"format":"fresh-visible-schedule-started-v1"}')
    try:
        persist(ledger.model_copy(deep=True))
        ledger = ledger.model_copy(update={"status": "running"})
        for index, state in enumerate(ledger.cells):
            cells = list(ledger.cells)
            cells[index] = state.model_copy(update={"status": "started"})
            ledger = ledger.model_copy(update={"cells": cells})
            persist(ledger.model_copy(deep=True))
            execute_and_settle(state.cell.model_copy(deep=True))
            cells[index] = state.model_copy(update={"status": "recorded"})
            ledger = ledger.model_copy(update={"cells": cells})
            persist(ledger.model_copy(deep=True))
        ledger = ledger.model_copy(update={"status": "finished"})
        persist(ledger.model_copy(deep=True))
    except BaseException as error:
        ledger = ledger.model_copy(
            update={
                "status": "stopped",
                "error_type": type(error).__name__,
                "stop_reason": str(error) if isinstance(error, ProtocolStop) else None,
            }
        )
        try:
            persist(ledger.model_copy(deep=True))
        except BaseException as recording_error:
            error.add_note(f"Final fresh ledger also failed: {type(recording_error).__name__}")
        raise
    return ledger
