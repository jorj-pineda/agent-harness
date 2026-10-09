"""Process-local admission for turns sharing session state or workspace paths."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def _overlap(left: Path, right: Path) -> bool:
    if left.is_relative_to(right) or right.is_relative_to(left):
        return True
    # Path.resolve resolves symlinks, but not all case aliases on macOS.
    for root, candidate in ((left, right), (right, left)):
        for ancestor in (candidate, *candidate.parents):
            try:
                if root.samefile(ancestor):
                    return True
            except OSError:
                continue
    return False


class TurnAdmission:
    """Reject overlap without queuing work or holding an event-loop-bound lock.

    The mutex covers only reservation bookkeeping. A reservation covers the whole
    configured turn, including snapshots and tool cleanup. External writers and
    other harness processes do not participate in this guard.
    """

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._active: dict[str, Path | None] = {}

    @contextmanager
    def claim(self, session_id: str, workspace_root: str | None) -> Iterator[str | None]:
        root = Path(workspace_root).expanduser().resolve() if workspace_root else None
        with self._mutex:
            reason = None
            if session_id in self._active:
                reason = "This session already has an active turn. Retry after it finishes."
            elif root is not None and any(
                active is not None and _overlap(root, active) for active in self._active.values()
            ):
                reason = "This workspace overlaps an active turn. Retry after it finishes."
            if reason is None:
                self._active[session_id] = root
        try:
            yield reason
        finally:
            if reason is None:
                with self._mutex:
                    del self._active[session_id]


# Shared by API and real-task runtime calls, across event loops in this process.
turn_admission = TurnAdmission()
