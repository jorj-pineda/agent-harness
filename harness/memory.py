"""Harness-side memory harvesting: turn history → `memory_writes`.

The memory tool (`remember_fact` / `remember`) persists notes into the
`FactStore` during a turn. This module answers the question the API layer
cares about: "what did this turn actually write to memory?"

Symmetric with `harness/grounding.py`:

* Pure function over `ToolCallRecord` list — no I/O, no provider calls.
* Well-known tool names (`remember_fact`, `remember`) are the signal.
* Only *successful* writes are harvested. `INSERT OR IGNORE` at the store
  layer means duplicates return `stored=False`; those are intent, not
  persistence, and don't belong in `memory_writes`.

`run_turn` always runs this scan — it's O(|tool_calls|) and returns [] when
no memory calls happened, so there's no need for an opt-in kwarg.
"""

from __future__ import annotations

import logging

from .state import ToolCallRecord

log = logging.getLogger(__name__)

REMEMBER_TOOL_NAMES = frozenset({"remember_fact", "remember"})


def harvest_memory_writes(tool_calls: list[ToolCallRecord]) -> list[str]:
    """Return engineering notes actually persisted this turn, in call order.

    A call counts as a write iff:
      * its name is `remember_fact` or `remember`,
      * no error was raised,
      * the result is a dict with `stored=True` and a non-empty `fact` string.
    """
    writes: list[str] = []
    for call in tool_calls:
        if call.name not in REMEMBER_TOOL_NAMES or call.error is not None:
            continue
        result = call.result
        if not isinstance(result, dict):
            continue
        if result.get("stored") is not True:
            continue
        fact = result.get("fact")
        if isinstance(fact, str) and fact:
            writes.append(fact)
    log.info("memory_harvest count=%d", len(writes))
    return writes
