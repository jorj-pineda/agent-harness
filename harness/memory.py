"""Harness-side memory harvesting: turn history → `memory_writes`.

The memory tool (`tools/memory.remember_fact`) persists facts into the
`FactStore` during a turn. This module answers the question the API layer
cares about: "what did this turn actually write to memory?"

Symmetric with `harness/grounding.py`:

* Pure function over `ToolCallRecord` list — no I/O, no provider calls.
* A single well-known tool name (`remember_fact`) is the signal, just as
  grounding keys off `search_docs`.
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

REMEMBER_FACT_TOOL_NAME = "remember_fact"


def harvest_memory_writes(tool_calls: list[ToolCallRecord]) -> list[str]:
    """Return facts actually persisted this turn, in call order.

    A call counts as a write iff:
      * its name is `remember_fact`,
      * no error was raised,
      * the result is a dict with `stored=True` and a non-empty `fact` string.
    """
    writes: list[str] = []
    for call in tool_calls:
        if call.name != REMEMBER_FACT_TOOL_NAME or call.error is not None:
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
