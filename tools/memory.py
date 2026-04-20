"""Long-term personalization memory tools.

Exposes two tools to the agent:

* `remember_fact(fact)` — persist a user-scoped fact via `FactStore.add`.
* `recall_facts(max_results?)` — return the user's facts, most recent first.

The `user_id` is closed over by the factory, not passed as a tool argument.
Cross-user isolation is therefore structural: nothing the model can emit
reaches another user's memory, because the only reference to `user_id`
lives in this module's closure.

Mirrors the `build_sql_tools(db_path)` factory flavor — plain Tool objects
so tests can wire tools into a fresh registry without module-level state.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from memory import DEFAULT_LIST_LIMIT, FactStore

from .base import Tool
from .registry import ToolRegistry

log = logging.getLogger(__name__)

MAX_RECALL = 50


class RememberFactInput(BaseModel):
    fact: str = Field(..., min_length=1, description="A concise fact about the user to persist.")


class RecallFactsInput(BaseModel):
    max_results: int = Field(DEFAULT_LIST_LIMIT, ge=1, le=MAX_RECALL)


def build_memory_tools(store: FactStore, user_id: str) -> list[Tool]:
    """Build memory Tool instances bound to a specific user.

    The `user_id` is captured in the closure — the agent cannot address
    another user's memory because the tool signatures do not accept it.
    """
    if not user_id:
        raise ValueError("user_id must not be empty")

    # Tools are async so the registry keeps them on the event-loop thread —
    # FactStore holds a persistent sqlite3 connection, which sqlite3 binds
    # to the thread that opened it. Running through `asyncio.to_thread`
    # would trigger `ProgrammingError` on cross-thread reuse.
    async def remember_fact(args: RememberFactInput) -> dict[str, object]:
        """Persist a long-term fact about the current user."""
        log.info("memory_tool=remember_fact user_id=%s", user_id)
        inserted = store.add(user_id, args.fact)
        return {"stored": inserted, "fact": args.fact.strip()}

    async def recall_facts(args: RecallFactsInput) -> list[str]:
        """Return facts previously remembered about the current user."""
        log.info(
            "memory_tool=recall_facts user_id=%s max_results=%d",
            user_id,
            args.max_results,
        )
        return [f.fact for f in store.list(user_id, limit=args.max_results)]

    return [
        Tool(
            name="remember_fact",
            description=(
                "Persist a durable, cross-session fact about the current user "
                "(e.g. 'prefers vegan options', 'size 9 shoe'). Returns "
                "{stored: true} on first insert, {stored: false} if the exact "
                "fact was already known."
            ),
            input_model=RememberFactInput,
            fn=remember_fact,
        ),
        Tool(
            name="recall_facts",
            description=(
                "Return the list of long-term facts previously remembered about "
                "the current user, most recently added first."
            ),
            input_model=RecallFactsInput,
            fn=recall_facts,
        ),
    ]


def register_memory_tools(
    registry: ToolRegistry,
    *,
    store: FactStore,
    user_id: str,
) -> None:
    """Register memory tools on the given registry for the given user."""
    for t in build_memory_tools(store, user_id):
        registry.register(t)
