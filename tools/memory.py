"""Long-term engineering memory tools.

Exposes four tool names (two write, two read) bound to the same handlers:

* `remember_fact` / `remember(note)` — persist a user-scoped engineering note.
* `recall_facts` / `recall(max_results?)` — return notes, most recent first.

The `user_id` is closed over by the factory, not passed as a tool argument.
Cross-user isolation is therefore structural: nothing the model can emit
reaches another user's memory, because the only reference to `user_id`
lives in this module's closure.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from memory import DEFAULT_LIST_LIMIT, FactStore

from .base import Tool
from .registry import ToolRegistry

log = logging.getLogger(__name__)

MAX_RECALL = 50

REMEMBER_TOOL_NAMES = frozenset({"remember_fact", "remember"})
RECALL_TOOL_NAMES = frozenset({"recall_facts", "recall"})


class RememberFactInput(BaseModel):
    fact: str = Field(
        ...,
        min_length=1,
        description=(
            "A concise engineering note to persist (e.g. 'prefer async handlers', "
            "'use ruff format', 'never force-push main')."
        ),
    )


class RememberInput(BaseModel):
    note: str = Field(
        ...,
        min_length=1,
        description="Alias for remember_fact — a durable engineering note.",
    )


class RecallFactsInput(BaseModel):
    max_results: int = Field(DEFAULT_LIST_LIMIT, ge=1, le=MAX_RECALL)


class RecallInput(BaseModel):
    max_results: int = Field(DEFAULT_LIST_LIMIT, ge=1, le=MAX_RECALL)


def build_memory_tools(store: FactStore, user_id: str) -> list[Tool]:
    """Build memory Tool instances bound to a specific user."""
    if not user_id:
        raise ValueError("user_id must not be empty")

    async def remember_fact(args: RememberFactInput) -> dict[str, object]:
        log.info("memory_tool=remember_fact user_id=%s", user_id)
        inserted = store.add(user_id, args.fact)
        return {"stored": inserted, "fact": args.fact.strip()}

    async def remember(args: RememberInput) -> dict[str, object]:
        log.info("memory_tool=remember user_id=%s", user_id)
        inserted = store.add(user_id, args.note)
        return {"stored": inserted, "fact": args.note.strip()}

    async def recall_facts(args: RecallFactsInput) -> list[str]:
        log.info(
            "memory_tool=recall_facts user_id=%s max_results=%d",
            user_id,
            args.max_results,
        )
        return [f.fact for f in store.list(user_id, limit=args.max_results)]

    async def recall(args: RecallInput) -> list[str]:
        log.info(
            "memory_tool=recall user_id=%s max_results=%d",
            user_id,
            args.max_results,
        )
        return [f.fact for f in store.list(user_id, limit=args.max_results)]

    remember_description = (
        "Persist a durable engineering note for this developer across sessions "
        "(stack choices, repo conventions, review preferences). Returns "
        "{stored: true} on first insert, {stored: false} if the exact note "
        "was already known."
    )
    recall_description = (
        "Return engineering notes previously remembered for this developer, "
        "most recently added first. Notes are also injected into the system "
        "prompt at the start of each turn."
    )

    return [
        Tool(
            name="remember_fact",
            description=remember_description,
            input_model=RememberFactInput,
            fn=remember_fact,
        ),
        Tool(
            name="remember",
            description=f"Alias for remember_fact. {remember_description}",
            input_model=RememberInput,
            fn=remember,
        ),
        Tool(
            name="recall_facts",
            description=recall_description,
            input_model=RecallFactsInput,
            fn=recall_facts,
        ),
        Tool(
            name="recall",
            description=f"Alias for recall_facts. {recall_description}",
            input_model=RecallInput,
            fn=recall,
        ),
    ]


def register_memory_tools(
    registry: ToolRegistry,
    *,
    store: FactStore,
    user_id: str,
) -> None:
    """Register memory tools on the given registry for the given user."""
    for tool in build_memory_tools(store, user_id):
        registry.register(tool)
