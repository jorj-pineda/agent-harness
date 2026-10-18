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

import json
import logging
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field

from memory import DEFAULT_LIST_LIMIT, FactStore
from memory.store import MAX_FACT_BYTES

from .base import Tool, ToolError
from .registry import ToolRegistry

log = logging.getLogger(__name__)

MAX_RECALL = 50
MAX_RECALL_BYTES = 32000

REMEMBER_TOOL_NAMES = frozenset({"remember_fact", "remember"})
RECALL_TOOL_NAMES = frozenset({"recall_facts", "recall"})


class MemoryScopeInput(BaseModel):
    scope: Literal["current", "global"] = Field(
        default="current",
        description="current uses this workspace (or global without one); global is user-wide.",
    )


class RememberFactInput(MemoryScopeInput):
    fact: str = Field(
        ...,
        min_length=1,
        description=(
            "A concise engineering note to persist (e.g. 'prefer async handlers', "
            "'use ruff format', 'never force-push main')."
        ),
    )


class RememberInput(MemoryScopeInput):
    note: str = Field(
        ...,
        min_length=1,
        description="Alias for remember_fact — a durable engineering note.",
    )


class RecallFactsInput(MemoryScopeInput):
    max_results: int = Field(DEFAULT_LIST_LIMIT, ge=1, le=MAX_RECALL)


class RecallInput(MemoryScopeInput):
    max_results: int = Field(DEFAULT_LIST_LIMIT, ge=1, le=MAX_RECALL)


def build_memory_tools(
    store: FactStore,
    user_id: str,
    *,
    source_turn_id: Callable[[], str | None] | None = None,
    workspace_root: str | None = None,
    include_aliases: bool = True,
) -> list[Tool]:
    """Build memory Tool instances bound to a specific user."""
    if not user_id:
        raise ValueError("user_id must not be empty")

    async def remember_fact(args: RememberFactInput) -> dict[str, object]:
        log.info("memory_tool=remember_fact user_id=%s", user_id)
        inserted = store.add(
            user_id,
            args.fact,
            source_turn_id=source_turn_id() if source_turn_id else None,
            workspace_root=workspace_root if args.scope == "current" else None,
        )
        return {"stored": inserted, "fact": args.fact.strip()}

    async def remember(args: RememberInput) -> dict[str, object]:
        log.info("memory_tool=remember user_id=%s", user_id)
        inserted = store.add(
            user_id,
            args.note,
            source_turn_id=source_turn_id() if source_turn_id else None,
            workspace_root=workspace_root if args.scope == "current" else None,
        )
        return {"stored": inserted, "fact": args.note.strip()}

    def recall_notes(limit: int, scope: str) -> list[str]:
        notes = [
            f.fact
            for f in store.list(
                user_id,
                limit=limit,
                scoped=True,
                workspace_root=workspace_root if scope == "current" else None,
            )
        ]
        # Match the loop's default JSON tool-message encoding, including ASCII
        # escapes. Raw UTF-8 note sizes underestimate this serialized payload.
        if len(json.dumps(notes, default=str).encode("utf-8")) > MAX_RECALL_BYTES:
            raise ToolError(
                f"Recall exceeds {MAX_RECALL_BYTES} serialized JSON bytes. Reduce max_results; "
                "if one legacy note still exceeds the limit, "
                "inspect stored memory outside the model."
            )
        return notes

    async def recall_facts(args: RecallFactsInput) -> list[str]:
        log.info(
            "memory_tool=recall_facts user_id=%s max_results=%d",
            user_id,
            args.max_results,
        )
        return recall_notes(args.max_results, args.scope)

    async def recall(args: RecallInput) -> list[str]:
        log.info(
            "memory_tool=recall user_id=%s max_results=%d",
            user_id,
            args.max_results,
        )
        return recall_notes(args.max_results, args.scope)

    remember_description = (
        "Persist a durable engineering note for this developer across sessions "
        "(stack choices, repo conventions, review preferences). Returns "
        "{stored: true} on first insert, {stored: false} if the exact note "
        "was already known in that namespace. Default current scope stores workspace conventions; "
        "choose global for user-wide preferences. "
        f"Notes must fit {MAX_FACT_BYTES} UTF-8 bytes after trimming."
    )
    recall_description = (
        "Return engineering notes previously remembered for this developer, "
        "most recently added first. Current scope returns global plus current-workspace notes; "
        "global scope returns only user-wide notes. Other workspaces cannot be selected. "
        "Notes are also injected into the system "
        "prompt at the start of each turn within a separate byte limit. "
        f"Recall rejects responses over {MAX_RECALL_BYTES} serialized JSON bytes; "
        "reduce max_results on overflow. No notes are silently dropped from a recall result."
    )

    tools = [
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
    return (
        tools
        if include_aliases
        else [t for t in tools if t.name in {"remember_fact", "recall_facts"}]
    )


def register_memory_tools(
    registry: ToolRegistry,
    *,
    store: FactStore,
    user_id: str,
    source_turn_id: Callable[[], str | None] | None = None,
    workspace_root: str | None = None,
    include_aliases: bool = True,
) -> None:
    """Register memory tools on the given registry for the given user."""
    for tool in build_memory_tools(
        store,
        user_id,
        source_turn_id=source_turn_id,
        workspace_root=workspace_root,
        include_aliases=include_aliases,
    ):
        registry.register(tool)
