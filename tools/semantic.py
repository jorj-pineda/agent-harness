"""Deferred semantic codebase search — stub only (Phase 7).

Indexing strategy is ripgrep-first via `grep_repo`. A future `semantic_search`
tool can plug in here behind the same `ToolRegistry` interface once evals
show grep-only explore failures.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .base import Tool, ToolError
from .registry import ToolRegistry


class SemanticSearchInput(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query.")
    k: int = Field(default=5, ge=1, le=20)


async def semantic_search(_args: SemanticSearchInput) -> list[dict[str, str]]:
    raise ToolError(
        "semantic_search is not enabled. Use grep_repo for codebase search "
        "(ripgrep-first indexing; see README)."
    )


def register_semantic_search_stub(registry: ToolRegistry) -> None:
    """Register the deferred semantic search tool (always raises ToolError)."""
    registry.register(
        Tool(
            name="semantic_search",
            description=(
                "Deferred: semantic codebase search via embeddings. Not enabled in v1; "
                "use grep_repo instead."
            ),
            input_model=SemanticSearchInput,
            fn=semantic_search,
        )
    )
