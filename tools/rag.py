"""Retrieval tool over the embedded support corpus.

`search_docs` takes a natural-language query, embeds it via a caller-supplied
embedder, and runs a nearest-neighbor query against a pre-opened Chroma
collection (populated by `data.embed`). Each hit is returned with its
citation metadata so the grounding layer can trace every claim back to a
chunk.

The factory takes an already-opened collection rather than a path + name:
tests can construct an in-tmp collection and pass it directly, and the
harness can share a single open collection across many tool invocations.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import chromadb
from pydantic import BaseModel, Field

from .base import Tool
from .registry import ToolRegistry

log = logging.getLogger(__name__)

DEFAULT_K = 4
MAX_K = 20


class Embedder(Protocol):
    """Minimal async embedding contract.

    Structurally identical to `data.embed.Embedder` and
    `providers.base.Embedder` — duplicated here so `tools/` does not take a
    dependency on the data pipeline or on any specific provider.
    """

    name: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class RagQueryInput(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query")
    k: int = Field(DEFAULT_K, ge=1, le=MAX_K, description="Top-k chunks to return (1-20)")
    category: str | None = Field(
        None,
        description="Optional doc category filter, e.g. 'returns', 'shipping', 'warranty'.",
    )


def _log_call(*, query: str, k: int, category: str | None, hits: int) -> None:
    log.info("rag_tool=search_docs q=%r k=%d category=%s hits=%d", query, k, category, hits)


def build_rag_tool(
    *,
    collection: chromadb.Collection,
    embedder: Embedder,
) -> Tool:
    """Build the `search_docs` Tool bound to a specific collection + embedder.

    The collection is assumed to already hold embedded chunks whose metadata
    matches the shape written by `data.embed.embed_corpus` (doc_id, title,
    section, category, source_path).
    """

    async def search_docs(args: RagQueryInput) -> list[dict[str, Any]]:
        """Retrieve the top-k most similar support-corpus chunks for a query."""
        [vector] = await embedder.embed([args.query])

        where = {"category": args.category} if args.category is not None else None
        response = collection.query(
            query_embeddings=[vector],
            n_results=args.k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]

        hits: list[dict[str, Any]] = []
        for chunk_id, text, meta, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": meta.get("doc_id"),
                    "title": meta.get("title"),
                    "section": meta.get("section"),
                    "category": meta.get("category"),
                    "source_path": meta.get("source_path"),
                    "text": text,
                    "score": 1.0 - float(distance),
                }
            )

        _log_call(query=args.query, k=args.k, category=args.category, hits=len(hits))
        return hits

    return Tool(
        name="search_docs",
        description=(
            "Search the embedded support documentation for chunks relevant to a "
            "natural-language query. Returns the top-k chunks with citation "
            "metadata (chunk_id, doc_id, title, section, category, source_path) "
            "and a similarity score. Optional category filter narrows to a single "
            "doc category such as 'returns' or 'shipping'."
        ),
        input_model=RagQueryInput,
        fn=search_docs,
    )


def register_rag_tool(
    registry: ToolRegistry,
    *,
    collection: chromadb.Collection,
    embedder: Embedder,
) -> None:
    """Register the `search_docs` tool on the given registry."""
    registry.register(build_rag_tool(collection=collection, embedder=embedder))
