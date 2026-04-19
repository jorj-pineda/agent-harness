"""Embed the support corpus into a persistent Chroma collection.

Embedding is a one-shot offline pipeline: load chunks via `data.corpus`,
run them through an `Embedder`, and upsert into a persistent Chroma
collection keyed by `chunk_id`. Because chunk IDs are stable (see
`data.corpus`), re-running the pipeline replaces only sections whose
heading or body actually changed — the downstream vector index never
accumulates stale duplicates.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import chromadb

from data.corpus import Chunk, load_chunks

DEFAULT_CHROMA_DIR = Path(__file__).parent / "chroma"
DEFAULT_COLLECTION = "support_corpus"
DEFAULT_BATCH_SIZE = 32


class Embedder(Protocol):
    """Minimal embedding contract.

    Duplicates the shape of `providers.base.Embedder` so this module can be
    exercised in isolation with a fake, without pulling the provider package
    into the data layer.
    """

    name: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


def open_collection(
    *,
    chroma_dir: Path | str = DEFAULT_CHROMA_DIR,
    name: str = DEFAULT_COLLECTION,
) -> chromadb.Collection:
    """Return a persistent Chroma collection, creating it if needed.

    `embedding_function=None` tells Chroma not to auto-embed — we always
    supply vectors explicitly, so the default ONNX embedder (which would
    try to download a model at runtime) stays out of the way.
    """
    path = Path(chroma_dir)
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(path),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(name=name, embedding_function=None)


async def embed_corpus(
    chunks: Iterable[Chunk],
    *,
    embedder: Embedder,
    collection: chromadb.Collection,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Embed and upsert every chunk; return the number written."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    chunk_list = list(chunks)
    if not chunk_list:
        return 0

    for start in range(0, len(chunk_list), batch_size):
        batch = chunk_list[start : start + batch_size]
        vectors = await embedder.embed([c.text for c in batch])
        if len(vectors) != len(batch):
            raise ValueError(f"Embedder returned {len(vectors)} vectors for {len(batch)} chunks")
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=vectors,
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "section": c.section,
                    "category": c.category,
                    "source_path": c.source_path,
                }
                for c in batch
            ],
        )
    return len(chunk_list)


async def _amain() -> None:
    # Imported inside _amain so the library surface of this module stays
    # free of any concrete provider — the CLI is the only place Ollama is wired.
    from providers import create_embedder

    chroma_dir = Path(os.getenv("CHROMA_PATH", str(DEFAULT_CHROMA_DIR)))
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    chat_model = os.getenv("OLLAMA_MODEL", "gemma4")

    embedder = create_embedder(
        "ollama",
        host=ollama_host,
        model=chat_model,
        embed_model=embed_model,
    )
    try:
        collection = open_collection(chroma_dir=chroma_dir)
        count = await embed_corpus(load_chunks(), embedder=embedder, collection=collection)
    finally:
        aclose = getattr(embedder, "aclose", None)
        if aclose is not None:
            await aclose()

    print(f"Embedded {count} chunks into '{DEFAULT_COLLECTION}' at {chroma_dir}")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
