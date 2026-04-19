from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from data.corpus import Chunk, load_chunks
from data.embed import (
    DEFAULT_COLLECTION,
    embed_corpus,
    open_collection,
)


class FakeEmbedder:
    """Deterministic hash-based embedder for test reproducibility."""

    name = "fake"
    dim = 8

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [_vec(t, self.dim) for t in texts]


class WrongLengthEmbedder:
    name = "wrong-length"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_vec(t, 4) for t in texts[:-1]]


def _vec(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[i % len(digest)] / 255.0 for i in range(dim)]


def _chunk(chunk_id: str, text: str, **overrides: str) -> Chunk:
    base = {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.split("#")[0],
        "title": "Test Doc",
        "section": "Test Section",
        "category": "test",
        "source_path": f"{chunk_id.split('#')[0]}.md",
        "text": text,
    }
    base.update(overrides)
    return Chunk(**base)


async def test_embed_corpus_persists_all_chunks_with_metadata(tmp_path: Path) -> None:
    chunks = [
        _chunk("a#s1", "first text", title="Alpha", section="S1", category="x"),
        _chunk("a#s2", "second text", title="Alpha", section="S2", category="x"),
    ]
    collection = open_collection(chroma_dir=tmp_path, name="test_persist")
    embedder = FakeEmbedder()

    count = await embed_corpus(chunks, embedder=embedder, collection=collection)

    assert count == 2
    assert collection.count() == 2
    got = collection.get(ids=["a#s1"], include=["embeddings", "documents", "metadatas"])
    assert got["documents"][0] == "first text"
    assert got["metadatas"][0] == {
        "doc_id": "a",
        "title": "Alpha",
        "section": "S1",
        "category": "x",
        "source_path": "a.md",
    }
    assert list(got["embeddings"][0]) == pytest.approx(_vec("first text", FakeEmbedder.dim))


async def test_embed_corpus_empty_is_noop(tmp_path: Path) -> None:
    collection = open_collection(chroma_dir=tmp_path, name="test_empty")
    embedder = FakeEmbedder()

    count = await embed_corpus([], embedder=embedder, collection=collection)

    assert count == 0
    assert collection.count() == 0
    assert embedder.calls == []


async def test_embed_corpus_is_idempotent_on_rerun(tmp_path: Path) -> None:
    chunks = [_chunk(f"d#s{i}", f"body {i}") for i in range(5)]
    collection = open_collection(chroma_dir=tmp_path, name="test_idempotent")

    await embed_corpus(chunks, embedder=FakeEmbedder(), collection=collection)
    assert collection.count() == 5

    # Rewriting the same IDs must not duplicate — upsert semantics.
    await embed_corpus(chunks, embedder=FakeEmbedder(), collection=collection)
    assert collection.count() == 5


async def test_embed_corpus_updates_changed_chunk_text(tmp_path: Path) -> None:
    collection = open_collection(chroma_dir=tmp_path, name="test_updates")

    original = [_chunk("d#s1", "original body")]
    await embed_corpus(original, embedder=FakeEmbedder(), collection=collection)

    edited = [_chunk("d#s1", "rewritten body")]
    await embed_corpus(edited, embedder=FakeEmbedder(), collection=collection)

    got = collection.get(ids=["d#s1"], include=["embeddings", "documents"])
    assert got["documents"][0] == "rewritten body"
    assert list(got["embeddings"][0]) == pytest.approx(_vec("rewritten body", FakeEmbedder.dim))


async def test_embed_corpus_raises_on_length_mismatch(tmp_path: Path) -> None:
    collection = open_collection(chroma_dir=tmp_path, name="test_mismatch")
    chunks = [_chunk("d#s1", "a"), _chunk("d#s2", "b")]

    with pytest.raises(ValueError, match="1 vectors for 2 chunks"):
        await embed_corpus(chunks, embedder=WrongLengthEmbedder(), collection=collection)


async def test_embed_corpus_batches_in_multiple_calls(tmp_path: Path) -> None:
    collection = open_collection(chroma_dir=tmp_path, name="test_batches")
    chunks = [_chunk(f"d#s{i}", f"body {i}") for i in range(5)]
    embedder = FakeEmbedder()

    await embed_corpus(chunks, embedder=embedder, collection=collection, batch_size=2)

    # 5 items, batch_size=2 → batches of 2, 2, 1
    assert [len(c) for c in embedder.calls] == [2, 2, 1]
    assert collection.count() == 5


async def test_embed_corpus_rejects_zero_batch_size(tmp_path: Path) -> None:
    collection = open_collection(chroma_dir=tmp_path, name="test_bad_batch")
    with pytest.raises(ValueError, match="batch_size"):
        await embed_corpus(
            [_chunk("d#s1", "x")],
            embedder=FakeEmbedder(),
            collection=collection,
            batch_size=0,
        )


async def test_open_collection_is_reopenable(tmp_path: Path) -> None:
    col_a = open_collection(chroma_dir=tmp_path, name="test_reopen")
    await embed_corpus(
        [_chunk("d#s1", "persisted")],
        embedder=FakeEmbedder(),
        collection=col_a,
    )
    assert col_a.count() == 1

    # Re-opening at the same path must see the previously-written row.
    col_b = open_collection(chroma_dir=tmp_path, name="test_reopen")
    assert col_b.count() == 1
    got = col_b.get(ids=["d#s1"], include=["documents"])
    assert got["documents"][0] == "persisted"


async def test_embed_corpus_on_real_chunks(tmp_path: Path) -> None:
    """Full round-trip against the actual corpus — catches schema drift
    between the loader and the pipeline without requiring a live Ollama."""
    collection = open_collection(chroma_dir=tmp_path, name="test_real_corpus")
    chunks = load_chunks()
    embedder = FakeEmbedder()

    count = await embed_corpus(chunks, embedder=embedder, collection=collection)

    assert count == len(chunks)
    assert collection.count() == len(chunks)
    sample = collection.get(ids=[chunks[0].chunk_id], include=["metadatas"])
    meta = sample["metadatas"][0]
    assert set(meta.keys()) == {"doc_id", "title", "section", "category", "source_path"}


def test_default_collection_name_is_slug_safe() -> None:
    # Chroma rejects collection names outside [a-zA-Z0-9._-].
    assert DEFAULT_COLLECTION.replace("_", "").isalnum()
