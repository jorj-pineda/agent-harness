from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import chromadb
import pytest

from data.corpus import Chunk, load_chunks
from data.embed import embed_corpus, open_collection
from tools import ToolError, ToolRegistry
from tools.rag import MAX_K, build_rag_tool, register_rag_tool


class FakeEmbedder:
    """Deterministic hash embedder — same shape as tests/test_data_embed.py."""

    name = "fake"
    dim = 8

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [_vec(t, self.dim) for t in texts]


def _vec(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[i % len(digest)] / 255.0 for i in range(dim)]


def _chunk(
    chunk_id: str,
    text: str,
    *,
    title: str = "Test Doc",
    section: str = "Test Section",
    category: str = "test",
) -> Chunk:
    doc_id = chunk_id.split("#")[0]
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        title=title,
        section=section,
        category=category,
        source_path=f"{doc_id}.md",
        text=text,
    )


@pytest.fixture
async def seeded_collection(tmp_path: Path) -> chromadb.Collection:
    """Small custom corpus across two categories; gives deterministic hits."""
    collection = open_collection(chroma_dir=tmp_path, name="test_rag")
    chunks = [
        _chunk("returns#window", "Returns accepted within 30 days.", category="returns"),
        _chunk("returns#condition", "Items must be unused.", category="returns"),
        _chunk("shipping#speed", "Standard shipping is 3-5 business days.", category="shipping"),
        _chunk("shipping#intl", "International shipping takes 10-14 days.", category="shipping"),
    ]
    await embed_corpus(chunks, embedder=FakeEmbedder(), collection=collection)
    return collection


@pytest.fixture
async def real_corpus_collection(tmp_path: Path) -> chromadb.Collection:
    """Full embedded corpus — for round-trip tests against real chunk IDs."""
    collection = open_collection(chroma_dir=tmp_path, name="test_rag_real")
    await embed_corpus(load_chunks(), embedder=FakeEmbedder(), collection=collection)
    return collection


@pytest.fixture
def registry(seeded_collection: chromadb.Collection) -> ToolRegistry:
    reg = ToolRegistry()
    register_rag_tool(reg, collection=seeded_collection, embedder=FakeEmbedder())
    return reg


async def test_known_query_surfaces_matching_chunk_in_top_k(
    seeded_collection: chromadb.Collection,
) -> None:
    # Query text matches a chunk verbatim → FakeEmbedder gives identical
    # vectors → distance 0 → that chunk ranks #1 with score ~1.0.
    query = "Returns accepted within 30 days."
    reg = ToolRegistry()
    register_rag_tool(reg, collection=seeded_collection, embedder=FakeEmbedder())

    result = await reg.invoke("search_docs", {"query": query, "k": 2})

    assert result[0]["chunk_id"] == "returns#window"
    assert result[0]["score"] == pytest.approx(1.0, abs=1e-6)
    assert len(result) == 2


async def test_result_carries_full_citation_shape(registry: ToolRegistry) -> None:
    result = await registry.invoke(
        "search_docs", {"query": "Standard shipping is 3-5 business days.", "k": 1}
    )
    assert result[0] == {
        "chunk_id": "shipping#speed",
        "doc_id": "shipping",
        "title": "Test Doc",
        "section": "Test Section",
        "category": "shipping",
        "source_path": "shipping.md",
        "text": "Standard shipping is 3-5 business days.",
        "score": pytest.approx(1.0, abs=1e-6),
    }


async def test_k_caps_result_count(registry: ToolRegistry) -> None:
    result = await registry.invoke("search_docs", {"query": "anything", "k": 3})
    assert len(result) == 3


async def test_default_k_is_4(registry: ToolRegistry) -> None:
    # Collection has 4 chunks; default k=4 returns all.
    result = await registry.invoke("search_docs", {"query": "anything"})
    assert len(result) == 4


async def test_category_filter_narrows_results(registry: ToolRegistry) -> None:
    result = await registry.invoke(
        "search_docs", {"query": "anything", "k": 10, "category": "shipping"}
    )
    assert len(result) == 2
    assert all(r["category"] == "shipping" for r in result)


async def test_category_filter_with_no_matches_returns_empty(registry: ToolRegistry) -> None:
    result = await registry.invoke(
        "search_docs", {"query": "anything", "k": 5, "category": "no-such-category"}
    )
    assert result == []


async def test_k_over_cap_is_rejected(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError, match="Invalid arguments"):
        await registry.invoke("search_docs", {"query": "anything", "k": MAX_K + 1})


async def test_k_below_one_is_rejected(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError, match="Invalid arguments"):
        await registry.invoke("search_docs", {"query": "anything", "k": 0})


async def test_empty_query_is_rejected(registry: ToolRegistry) -> None:
    with pytest.raises(ToolError, match="Invalid arguments"):
        await registry.invoke("search_docs", {"query": ""})


async def test_embedder_is_called_with_the_query(
    seeded_collection: chromadb.Collection,
) -> None:
    embedder = FakeEmbedder()
    reg = ToolRegistry()
    register_rag_tool(reg, collection=seeded_collection, embedder=embedder)

    await reg.invoke("search_docs", {"query": "how do returns work", "k": 1})

    assert embedder.calls == [["how do returns work"]]


async def test_score_is_one_minus_distance(registry: ToolRegistry) -> None:
    # Distinct query text → non-zero distance → score strictly less than 1.
    result = await registry.invoke(
        "search_docs", {"query": "completely unrelated phrase about weather", "k": 4}
    )
    assert len(result) == 4
    for hit in result:
        assert hit["score"] < 1.0
        # Cosine distance in [0, 2] → score in [-1, 1].
        assert -1.0 <= hit["score"] <= 1.0


async def test_empty_collection_returns_empty_list(tmp_path: Path) -> None:
    collection = open_collection(chroma_dir=tmp_path, name="test_rag_empty")
    reg = ToolRegistry()
    register_rag_tool(reg, collection=collection, embedder=FakeEmbedder())

    result = await reg.invoke("search_docs", {"query": "anything", "k": 5})

    assert result == []


async def test_every_call_is_logged(
    registry: ToolRegistry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="tools.rag")
    await registry.invoke(
        "search_docs", {"query": "how long can I return items", "k": 2, "category": "returns"}
    )
    messages = [r.getMessage() for r in caplog.records if r.name == "tools.rag"]
    assert any("rag_tool=search_docs" in m for m in messages)
    assert any("how long can I return items" in m for m in messages)
    assert any("k=2" in m for m in messages)
    assert any("category=returns" in m for m in messages)


def test_tool_produces_a_valid_provider_spec(
    seeded_collection: chromadb.Collection,
) -> None:
    tool = build_rag_tool(collection=seeded_collection, embedder=FakeEmbedder())
    spec = tool.to_spec()
    assert spec.name == "search_docs"
    assert spec.description
    schema = spec.parameters_schema
    assert schema["type"] == "object"
    assert "query" in schema["properties"]
    assert "k" in schema["properties"]
    assert "category" in schema["properties"]


def test_register_rag_tool_exposes_the_tool(
    seeded_collection: chromadb.Collection,
) -> None:
    reg = ToolRegistry()
    register_rag_tool(reg, collection=seeded_collection, embedder=FakeEmbedder())
    assert reg.names() == ["search_docs"]


async def test_round_trip_against_real_corpus(
    real_corpus_collection: chromadb.Collection,
) -> None:
    """Catches any drift between data.embed's metadata shape and what the RAG
    tool expects to read back out."""
    chunks = load_chunks()
    target = chunks[0]
    reg = ToolRegistry()
    register_rag_tool(reg, collection=real_corpus_collection, embedder=FakeEmbedder())

    result = await reg.invoke("search_docs", {"query": target.text, "k": 1})

    assert result[0]["chunk_id"] == target.chunk_id
    assert result[0]["doc_id"] == target.doc_id
    assert result[0]["title"] == target.title
    assert result[0]["section"] == target.section
    assert result[0]["category"] == target.category
    assert result[0]["source_path"] == target.source_path
