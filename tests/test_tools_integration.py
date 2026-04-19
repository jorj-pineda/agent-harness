"""End-to-end smoke: one ToolRegistry, both SQL + RAG tools, dispatch by name.

Proves the two factories compose cleanly on a single registry. Actual wiring
(config → paths → registry) lives in the harness layer (step 6+).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
import pytest

from data.corpus import load_chunks
from data.embed import embed_corpus, open_collection
from data.seed import seed
from data.sqlite import init_schema, open_rw
from tools import ToolRegistry
from tools.rag import register_rag_tool
from tools.sql import register_sql_tools


class FakeEmbedder:
    name = "fake"
    dim = 8

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_vec(t, self.dim) for t in texts]


def _vec(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[i % len(digest)] / 255.0 for i in range(dim)]


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "support.db"
    conn = open_rw(db_path)
    try:
        init_schema(conn)
        seed(conn)
    finally:
        conn.close()
    return db_path


@pytest.fixture
async def seeded_collection(tmp_path: Path) -> chromadb.Collection:
    collection = open_collection(chroma_dir=tmp_path / "chroma", name="test_integration")
    await embed_corpus(load_chunks(), embedder=FakeEmbedder(), collection=collection)
    return collection


@pytest.fixture
def registry(seeded_db: Path, seeded_collection: chromadb.Collection) -> ToolRegistry:
    reg = ToolRegistry()
    register_sql_tools(reg, db_path=seeded_db)
    register_rag_tool(reg, collection=seeded_collection, embedder=FakeEmbedder())
    return reg


def test_all_tools_exposed_on_one_registry(registry: ToolRegistry) -> None:
    assert set(registry.names()) == {
        "lookup_customer_by_email",
        "lookup_order",
        "list_customer_orders",
        "list_customer_tickets",
        "lookup_product",
        "search_docs",
    }


async def test_sql_tool_dispatches_by_name(registry: ToolRegistry) -> None:
    result = await registry.invoke(
        "lookup_customer_by_email",
        {"email": "alice.chen@example.com"},
    )
    assert result is not None
    assert result["email"] == "alice.chen@example.com"


async def test_rag_tool_dispatches_by_name(registry: ToolRegistry) -> None:
    target = load_chunks()[0]
    result = await registry.invoke("search_docs", {"query": target.text, "k": 1})
    assert len(result) == 1
    assert result[0]["chunk_id"] == target.chunk_id


async def test_both_tools_callable_in_sequence(registry: ToolRegistry) -> None:
    """Simulates an agent turn: one SQL lookup, one RAG lookup, same registry."""
    customer = await registry.invoke(
        "lookup_customer_by_email",
        {"email": "alice.chen@example.com"},
    )
    assert customer is not None

    docs = await registry.invoke(
        "search_docs",
        {"query": "how do returns work", "k": 2, "category": "returns"},
    )
    assert len(docs) == 2
    assert all(d["category"] == "returns" for d in docs)
