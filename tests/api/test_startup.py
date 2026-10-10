from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from api import server
from harness.config import Settings
from providers.base import ToolCall

from .conftest import FakeEmbedder, ScriptedProvider, make_response


def test_coding_api_import_startup_and_turn_without_support_modules(tmp_path: Path) -> None:
    # Use a fresh interpreter: other test modules import Chroma before collection.
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        from pathlib import Path

        class NoSupport(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'chromadb' or fullname.startswith('chromadb.') or fullname in ('data.embed', 'tools.rag', 'tools.sql'):
                    raise AssertionError('Coding mode imported a support module: ' + fullname)
        sys.meta_path.insert(0, NoSupport())
        from fastapi.testclient import TestClient
        from api import server
        from harness.config import Settings
        from providers.base import ProviderResponse, ToolCall

        class Store(server.FactStore):
            closed = False
            def close(self):
                super().close()
                self.closed = True
        server.FactStore = Store

        root = Path(sys.argv[1])
        repo = root / 'repo'
        repo.mkdir()
        class Provider:
            name = 'scripted'
            closed = False
            calls = 0
            async def chat(self, messages, *, tools=None, **kwargs):
                self.calls += 1
                assert 'search_docs' not in [s.name for s in tools]
                assert 'lookup_customer_by_email' not in [s.name for s in tools]
                if self.calls == 1:
                    return ProviderResponse(content='', tool_calls=[ToolCall(id='edit', name='write_file', arguments={'path':'a.py','content':'a = 1\\n'})], finish_reason='tool_use', model='scripted', latency_ms=0)
                return ProviderResponse(content='Done', finish_reason='stop', model='scripted', latency_ms=0)
            async def aclose(self):
                self.closed = True
        provider = Provider()
        server._build_providers = lambda _: {'scripted': provider}
        def no_embedding(*args, **kwargs):
            raise AssertionError('Coding mode allocated an embedder')
        server.create_embedder = no_embedding
        # Intentionally unusable support storage must not block coding startup.
        support = root / 'support-unavailable'
        support.write_text('not a directory')
        settings = Settings(_env_file=None, default_provider='scripted',
            memory_db_path=root/'memory.db', session_db_path=root/'sessions.db',
            chroma_path=support/'chroma',
            sqlite_db_path=support/'db', ollama_host='http://127.0.0.1:1')
        app = server.create_app(settings=settings)
        with TestClient(app) as client:
            assert app.state.components.embedder is None
            assert app.state.components.collection is None
            sid = client.post('/sessions', json={'user_id':'dev','workspace_root':str(repo)}).json()['session_id']
            result = client.post('/chat', json={'user_id':'dev','session_id':sid,'message':'Add a module'})
            assert result.status_code == 200, result.text
            body = result.json()
            assert body['completion_status'] == 'completed'
            assert body['workspace_changes']['added'] == ['a.py']
            assert '+a = 1\\n' in body['workspace_changes']['diffs'][0]['diff']
            assert client.get('/').status_code == 200
        assert provider.closed and provider.calls == 2
        assert app.state.components.fact_store.closed
        assert not any(m in sys.modules for m in ('chromadb', 'data.embed', 'tools.rag', 'tools.sql'))
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_explicit_support_mode_opens_resources_and_executes_support_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from data.embed import open_collection

    settings = Settings(
        _env_file=None,
        enable_support_tools=True,
        default_provider="scripted",
        memory_db_path=tmp_path / "memory.db",
        session_db_path=tmp_path / "sessions.db",
        chroma_path=tmp_path / "chroma",
        sqlite_db_path=tmp_path / "support.db",
    )
    collection = open_collection(chroma_dir=settings.chroma_path)
    embedder = FakeEmbedder()
    # A single independently seeded support document exercises real retrieval.
    from tests.api.conftest import _vec

    collection.upsert(
        ids=["doc#one"],
        embeddings=[_vec("known support answer", embedder.dim)],
        documents=["known support answer"],
        metadatas=[
            {
                "doc_id": "doc",
                "title": "Title",
                "section": "One",
                "category": "test",
                "source_path": "test.md",
            }
        ],
    )
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        conn.execute(
            "CREATE TABLE customers (id INT, email TEXT, name TEXT, tier TEXT, created_at TEXT)"
        )
        conn.execute(
            "INSERT INTO customers VALUES (1, 'test@example.test', 'Known customer', 'basic', '2026-01-01')"
        )
    embedder.aclose = AsyncMock()
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="rag",
                    name="search_docs",
                    arguments={"query": "known support answer", "k": 1},
                ),
                ToolCall(
                    id="sql",
                    name="lookup_customer_by_email",
                    arguments={"email": "test@example.test"},
                ),
            ]
        ),
        make_response(content="Support results available"),
    )
    monkeypatch.setattr(server, "_build_providers", lambda _: {"scripted": provider})
    created = Mock(return_value=embedder)
    monkeypatch.setattr(server, "create_embedder", created)
    app = server.create_app(settings=settings)
    with TestClient(app) as client:
        assert app.state.components.embedder is embedder
        assert app.state.components.collection is not None
        sid = client.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        response = client.post(
            "/chat",
            json={"user_id": "dev", "session_id": sid, "message": "Look up the support facts"},
        )
        assert response.status_code == 200, response.text
        calls = response.json()["tool_calls"]
        assert all(call["error"] is None for call in calls), calls
        assert calls[0]["result"][0]["text"] == "known support answer"
        assert calls[1]["result"]["name"] == "Known customer"
    created.assert_called_once()
    embedder.aclose.assert_awaited_once()


def test_support_custom_factory_fails_at_startup_and_closes_open_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None,
        default_provider="scripted",
        memory_db_path=tmp_path / "memory.db",
        session_db_path=tmp_path / "sessions.db",
        enable_support_tools=True,
    )
    provider = ScriptedProvider()
    provider.aclose = AsyncMock()
    monkeypatch.setattr(server, "_build_providers", lambda _: {"scripted": provider})

    def factory(_):
        components = server.build_components(
            settings.model_copy(update={"enable_support_tools": False})
        )
        components.fact_store.close = Mock(wraps=components.fact_store.close)
        return components

    app = server.create_app(settings=settings, components_factory=factory)
    with pytest.raises(RuntimeError, match="collection and embedder at startup"), TestClient(app):
        pass
    app.state.components.fact_store.close.assert_called_once()
    provider.aclose.assert_awaited_once()
