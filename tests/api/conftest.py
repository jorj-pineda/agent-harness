"""Fixtures for the API end-to-end tests.

The API server's `create_app(components_factory=...)` seam lets tests inject
fake provider/embedder instances without spinning up Ollama, Anthropic, or
OpenAI. The fixtures here:

  * Build a real `FactStore` and a real (empty) Chroma collection on
    `tmp_path` so memory + RAG wiring runs end-to-end.
  * Plug a `ScriptedProvider` whose responses tests script per-test, with
    every `chat()` call recorded for cross-session assertions.
  * Hand back a `TestClient` whose lifespan opens (and closes) the real
    components — sqlite3's thread-binding means the FactStore connection
    is opened on the same loop the request handlers run on.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

from api.server import Components, create_app
from api.settings import Settings
from data.embed import open_collection
from harness.grounding import Grounder
from harness.router import ProviderRouter
from memory import FactStore
from providers.base import (
    ChatMessage,
    ChatProvider,
    Embedder,
    FinishReason,
    ProviderResponse,
    ToolCall,
    ToolSpec,
)


class ScriptedProvider:
    """Pops pre-canned ProviderResponses; records every call for assertions."""

    name = "scripted"

    def __init__(self) -> None:
        self._queue: list[ProviderResponse] = []
        self.calls: list[tuple[list[ChatMessage], list[ToolSpec] | None]] = []

    def script(self, *responses: ProviderResponse) -> None:
        self._queue.extend(responses)

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        self.calls.append(([m.model_copy(deep=True) for m in messages], tools))
        assert self._queue, "ScriptedProvider ran out of scripted responses"
        return self._queue.pop(0)


class FakeEmbedder:
    """Deterministic hash embedder — same shape as tests/test_tools_rag.py."""

    name = "fake"
    dim = 8

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_vec(t, self.dim) for t in texts]


def _vec(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[i % len(digest)] / 255.0 for i in range(dim)]


def make_response(
    *,
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
) -> ProviderResponse:
    return ProviderResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=cast(FinishReason, finish_reason),
        model="scripted-model",
        latency_ms=1.0,
    )


@dataclass
class Harness:
    """Bundle returned by the `harness` fixture so tests can poke at internals.

    Note: `FactStore` is not exposed here. sqlite3 binds a connection to the
    thread that opened it, and `TestClient` opens the lifespan on a worker
    thread (Starlette's portal). Assertions about persisted facts therefore
    go through the HTTP layer — a follow-up request reads what the previous
    request wrote, both on the same lifespan thread.
    """

    client: TestClient
    provider: ScriptedProvider


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    """Build a TestClient backed by fake provider + real FactStore + empty Chroma.

    The SQL tool DB path points at a non-existent file: the tool only opens
    it on invocation, and these tests never script a SQL tool call.
    """
    provider = ScriptedProvider()
    fake_embedder = FakeEmbedder()

    settings = Settings(
        default_provider="scripted",
        sqlite_db_path=tmp_path / "support.db",
        chroma_path=tmp_path / "chroma",
        memory_db_path=tmp_path / "memory.db",
        confidence_escalation_threshold=0.55,
        max_tool_iterations=8,
    )

    def _factory(_: Settings) -> Components:
        providers: dict[str, ChatProvider] = {"scripted": provider}
        embedder: Embedder = fake_embedder
        collection = open_collection(chroma_dir=tmp_path / "chroma", name="test_api")
        fact_store = FactStore(tmp_path / "memory.db")
        return Components(
            providers=providers,
            router=ProviderRouter(providers, default="scripted"),
            embedder=embedder,
            collection=collection,
            fact_store=fact_store,
            grounder=Grounder(escalation_threshold=settings.confidence_escalation_threshold),
        )

    app = create_app(settings=settings, components_factory=_factory)
    with TestClient(app) as client:
        yield Harness(client=client, provider=provider)


@pytest.fixture
def make_session(harness: Harness) -> Callable[[str], str]:
    """Helper: POST /sessions and return the new session_id."""

    def _create(user_id: str) -> str:
        resp = harness.client.post("/sessions", json={"user_id": user_id})
        assert resp.status_code == 200, resp.text
        return cast(str, resp.json()["session_id"])

    return _create
