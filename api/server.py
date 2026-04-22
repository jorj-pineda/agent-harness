"""FastAPI app: a thin HTTP wrapper around `harness.loop.run_turn`.

Process-wide state is bundled into a `Components` dataclass and built once
during the lifespan:

  * `ProviderRouter`   — name → ChatProvider, populated from `Settings`.
  * `Embedder`         — single Ollama embedding client shared by RAG.
  * `chromadb.Collection` — the persistent corpus collection.
  * `FactStore`        — long-term personalization memory. sqlite3 binds the
    connection to its opening thread, so opening it on the event-loop
    thread (the lifespan's caller) keeps every async handler within reach
    without `asyncio.to_thread`.
  * `Grounder`         — confidence-scoring heuristic.
  * `sessions`         — in-memory `dict[session_id, Session]`. Persistence
    is deferred (see commit message for the trade-off).

A `ToolRegistry` is built **per request** so the memory tools can close over
the request's `user_id`. That closure is the only structural barrier
preventing one user's turn from reading another user's facts (per the
contract in `tools/memory.py`); rebuilding the registry on every call keeps
that closure honest.

`create_app(components_factory=...)` lets tests inject fakes without ever
opening a real provider — the lifespan hands the factory the validated
`Settings` and consumes whatever it returns.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import chromadb
from fastapi import FastAPI, HTTPException, Request

from data.embed import open_collection
from harness.grounding import Grounder
from harness.loop import run_turn
from harness.router import ProviderNotFoundError, ProviderRouter
from harness.state import Session
from memory import FactStore
from providers import create_chat_provider, create_embedder
from providers.base import ChatMessage, ChatProvider, Embedder
from tools import ToolRegistry
from tools.memory import register_memory_tools
from tools.rag import register_rag_tool
from tools.sql import register_sql_tools

from .models import ChatRequest, ChatResponse, CreateSessionRequest, CreateSessionResponse
from .settings import Settings, get_settings

log = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = (
    "You are a customer-support assistant. Use the available tools to look up "
    "customer, order, and product data; search the support documentation for "
    "policy questions; and remember durable facts about the user across sessions. "
    "Cite documentation chunks you draw from when answering policy questions. "
    "Decline politely if a question is outside the support scope."
)


@dataclass
class Components:
    """Bundle of process-wide objects the request handlers need."""

    providers: dict[str, ChatProvider]
    router: ProviderRouter
    embedder: Embedder
    collection: chromadb.Collection
    fact_store: FactStore
    grounder: Grounder
    sessions: dict[str, Session] = field(default_factory=dict)


ComponentsFactory = Callable[[Settings], Components]


def build_components(settings: Settings) -> Components:
    """Default factory: open real backends from the validated settings."""
    providers = _build_providers(settings)
    embedder = create_embedder(
        "ollama",
        host=settings.ollama_host,
        model=settings.ollama_model,
        embed_model=settings.ollama_embed_model,
        timeout_seconds=float(settings.request_timeout_seconds),
    )
    settings.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
    return Components(
        providers=providers,
        router=ProviderRouter(providers, default=settings.default_provider),
        embedder=embedder,
        collection=open_collection(chroma_dir=settings.chroma_path),
        fact_store=FactStore(settings.memory_db_path),
        grounder=Grounder(escalation_threshold=settings.confidence_escalation_threshold),
    )


def _build_providers(settings: Settings) -> dict[str, ChatProvider]:
    providers: dict[str, ChatProvider] = {
        "ollama": create_chat_provider(
            "ollama",
            host=settings.ollama_host,
            model=settings.ollama_model,
            embed_model=settings.ollama_embed_model,
            timeout_seconds=float(settings.request_timeout_seconds),
        )
    }
    if settings.anthropic_api_key:
        providers["anthropic"] = create_chat_provider(
            "anthropic",
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            timeout_seconds=float(settings.request_timeout_seconds),
        )
    if settings.openai_api_key:
        providers["openai"] = create_chat_provider(
            "openai",
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout_seconds=float(settings.request_timeout_seconds),
        )
    if settings.default_provider not in providers:
        raise RuntimeError(
            f"DEFAULT_PROVIDER={settings.default_provider!r} is not configured "
            f"(available: {sorted(providers)}). Set the matching API key or pick a "
            "different default."
        )
    return providers


async def _close_components(components: Components) -> None:
    for provider in components.providers.values():
        aclose = getattr(provider, "aclose", None)
        if aclose is not None:
            await aclose()
    embedder_close = getattr(components.embedder, "aclose", None)
    if embedder_close is not None:
        await embedder_close()
    components.fact_store.close()


def create_app(
    *,
    settings: Settings | None = None,
    components_factory: ComponentsFactory = build_components,
) -> FastAPI:
    """Build the FastAPI app. Tests pass a factory that returns fake components."""
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(level=resolved_settings.log_level.upper())
        components = components_factory(resolved_settings)
        app.state.components = components
        try:
            yield
        finally:
            await _close_components(components)

    app = FastAPI(title="agent-harness", version="0.1.0", lifespan=lifespan)
    app.state.settings = resolved_settings
    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    @app.post("/sessions", response_model=CreateSessionResponse)
    async def create_session(req: CreateSessionRequest, request: Request) -> CreateSessionResponse:
        components: Components = request.app.state.components
        session = Session(user_id=req.user_id)
        components.sessions[session.session_id] = session
        log.info(
            "api=create_session user_id=%s session_id=%s",
            req.user_id,
            session.session_id,
        )
        return CreateSessionResponse(session_id=session.session_id)

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, request: Request) -> ChatResponse:
        components: Components = request.app.state.components
        settings: Settings = request.app.state.settings

        session = components.sessions.get(req.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown session_id")
        if session.user_id != req.user_id:
            raise HTTPException(status_code=403, detail="Session does not belong to this user_id")

        _refresh_facts_system_message(session, components.fact_store, req.user_id)

        registry = _build_registry(components=components, settings=settings, user_id=req.user_id)

        try:
            provider = components.router.resolve(req.provider)
        except ProviderNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return await run_turn(
            session=session,
            user_input=req.message,
            provider=provider,
            registry=registry,
            max_iterations=settings.max_tool_iterations,
            grounder=components.grounder,
        )


def _build_registry(
    *,
    components: Components,
    settings: Settings,
    user_id: str,
) -> ToolRegistry:
    registry = ToolRegistry()
    register_sql_tools(registry, db_path=settings.sqlite_db_path)
    register_rag_tool(registry, collection=components.collection, embedder=components.embedder)
    register_memory_tools(registry, store=components.fact_store, user_id=user_id)
    return registry


def _refresh_facts_system_message(
    session: Session,
    fact_store: FactStore,
    user_id: str,
) -> None:
    """Replace (or insert) the facts system message at index 0.

    Rebuilt every turn so a `remember_fact` call surfaces on the next turn.
    `format_for_system_prompt` returns "" when the user has no facts, so the
    concatenation stays unconditional.
    """
    facts_block = fact_store.format_for_system_prompt(user_id)
    content = BASE_SYSTEM_PROMPT + (f"\n\n{facts_block}" if facts_block else "")
    message = ChatMessage(role="system", content=content)
    if session.messages and session.messages[0].role == "system":
        session.messages[0] = message
    else:
        session.messages.insert(0, message)


app = create_app()
