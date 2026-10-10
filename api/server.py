"""FastAPI app: a thin HTTP wrapper around `harness.loop.run_turn`.

Process-wide state is bundled into a `Components` dataclass and built once
during the lifespan:

  * `ProviderRouter`   — name → ChatProvider, populated from `Settings`.
  * `Embedder` / `chromadb.Collection` — optional legacy support resources,
    created only when support tools are explicitly enabled.
  * `FactStore`        — long-term personalization memory. sqlite3 binds the
    connection to its opening thread, so opening it on the event-loop
    thread (the lifespan's caller) keeps every async handler within reach
    without `asyncio.to_thread`.
  * `Grounder`         — confidence-scoring heuristic.
  * `sessions`         — live conversations for this process only.
  * `session_store`    — SQLite review archives; never restored for execution.

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

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from harness.cancellation import TurnCancellation
from harness.grounding import Grounder
from harness.policy import classify_task, is_out_of_scope_request
from harness.providers import build_configured_provider, configured_model, configured_provider_names
from harness.router import ProviderNotFoundError, ProviderRouter
from harness.runs import RequestConflictError, RequestId, RunInput, RunRecord
from harness.runtime import build_registry, out_of_scope_response, run_configured_turn
from harness.session_store import SessionArchive, SessionStore, SessionSummary
from harness.state import Session, TurnResponse
from harness.stream import ErrorEvent, EventCallback, StreamEvent, TurnDoneEvent
from memory import FactStore
from providers import create_embedder
from providers.base import ChatProvider, Embedder
from tools import ToolRegistry

from .models import (
    CancelSessionRequest,
    CancelSessionResponse,
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    SubmitRunRequest,
)
from .settings import Settings, get_settings

if TYPE_CHECKING:
    import chromadb

log = logging.getLogger(__name__)


@dataclass
class Components:
    """Bundle of process-wide objects the request handlers need."""

    providers: dict[str, ChatProvider]
    router: ProviderRouter
    embedder: Embedder | None
    collection: chromadb.Collection | None
    fact_store: FactStore
    grounder: Grounder
    sessions: dict[str, Session] = field(default_factory=dict)
    session_store: SessionStore | None = None
    cancellations: dict[str, TurnCancellation] = field(default_factory=dict)
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    closing: bool = False
    run_tasks: dict[str, asyncio.Task[TurnResponse]] = field(default_factory=dict)


ComponentsFactory = Callable[[Settings], Components]


def build_components(settings: Settings) -> Components:
    """Default factory: open real backends from the validated settings."""
    providers = _build_providers(settings)
    embedder = None
    collection = None
    if settings.enable_support_tools:
        from data.embed import open_collection

        collection = open_collection(chroma_dir=settings.chroma_path)
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
        collection=collection,
        fact_store=FactStore(settings.memory_db_path),
        grounder=Grounder(escalation_threshold=settings.confidence_escalation_threshold),
    )


def _build_providers(settings: Settings) -> dict[str, ChatProvider]:
    providers = {
        name: build_configured_provider(name, settings)
        for name in configured_provider_names(settings)
    }
    if settings.default_provider not in providers:
        raise RuntimeError(
            f"DEFAULT_PROVIDER={settings.default_provider!r} is not configured "
            f"(available: {sorted(providers)}). Set the matching API key or pick a "
            "different default."
        )
    return providers


async def _close_components(components: Components) -> None:
    # Keep stores/providers alive until detached turns finish cancellation and review.
    components.closing = True
    for cancellation in components.cancellations.values():
        cancellation.request()
    if components.tasks:
        await asyncio.gather(*components.tasks, return_exceptions=True)
    for provider in components.providers.values():
        aclose = getattr(provider, "aclose", None)
        if aclose is not None:
            await aclose()
    embedder_close = getattr(components.embedder, "aclose", None)
    if embedder_close is not None:
        await embedder_close()
    components.fact_store.close()
    if components.session_store is not None:
        components.session_store.close()


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
            if resolved_settings.enable_support_tools and (
                components.collection is None or components.embedder is None
            ):
                raise RuntimeError("Support tools require a collection and embedder at startup")
            if components.session_store is None:
                components.session_store = SessionStore(resolved_settings.session_db_path)
            components.session_store.recover_runs()
            yield
        finally:
            await _close_components(components)

    app = FastAPI(title="agent-harness", version="0.1.0", lifespan=lifespan)
    app.state.settings = resolved_settings
    _register_routes(app)
    _mount_ui(app)
    return app


# Mounted last so the explicit API routes (and FastAPI's /docs, /openapi.json)
# resolve first; the catch-all "/" mount only serves the demo panel and assets.
UI_DIR = Path(__file__).resolve().parent.parent / "ui"


def _mount_ui(app: FastAPI) -> None:
    """Serve the static demo panel at `/` when the `ui/` directory is present.

    The panel is a thin HTTP client (POST /sessions, POST /chat) — no agent
    logic lives in the frontend. Absent `ui/`, the API runs headless.
    """
    if UI_DIR.is_dir():
        app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")


def _register_routes(app: FastAPI) -> None:
    @app.post("/sessions", response_model=CreateSessionResponse)
    async def create_session(req: CreateSessionRequest, request: Request) -> CreateSessionResponse:
        components: Components = request.app.state.components
        settings: Settings = request.app.state.settings
        workspace_root = _resolve_workspace_root(
            req.workspace_root or _default_workspace_root(settings)
        )
        session = Session(user_id=req.user_id, workspace_root=workspace_root)
        _session_store(components).save(session)
        components.sessions[session.session_id] = session
        log.info(
            "api=create_session user_id=%s session_id=%s workspace_root=%s",
            req.user_id,
            session.session_id,
            workspace_root,
        )
        return CreateSessionResponse(session_id=session.session_id)

    @app.get("/sessions", response_model=list[SessionSummary])
    async def list_sessions(
        request: Request,
        user_id: str = Query(min_length=1),
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> list[SessionSummary]:
        """List committed review snapshots for this user (newest first)."""
        return _session_store(request.app.state.components).list(
            user_id, limit=limit, offset=offset
        )

    @app.get("/sessions/{session_id}", response_model=SessionArchive)
    async def inspect_session(
        session_id: str, request: Request, user_id: str = Query(min_length=1)
    ) -> SessionArchive:
        """Inspect saved transcript and final evidence; this does not enable resume."""
        return _lookup_archive(request.app.state.components, session_id, user_id)

    @app.post(
        "/sessions/{session_id}/cancel", response_model=CancelSessionResponse, status_code=202
    )
    async def cancel_session(
        session_id: str, req: CancelSessionRequest, request: Request
    ) -> CancelSessionResponse:
        """Request cancellation; 202 does not mean cleanup or review has finished."""
        components: Components = request.app.state.components
        _lookup_session(components, session_id, req.user_id)
        cancellation = components.cancellations.get(session_id)
        if cancellation is None or not cancellation.request():
            raise HTTPException(status_code=409, detail="No cancellable turn; idle or finalizing.")
        return CancelSessionResponse()

    @app.post("/runs", response_model=RunRecord)
    async def submit_run(req: SubmitRunRequest, request: Request, response: Response) -> RunRecord:
        """Submit once, then inspect by run ID without keeping a transport open."""
        record, _ = _start_run(request.app.state.components, request.app.state.settings, req)
        response.status_code = 202 if record.status == "running" else 200
        return record

    @app.get("/runs", response_model=list[RunRecord])
    async def list_runs(
        request: Request,
        user_id: str = Query(min_length=1),
        session_id: str | None = Query(default=None, min_length=1),
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> list[RunRecord]:
        return _session_store(request.app.state.components).list_runs(
            user_id,
            session_id=session_id,
            limit=limit,
            offset=offset,
        )

    @app.get("/runs/{run_id}", response_model=RunRecord)
    async def inspect_run(
        run_id: str,
        request: Request,
        user_id: str = Query(min_length=1),
    ) -> RunRecord:
        return _lookup_run(request.app.state.components, run_id, user_id)

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, request: Request) -> ChatResponse:
        components: Components = request.app.state.components
        record, task = _start_run(components, request.app.state.settings, req)
        return await _wait_run(record, task)

    @app.get("/chat/stream")
    async def chat_stream(
        request: Request,
        user_id: str = Query(min_length=1),
        session_id: str = Query(min_length=1),
        message: str = Query(min_length=1),
        provider: str | None = None,
        request_id: RequestId | None = None,
    ) -> StreamingResponse:
        """SSE variant of /chat — emits tool_start/tool_end as the loop runs.

        EventSource only issues GET, so the turn inputs ride as query params.
        Session/ownership/provider failures surface as real HTTP status codes
        before the stream opens; failures *during* the turn arrive as an
        `error` SSE event since the response status is already committed.
        """
        components: Components = request.app.state.components
        settings: Settings = request.app.state.settings

        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()

        async def on_event(event: StreamEvent) -> None:
            await queue.put(event)

        req = ChatRequest(
            user_id=user_id,
            session_id=session_id,
            message=message,
            provider=provider,
            request_id=request_id,
        )
        record, task = _start_run(components, settings, req, on_event=on_event)

        async def runner() -> None:
            try:
                result = await _wait_run(record, task)
                await queue.put(TurnDoneEvent(response=result))
            except Exception as exc:
                log.error("api=chat_stream failed exception=%s", type(exc).__name__)
                await queue.put(
                    ErrorEvent(detail=f"Run {record.run_id} failed; inspect /runs/{record.run_id}")
                )
            finally:
                await queue.put(None)

        async def event_gen() -> AsyncIterator[str]:
            # A duplicate stream waits for final evidence; live events are not replayed.
            _own_task(components, asyncio.create_task(runner()))
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield _sse(event)

        return StreamingResponse(event_gen(), media_type="text/event-stream")


def _lookup_session(components: Components, session_id: str, user_id: str) -> Session:
    """Fetch a session, enforcing existence (404) and ownership (403)."""
    session = components.sessions.get(session_id)
    if session is None:
        _lookup_archive(components, session_id, user_id)
        raise HTTPException(
            status_code=409,
            detail=(
                "Saved session is read-only after restart or interrupted execution. "
                "Create a new session; resume is not supported."
            ),
        )
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Session does not belong to this user_id")
    return session


def _session_store(components: Components) -> SessionStore:
    assert components.session_store is not None
    return components.session_store


def _lookup_archive(components: Components, session_id: str, user_id: str) -> SessionArchive:
    archive = _session_store(components).get(session_id)
    if archive is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    if archive.session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Session does not belong to this user_id")
    return archive


def _lookup_run(components: Components, run_id: str, user_id: str) -> RunRecord:
    record = _session_store(components).get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    if record.request.user_id != user_id:
        raise HTTPException(status_code=403, detail="Run does not belong to this user_id")
    return record


def _start_run(
    components: Components,
    settings: Settings,
    req: ChatRequest,
    *,
    on_event: EventCallback | None = None,
) -> tuple[RunRecord, asyncio.Task[TurnResponse] | None]:
    """Bind request identity durably before scheduling any execution (no awaits)."""
    store = _session_store(components)
    run_input = RunInput(
        user_id=req.user_id,
        session_id=req.session_id,
        message=req.message,
        provider=req.provider,
        request_id=req.request_id or uuid.uuid4().hex,
    )
    existing = store.find_run(req.user_id, run_input.request_id)
    if existing is not None:
        if existing.request != run_input:
            raise HTTPException(
                status_code=409, detail="request_id already belongs to different input"
            )
        return existing, components.run_tasks.get(existing.run_id)
    if components.closing:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    session = _lookup_session(components, req.session_id, req.user_id)
    policy = is_out_of_scope_request(req.message)
    provider = None if policy else _resolve_provider_or_400(components, req.provider)
    model = None
    if provider is not None:
        with suppress(ValueError):
            model = configured_model(provider.name, settings)
    try:
        record, created = store.create_run(
            run_input,
            provider=provider.name if provider else "policy",
            configured_model=model,
        )
    except RequestConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Run storage unavailable; execution not started"
        ) from exc
    if not created:
        return record, components.run_tasks.get(record.run_id)

    log.info(
        "api=run run_id=%s user_id=%s session_id=%s task_kind=%s",
        record.run_id,
        req.user_id,
        req.session_id,
        classify_task(req.message),
    )

    async def execute() -> TurnResponse:
        try:
            if provider is None:
                result = out_of_scope_response()
            else:
                result = await _run_configured_turn(
                    components=components,
                    settings=settings,
                    session=session,
                    user_id=req.user_id,
                    message=req.message,
                    provider=provider,
                    on_event=on_event,
                    run_id=record.run_id,
                )
            result.run_id = record.run_id
            # Busy/scope rejections have no finalized Turn, but still have a run result.
            saved = store.get_run(record.run_id)
            if saved is not None and saved.status == "running":
                store.finish_run(record.run_id, result)
            return result
        except BaseException:
            try:
                store.fail_run(record.run_id)
            except Exception as exc:
                log.error("api=run_failure_storage exception=%s", type(exc).__name__)
            raise

    task = asyncio.create_task(execute())
    components.run_tasks[record.run_id] = task
    task.add_done_callback(lambda _: components.run_tasks.pop(record.run_id, None))
    _own_task(components, task)
    return record, task


async def _wait_run(
    record: RunRecord,
    task: asyncio.Task[TurnResponse] | None,
) -> TurnResponse:
    if task is not None:
        return await asyncio.shield(task)
    if record.response is not None:
        return record.response
    raise HTTPException(
        status_code=409,
        detail={
            "run_id": record.run_id,
            "status": record.status,
            "message": record.error
            or "Run has no active execution in this process; inspect before new work.",
        },
    )


def _resolve_provider_or_400(components: Components, provider_name: str | None) -> ChatProvider:
    try:
        return components.router.resolve(provider_name)
    except ProviderNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _sse(event: StreamEvent) -> str:
    """Format one Server-Sent Event frame: a named event + JSON data line."""
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


def _own_task(components: Components, task: asyncio.Task[Any]) -> None:
    components.tasks.add(task)

    def settled(task: asyncio.Task[Any]) -> None:
        components.tasks.discard(task)
        if not task.cancelled():
            task.exception()  # Retrieve failures even when the transport disconnected.

    task.add_done_callback(settled)


async def _run_configured_turn(
    *,
    components: Components,
    settings: Settings,
    session: Session,
    user_id: str,
    message: str,
    provider: ChatProvider,
    on_event: EventCallback | None = None,
    run_id: str | None = None,
) -> TurnResponse:
    cancellation = TurnCancellation()
    if components.closing:
        cancellation.request()
    owns_signal = session.session_id not in components.cancellations
    if owns_signal:
        components.cancellations[session.session_id] = cancellation

    async def execute() -> TurnResponse:
        try:
            response = await _execute_turn(
                components=components,
                settings=settings,
                session=session,
                user_id=user_id,
                message=message,
                provider=provider,
                on_event=on_event,
                cancellation=cancellation,
                run_id=run_id,
            )
            if response.completion_status == "cancelled":
                components.sessions.pop(session.session_id, None)
            return response
        finally:
            cancellation.close()
            if owns_signal:
                components.cancellations.pop(session.session_id, None)

    task = asyncio.create_task(execute())
    _own_task(components, task)
    return await asyncio.shield(task)


async def _execute_turn(
    *,
    components: Components,
    settings: Settings,
    session: Session,
    user_id: str,
    message: str,
    provider: ChatProvider,
    on_event: EventCallback | None = None,
    cancellation: TurnCancellation,
    run_id: str | None = None,
) -> TurnResponse:
    """Refresh injected facts, build the per-request registry, and run the loop."""
    support_registry = ToolRegistry()
    if settings.enable_support_tools:
        from tools.rag import register_rag_tool
        from tools.sql import register_sql_tools

        if components.collection is None or components.embedder is None:
            raise RuntimeError("Support tools require a collection and embedder at startup")
        register_sql_tools(support_registry, db_path=settings.sqlite_db_path)
        register_rag_tool(
            support_registry, collection=components.collection, embedder=components.embedder
        )
    registry = build_registry(
        fact_store=components.fact_store,
        user_id=user_id,
        workspace_root=session.workspace_root,
        support_tools=support_registry,
        coding_toolset=settings.coding_toolset,
    )
    try:
        model = configured_model(provider.name, settings)
    except ValueError:
        model = None  # Custom test/factory providers may have no configured model.

    def save_finalized(session: Session, response: TurnResponse) -> None:
        try:
            if run_id is None:
                _session_store(components).save(session, response, configured_model=model)
            else:
                response.run_id = run_id
                _session_store(components).finish_run(run_id, response, session=session)
        except Exception as exc:
            # The workspace may already have changed. Do not let a retry silently
            # continue from state whose final evidence failed to reach storage.
            components.sessions.pop(session.session_id, None)
            log.error("api=archive_failed exception=%s", type(exc).__name__)
            raise HTTPException(
                status_code=503,
                detail=(
                    "Session archive failed; workspace edits may exist. "
                    "Only the previous saved snapshot is available for review. "
                    "Create a new session after inspecting the workspace."
                ),
            ) from exc

    try:
        return await run_configured_turn(
            settings=settings,
            session=session,
            user_id=user_id,
            message=message,
            provider=provider,
            fact_store=components.fact_store,
            registry=registry,
            grounder=components.grounder,
            on_event=on_event,
            on_finalized=save_finalized,
            cancellation=cancellation,
        )
    except BaseException:
        # Failed/interrupted execution can leave an unfinished transcript. Keep
        # the committed review snapshot, but never archive or reuse that state.
        components.sessions.pop(session.session_id, None)
        raise


def _default_workspace_root(settings: Settings) -> str | None:
    if settings.default_workspace_root is None:
        return None
    return str(settings.default_workspace_root)


def _resolve_workspace_root(raw: str | None) -> str | None:
    """Resolve an optional client path to an absolute directory for the session."""
    if raw is None:
        return None
    resolved = Path(raw).expanduser().resolve()
    if not resolved.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"workspace_root is not a directory: {resolved}",
        )
    return str(resolved)


app = create_app()
