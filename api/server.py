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
import json
import logging
import sqlite3
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from harness.admission import turn_admission
from harness.cancellation import TurnCancellation
from harness.checkpoints import CheckpointRecord, CheckpointSummary, capture_checkpoint
from harness.configuration_review import ConfigurationReview, review_configuration
from harness.continuation import ContinuationBlocked, check_continuation
from harness.execution_profile import ExecutionProfile
from harness.grounding import Grounder
from harness.policy import classify_task, is_out_of_scope_request
from harness.providers import build_configured_provider, configured_model, configured_provider_names
from harness.reverts import FileRevertInput, FileRevertOutcome, FileRevertRecord, revert_file
from harness.router import ProviderNotFoundError, ProviderRouter
from harness.runs import RequestConflictError, RequestId, RunInput, RunRecord
from harness.runtime import build_registry, out_of_scope_response, run_configured_turn
from harness.session_store import SessionArchive, SessionStore, SessionSummary
from harness.state import ContinuationSource, Session, TurnResponse
from harness.stream import ErrorEvent, EventCallback, StreamEvent, TurnDoneEvent
from harness.tool_progress import SavedToolEvent, ToolProgress
from harness.transcript import TranscriptReview, inspect_transcript
from harness.workspace_review import WorkspaceReview, WorkspaceReviewBaseline, compare_workspace
from memory import FactStore
from memory.review import MemoryPage, review_notes
from memory.store import FactConflictError
from providers import create_embedder
from providers.base import ChatProvider, Embedder
from tools import ToolRegistry
from workspace.core import WorkspaceError

from .models import (
    CancelSessionRequest,
    CancelSessionResponse,
    ChatRequest,
    ChatResponse,
    ConfigurationReviewRequest,
    ContinueSessionRequest,
    CreateCheckpointRequest,
    CreateSessionRequest,
    CreateSessionResponse,
    DeleteMemoryRequest,
    DeleteMemoryResponse,
    RevertFileRequest,
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
            components.session_store.recover_file_reverts()
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
    @app.post("/sessions/{session_id}/continue", response_model=RunRecord, status_code=202)
    async def continue_session(
        session_id: str, req: ContinueSessionRequest, request: Request
    ) -> RunRecord:
        """Fork saved history; execution rechecks preconditions inside admission."""
        components: Components = request.app.state.components
        settings: Settings = request.app.state.settings
        source = ContinuationSource(
            session_id=session_id,
            turn_id=req.expected_turn_id,
            acknowledge_uncaptured_state=req.acknowledge_uncaptured_state,
        )
        fork_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            json.dumps(
                [
                    "agent-harness-continuation",
                    req.user_id,
                    session_id,
                    req.expected_turn_id,
                    req.request_id,
                ]
            ),
        ).hex
        chat_req = ChatRequest(
            user_id=req.user_id,
            session_id=fork_id,
            message=req.message,
            provider=req.provider,
            request_id=req.request_id,
        )
        expected_input = RunInput(**chat_req.model_dump(), continuation_source=source)
        store = _session_store(components)
        existing = store.find_run(req.user_id, req.request_id)
        if existing is not None:
            if existing.request != expected_input:
                raise HTTPException(
                    status_code=409, detail="Continuation key belongs to different input."
                )
            return existing
        archive = _lookup_archive(components, session_id, req.user_id)
        if not archive.session.turns or archive.session.turns[-1].turn_id != req.expected_turn_id:
            raise HTTPException(
                status_code=409, detail="Source turn is not the latest archived turn."
            )
        fork = archive.session.model_copy(deep=True)
        fork.session_id = fork_id
        fork.created_at = datetime.now(UTC)
        fork.checkpoint_turns = True
        fork.continuation_source = source
        try:
            components.sessions[fork_id] = fork
            record, _ = _start_run(
                components, settings, chat_req, continuation_source=source, fork_archive=archive
            )
            return record
        except BaseException:
            components.sessions.pop(fork_id, None)
            raise

    @app.get("/file-reverts", response_model=FileRevertRecord | None)
    async def find_revert(
        request: Request, user_id: str = Query(min_length=1), request_id: RequestId = Query()
    ) -> FileRevertRecord | None:
        return _session_store(request.app.state.components).find_file_revert(user_id, request_id)

    @app.get("/file-reverts/{revert_id}", response_model=FileRevertRecord)
    async def inspect_revert(
        revert_id: str, request: Request, user_id: str = Query(min_length=1)
    ) -> FileRevertRecord:
        record = _session_store(request.app.state.components).get_file_revert(revert_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Revert not found.")
        if record.request.user_id != user_id:
            raise HTTPException(status_code=403, detail="Revert belongs to another user.")
        return record

    @app.post("/runs/{run_id}/revert-file", response_model=FileRevertRecord)
    async def apply_file_revert(
        run_id: str, req: RevertFileRequest, request: Request, response: Response
    ) -> FileRevertRecord:
        components: Components = request.app.state.components
        run = _lookup_run(components, run_id, req.user_id)
        submission = FileRevertInput(
            run_id=run_id, user_id=req.user_id, path=req.path, request_id=req.request_id
        )
        store = _session_store(components)

        def replay() -> FileRevertRecord | None:
            existing = store.find_file_revert(req.user_id, req.request_id)
            if existing is not None and existing.request != submission:
                raise HTTPException(
                    status_code=409, detail="Revert key belongs to different input."
                )
            return existing

        def deliver(record: FileRevertRecord) -> FileRevertRecord:
            response.status_code = {
                "pending": 202,
                "applied": 200,
                "rejected": 409,
                "interrupted": 409,
                "uncertain": 503,
            }[record.status]
            return record

        existing = replay()
        if existing is not None:
            return deliver(existing)
        if components.closing or run.status == "running":
            raise HTTPException(status_code=409, detail="Run/server is not ready for file revert.")
        before = store.get_checkpoint(run.before_checkpoint_id or "")
        after = store.get_checkpoint(run.after_checkpoint_id or "")
        if (
            before is None
            or after is None
            or before.kind != "turn_before"
            or after.kind != "turn_after"
            or after.paired_before_id != before.checkpoint_id
            or before.run_id != run_id
            or after.run_id != run_id
            or before.session_id != run.request.session_id
            or after.session_id != run.request.session_id
        ):
            raise HTTPException(
                status_code=409, detail="No valid turn checkpoint pair for this run."
            )
        if before.files.get(req.path) == after.files.get(req.path):
            raise HTTPException(
                status_code=409, detail="No eligible checkpoint difference for this path."
            )

        async def apply_and_save() -> FileRevertRecord:
            with turn_admission.claim(run.request.session_id, before.root) as busy:
                if busy is not None:
                    raise HTTPException(status_code=409, detail=busy)
                existing = replay()
                if existing is not None:
                    return existing
                try:
                    content = store.checkpoint_content(before, req.path)
                    store.checkpoint_content(after, req.path)  # Validate saved after evidence too.
                    record = store.create_file_revert(
                        submission, before.checkpoint_id, after.checkpoint_id
                    )
                except (ValueError, sqlite3.Error) as exc:
                    raise HTTPException(
                        status_code=503,
                        detail="Revert preparation/storage failed; no file operation started.",
                    ) from exc
                try:
                    outcome = await asyncio.to_thread(revert_file, before, after, req.path, content)
                except Exception:
                    outcome = FileRevertOutcome(
                        "uncertain", "File operation failed unexpectedly; inspect file and journal."
                    )
                if outcome.status in ("applied", "uncertain"):
                    components.sessions.pop(run.request.session_id, None)
                try:
                    store.finish_file_revert(record, outcome)
                except (ValueError, sqlite3.Error) as exc:
                    raise HTTPException(
                        status_code=503,
                        detail="Revert outcome storage failed; inspect journal and file. "
                        "Do not automatically retry.",
                    ) from exc
                return record

        task = asyncio.create_task(apply_and_save())
        _own_task(components, task)
        return deliver(await asyncio.shield(task))

    @app.get("/memory", response_model=MemoryPage)
    async def inspect_memory(
        request: Request,
        user_id: str = Query(min_length=1),
        limit: int = Query(default=20, ge=1, le=50),
        before_id: int | None = Query(default=None, ge=1),
    ) -> MemoryPage:
        """Inspect bounded note previews without inference, tools or submissions."""
        return review_notes(
            request.app.state.components.fact_store, user_id, limit=limit, before_id=before_id
        )

    @app.delete("/memory/{fact_id}", response_model=DeleteMemoryResponse)
    async def delete_memory(
        fact_id: int, req: DeleteMemoryRequest, request: Request
    ) -> DeleteMemoryResponse:
        """Delete this owner's note only if its content hash still matches the review."""
        if fact_id < 1:
            raise HTTPException(status_code=422, detail="fact_id must be positive.")
        try:
            deleted = request.app.state.components.fact_store.delete(
                req.user_id, fact_id, req.expected_fact_sha256
            )
        except FactConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except sqlite3.Error as exc:
            raise HTTPException(
                status_code=503, detail="Memory deletion failed; inspect memory before retrying."
            ) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="No matching owner-scoped note.")
        return DeleteMemoryResponse(fact_id=fact_id)

    @app.post("/sessions", response_model=CreateSessionResponse)
    async def create_session(req: CreateSessionRequest, request: Request) -> CreateSessionResponse:
        components: Components = request.app.state.components
        settings: Settings = request.app.state.settings
        workspace_root = _resolve_workspace_root(
            req.workspace_root or _default_workspace_root(settings)
        )
        if req.checkpoint_turns and workspace_root is None:
            raise HTTPException(status_code=400, detail="Turn checkpoints require a workspace.")
        session = Session(
            user_id=req.user_id,
            workspace_root=workspace_root,
            checkpoint_turns=req.checkpoint_turns,
        )
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

    @app.get("/sessions/{session_id}/transcript-review", response_model=TranscriptReview)
    async def review_transcript(
        session_id: str,
        request: Request,
        user_id: str = Query(min_length=1),
        max_issues: int = Query(default=100, ge=1, le=1000),
    ) -> TranscriptReview:
        """Inspect committed history structure without repair, inference, or resume."""
        archive = _lookup_archive(request.app.state.components, session_id, user_id)
        review = inspect_transcript(archive.session.messages, max_issues=max_issues)
        review.turn_id = archive.session.turns[-1].turn_id if archive.session.turns else None
        return review

    @app.post(
        "/sessions/{session_id}/checkpoints", response_model=CheckpointRecord, status_code=201
    )
    async def create_checkpoint(
        session_id: str, req: CreateCheckpointRequest, request: Request
    ) -> CheckpointRecord:
        """Explicit current file capture, independent of archived turns; never restores work."""
        components: Components = request.app.state.components
        archive = _lookup_archive(components, session_id, req.user_id)
        root = archive.session.workspace_root
        if root is None:
            raise HTTPException(status_code=409, detail="No checkpoint workspace configured.")

        async def capture_and_save() -> CheckpointRecord:
            with turn_admission.claim(session_id, root) as busy:
                if busy is not None:
                    raise HTTPException(status_code=409, detail=busy)
                settings: Settings = request.app.state.settings
                store = _session_store(components)
                try:
                    captured = await asyncio.to_thread(
                        capture_checkpoint,
                        Path(root),
                        session_id,
                        protected_paths=(
                            store.path,
                            components.fact_store.path,
                            settings.chroma_path,
                        ),
                    )
                except (OSError, ValueError) as exc:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Checkpoint capture unavailable ({type(exc).__name__}); "
                        "no checkpoint saved. Check capture limits, links and storage location.",
                    ) from exc
                try:
                    store.save_checkpoint(captured)
                except (sqlite3.Error, ValueError) as exc:
                    raise HTTPException(
                        status_code=503, detail="Checkpoint storage failed; no checkpoint saved."
                    ) from exc
                return captured.record

        task = asyncio.create_task(capture_and_save())
        _own_task(components, task)
        return await asyncio.shield(task)

    @app.get("/sessions/{session_id}/checkpoints", response_model=list[CheckpointSummary])
    async def list_checkpoints(
        session_id: str,
        request: Request,
        user_id: str = Query(min_length=1),
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> list[CheckpointSummary]:
        components: Components = request.app.state.components
        _lookup_archive(components, session_id, user_id)
        return _session_store(components).list_checkpoints(session_id, limit=limit, offset=offset)

    @app.get("/sessions/{session_id}/checkpoints/{checkpoint_id}", response_model=CheckpointRecord)
    async def inspect_checkpoint(
        session_id: str, checkpoint_id: str, request: Request, user_id: str = Query(min_length=1)
    ) -> CheckpointRecord:
        components: Components = request.app.state.components
        _lookup_archive(components, session_id, user_id)
        record = _session_store(components).get_checkpoint(checkpoint_id)
        if record is None or record.session_id != session_id:
            raise HTTPException(status_code=404, detail="Checkpoint not found for this session.")
        return record

    @app.post("/sessions/{session_id}/configuration-review", response_model=ConfigurationReview)
    async def inspect_configuration(
        session_id: str, req: ConfigurationReviewRequest, request: Request
    ) -> ConfigurationReview:
        """Read-only configuration/context computation; never creates or resumes a run."""
        components: Components = request.app.state.components
        archive = _lookup_archive(components, session_id, req.user_id)
        provider = _resolve_provider_or_400(components, req.provider)
        with turn_admission.claim(session_id, archive.session.workspace_root) as busy:
            if busy is not None:
                return ConfigurationReview(status="busy", reason=busy, provider=provider.name)
            settings: Settings = request.app.state.settings
            try:
                registry = _build_turn_registry(components, settings, archive.session, req.user_id)
                return review_configuration(
                    archive,
                    settings=settings,
                    provider=provider.name,
                    registry=registry,
                    fact_store=components.fact_store,
                    user_id=req.user_id,
                    message=req.message,
                )
            except (OSError, ValueError, RuntimeError, WorkspaceError, sqlite3.Error) as exc:
                return ConfigurationReview(
                    status="unavailable",
                    provider=provider.name,
                    reason=f"Configuration review unavailable ({type(exc).__name__}).",
                )

    @app.get("/sessions/{session_id}/workspace-review", response_model=WorkspaceReview)
    async def inspect_workspace(
        session_id: str,
        request: Request,
        user_id: str = Query(min_length=1),
        max_paths: int = Query(default=100, ge=1, le=1000),
    ) -> WorkspaceReview:
        """Compare scoped workspace state to its saved final baseline; never resumes work."""
        components: Components = request.app.state.components
        archive = _lookup_archive(components, session_id, user_id)
        saved = _session_store(components).get_workspace_review(session_id)
        if saved is None:
            return WorkspaceReview(
                status="not_recorded",
                reason="No durable final workspace baseline for this session.",
            )
        if (
            not archive.session.turns
            or saved.turn_id != archive.session.turns[-1].turn_id
            or saved.baseline.root != archive.session.workspace_root
        ):
            return WorkspaceReview(
                status="unavailable",
                reason="Saved baseline does not match the latest archived turn.",
            )

        async def compare_saved() -> WorkspaceReview:
            with turn_admission.claim(session_id, archive.session.workspace_root) as busy:
                if busy is not None:
                    return WorkspaceReview(status="busy", turn_id=saved.turn_id, reason=busy)
                settings: Settings = request.app.state.settings
                comparison = await asyncio.to_thread(
                    compare_workspace,
                    saved.baseline,
                    max_files=settings.max_tracked_files,
                    max_bytes=settings.max_tracked_bytes,
                    max_paths=max_paths,
                )
                comparison.turn_id = saved.turn_id
                return comparison

        task = asyncio.create_task(compare_saved())
        _own_task(components, task)
        return await asyncio.shield(task)

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
        request_id: RequestId | None = None,
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> list[RunRecord]:
        return _session_store(request.app.state.components).list_runs(
            user_id,
            session_id=session_id,
            request_id=request_id,
            limit=limit,
            offset=offset,
        )

    @app.get("/runs/{run_id}/tool-progress", response_model=ToolProgress)
    async def inspect_tool_progress(
        run_id: str, request: Request, user_id: str = Query(min_length=1)
    ) -> ToolProgress:
        """Read committed tool metadata only; never replays or repairs execution."""
        components: Components = request.app.state.components
        _lookup_run(components, run_id, user_id)
        return _session_store(components).get_tool_progress(run_id)

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
                "Saved session is read-only after restart, interrupted execution "
                "or file reversal. Create a new session or request explicit guarded continuation."
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
    continuation_source: ContinuationSource | None = None,
    fork_archive: SessionArchive | None = None,
) -> tuple[RunRecord, asyncio.Task[TurnResponse] | None]:
    """Bind request identity durably before scheduling any execution (no awaits)."""
    store = _session_store(components)
    run_input = RunInput(
        user_id=req.user_id,
        session_id=req.session_id,
        message=req.message,
        provider=req.provider,
        request_id=req.request_id or uuid.uuid4().hex,
        continuation_source=continuation_source,
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
            fork=(fork_archive, session) if fork_archive else None,
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
            if session.continuation_source is not None and result.completion_status == "blocked":
                components.sessions.pop(session.session_id, None)
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
            if response.completion_status == "cancelled" or (
                session.continuation_source is not None and response.completion_status == "blocked"
            ):
                components.sessions.pop(session.session_id, None)
            return response
        finally:
            cancellation.close()
            if owns_signal:
                components.cancellations.pop(session.session_id, None)

    task = asyncio.create_task(execute())
    _own_task(components, task)
    return await asyncio.shield(task)


def _build_turn_registry(
    components: Components, settings: Settings, session: Session, user_id: str
) -> ToolRegistry:
    """Construct current tool definitions without invoking them."""
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
    return build_registry(
        fact_store=components.fact_store,
        user_id=user_id,
        workspace_root=session.workspace_root,
        support_tools=support_registry,
        coding_toolset=settings.coding_toolset,
    )


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
    registry = _build_turn_registry(components, settings, session, user_id)
    try:
        model = configured_model(provider.name, settings)
    except ValueError:
        model = None  # Custom test/factory providers may have no configured model.

    workspace_baseline: WorkspaceReviewBaseline | None = None
    event_sequence = 0
    known_tools = {spec.name for spec in registry.as_tool_specs()}

    async def save_tool_progress(event: StreamEvent) -> None:
        nonlocal event_sequence
        if run_id is not None and event.type in ("tool_start", "tool_end"):
            event_sequence += 1
            metadata = SavedToolEvent(
                sequence=event_sequence,
                type=event.type,
                tool=event.tool
                if event.tool in known_tools and len(event.tool) <= 128
                else "unknown_tool",
                error_observed=event.error is not None if event.type == "tool_end" else None,
                latency_ms=event.latency_ms if event.type == "tool_end" else None,
            )
            try:
                _session_store(components).record_tool_event(run_id, metadata)
            except Exception as exc:
                log.error("api=tool_progress_failed exception=%s", type(exc).__name__)
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Tool progress storage failed; workspace edits may exist. "
                        "Execution stopped; inspect saved metadata and workspace before new work."
                    ),
                ) from exc
        if on_event is not None:
            await on_event(event)

    def save_execution_profile(profile: ExecutionProfile) -> None:
        if run_id is None:
            return
        try:
            _session_store(components).record_execution_profile(run_id, profile)
        except Exception as exc:
            log.error("api=execution_profile_failed exception=%s", type(exc).__name__)
            raise HTTPException(
                status_code=503,
                detail="Execution configuration storage failed; no inference or tools started.",
            ) from exc

    def capture_workspace_review(baseline: WorkspaceReviewBaseline) -> None:
        nonlocal workspace_baseline
        workspace_baseline = baseline

    async def save_turn_checkpoint(phase: Literal["before", "after"]) -> None:
        if run_id is None or session.workspace_root is None:
            raise HTTPException(status_code=503, detail="Turn checkpoint identity unavailable.")
        root = Path(session.workspace_root)

        async def capture_and_bind() -> None:
            store = _session_store(components)
            try:
                captured = await asyncio.to_thread(
                    capture_checkpoint,
                    root,
                    session.session_id,
                    protected_paths=(store.path, components.fact_store.path, settings.chroma_path),
                )
                if phase == "before" and session.continuation_source is not None:
                    archive = _lookup_archive(components, session.session_id, user_id)
                    saved = archive.responses[-1] if archive.responses else None
                    previous_run = (
                        store.get_run(saved.response.run_id)
                        if saved and saved.response.run_id
                        else None
                    )
                    expected = (
                        store.get_checkpoint(previous_run.after_checkpoint_id or "")
                        if previous_run
                        else None
                    )
                    if (
                        previous_run is None
                        or expected is None
                        or previous_run.status == "running"
                        or previous_run.request.user_id != user_id
                        or expected.kind != "turn_after"
                        or expected.run_id != previous_run.run_id
                        or expected.session_id != previous_run.request.session_id
                        or expected.paired_before_id != previous_run.before_checkpoint_id
                        or previous_run.response is None
                        or saved.response != previous_run.response
                    ):
                        raise ContinuationBlocked(
                            "No valid finalized source checkpoint for continuation."
                        )
                    if previous_run.request.session_id != session.session_id:
                        parent = _lookup_archive(
                            components, session.continuation_source.session_id, user_id
                        )
                        if (
                            not parent.session.turns
                            or parent.session.turns[-1].turn_id
                            != session.continuation_source.turn_id
                        ):
                            raise ContinuationBlocked(
                                "Source archive advanced before continuation admission."
                            )
                    relevant_sessions = {
                        session.session_id,
                        session.continuation_source.session_id,
                        previous_run.request.session_id,
                    }
                    if any(
                        store.has_newer_unfinished_work(
                            user_id,
                            identity,
                            previous_run.created_at,
                            excluding_run_id=run_id,
                            workspace_root=session.workspace_root,
                        )
                        for identity in relevant_sessions
                    ):
                        raise ContinuationBlocked(
                            "Source has newer unfinished/failed work; inspect and start fresh."
                        )
                    check_continuation(
                        archive,
                        expected,
                        captured.record,
                        settings=settings,
                        provider=provider.name,
                        registry=registry,
                        fact_store=components.fact_store,
                        user_id=user_id,
                        message=message,
                    )
                store.bind_run_checkpoint(run_id, captured, phase)
            except ContinuationBlocked as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except (OSError, ValueError, sqlite3.Error) as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"Turn {phase} checkpoint failed; execution stopped. "
                    "Workspace edits may exist; inspect saved run/checkpoint evidence.",
                ) from exc

        task = asyncio.create_task(capture_and_bind())
        _own_task(components, task)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # Keep admission until owned checkpoint work has settled.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()
            raise

    def save_finalized(session: Session, response: TurnResponse) -> None:
        try:
            if run_id is None:
                _session_store(components).save(session, response, configured_model=model)
            else:
                response.run_id = run_id
                _session_store(components).finish_run(
                    run_id, response, session=session, workspace_baseline=workspace_baseline
                )
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
            on_event=save_tool_progress,
            on_finalized=save_finalized,
            on_workspace_review=capture_workspace_review,
            on_execution_profile=save_execution_profile if run_id is not None else None,
            on_checkpoint=save_turn_checkpoint if session.checkpoint_turns else None,
            cancellation=cancellation,
        )
    except HTTPException as exc:
        components.sessions.pop(session.session_id, None)
        if session.continuation_source is not None and exc.status_code == 409:
            reason = str(exc.detail)
            return TurnResponse(
                answer=reason,
                completion_status="blocked",
                completion_reason=reason,
                provider="policy",
                latency_ms=0.0,
            )
        raise
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
