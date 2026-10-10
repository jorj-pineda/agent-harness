"""Transport-independent setup shared by the API and model smoke evaluations.

Tool simulation is an explicit callable transformation. It preserves public
schemas and descriptions while allowing evals to replace execution completely.
The low-level loop remains available for historical scripted contract tests.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from memory import FactStore
from providers.base import ChatMessage, ChatProvider
from tools import Tool, ToolRegistry
from tools.code import build_code_tools
from tools.memory import register_memory_tools
from tools.semantic import register_semantic_search_stub
from workspace import SnapshotLimitError, Workspace, compare, snapshot
from workspace.changes import TextCapture, review_diffs
from workspace.core import DEFAULT_IGNORE_GLOBS

from .admission import turn_admission
from .config import CodingToolset, Settings
from .grounding import Grounder
from .loop import run_turn
from .policy import is_out_of_scope_request
from .prompts import BASE_SYSTEM_PROMPT, coding_prompt
from .state import Session, TurnResponse, WorkspaceChangeReport, WorkspaceFileDiff
from .stream import EventCallback
from .trace import TurnTraceRecord


def build_registry(
    *,
    fact_store: FactStore,
    user_id: str,
    workspace_root: str | None,
    support_tools: Iterable[Tool] = (),
    coding_toolset: CodingToolset = "full",
    code_tool_transform: Callable[[Tool], Tool] | None = None,
    command_env: Mapping[str, str] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in support_tools:
        registry.register(tool)
    register_memory_tools(registry, store=fact_store, user_id=user_id)
    if workspace_root is not None:
        code_registry = ToolRegistry()
        for tool in build_code_tools(Workspace(root=Path(workspace_root)), command_env=command_env):
            if coding_toolset == "whole_file" and tool.name == "replace_text":
                continue
            code_registry.register(tool)
        register_semantic_search_stub(code_registry)
        for tool in code_registry:
            registry.register(code_tool_transform(tool) if code_tool_transform else tool)
    return registry


def refresh_system_message(
    session: Session,
    fact_store: FactStore,
    user_id: str,
    *,
    system_prompt: str = BASE_SYSTEM_PROMPT,
    project_check_argv: list[str] | None = None,
) -> None:
    blocks = [system_prompt]
    if project_check_argv is not None:
        blocks.append(
            "After editing, run this project check with run_command argv: "
            f"{json.dumps(project_check_argv)}"
        )
    if session.workspace_root:
        blocks.append(f"Workspace root: {session.workspace_root}")
    facts = fact_store.format_for_system_prompt(user_id)
    if facts:
        blocks.append(facts)
    message = ChatMessage(role="system", content="\n\n".join(blocks))
    if session.messages and session.messages[0].role == "system":
        session.messages[0] = message
    else:
        session.messages.insert(0, message)


def out_of_scope_response() -> TurnResponse:
    return TurnResponse(
        answer=(
            "This request is out of scope for a single agent turn "
            "(unsafe or unbounded). Please narrow the task."
        ),
        escalated=True,
        completion_status="blocked",
        completion_reason="Request is outside the bounded single-turn scope.",
        provider="policy",
        latency_ms=0.0,
    )


async def run_configured_turn(
    *,
    settings: Settings,
    session: Session,
    user_id: str,
    message: str,
    provider: ChatProvider,
    fact_store: FactStore,
    registry: ToolRegistry,
    grounder: Grounder | None = None,
    on_event: EventCallback | None = None,
    system_prompt: str | None = None,
    trace: list[TurnTraceRecord] | None = None,
    on_finalized: Callable[[Session, TurnResponse], None] | None = None,
) -> TurnResponse:
    """Execute a turn, optionally committing final evidence before releasing admission.

    The synchronous finalization callback runs only for actual finalized turns,
    after workspace review. Its failure propagates rather than reporting success.
    """
    if is_out_of_scope_request(message):
        return out_of_scope_response()
    with turn_admission.claim(session.session_id, session.workspace_root) as busy:
        if busy is not None:
            return TurnResponse(
                answer=busy,
                escalated=True,
                completion_status="blocked",
                completion_reason=busy,
                provider="policy",
                latency_ms=0.0,
            )
        refresh_system_message(
            session,
            fact_store,
            user_id,
            system_prompt=system_prompt
            if system_prompt is not None
            else coding_prompt(settings.coding_toolset),
            project_check_argv=settings.project_check_argv,
        )
        tracked_root = (
            Path(session.workspace_root)
            if settings.track_workspace_changes and session.workspace_root
            else None
        )
        before = await _snapshot_or_reason(tracked_root, settings) if tracked_root else None
        response = await run_turn(
            session=session,
            user_input=message,
            provider=provider,
            registry=registry,
            max_iterations=settings.max_tool_iterations,
            grounder=grounder
            or Grounder(escalation_threshold=settings.confidence_escalation_threshold),
            require_verification_before_finish=settings.require_verification_before_finish,
            require_plan_before_edit=settings.require_plan_before_edit,
            max_files_touched_per_turn=settings.max_files_touched_per_turn,
            max_tool_calls_per_turn=settings.max_tool_calls_per_turn,
            max_turn_wall_seconds=settings.max_turn_wall_seconds,
            max_completion_tokens_per_turn=settings.max_completion_tokens_per_turn,
            max_total_tokens_per_turn=settings.max_total_tokens_per_turn,
            max_context_tokens=settings.max_context_tokens,
            min_request_output_tokens=settings.min_request_output_tokens,
            max_identical_tool_calls=settings.max_identical_tool_calls,
            max_completion_retries=settings.max_completion_retries,
            required_check=settings.project_check_argv,
            on_event=on_event,
            trace=trace,
        )
        if tracked_root is not None and before is not None:
            response.workspace_changes = await _change_report(tracked_root, settings, before)
        if on_finalized is not None:
            on_finalized(session, response)
        return response


@dataclass
class _TurnSnapshot:
    fingerprints: dict[str, str]
    text: TextCapture | None


async def _snapshot_or_reason(root: Path, settings: Settings) -> _TurnSnapshot | str:
    text = (
        TextCapture(settings.max_diff_file_bytes, settings.max_diff_snapshot_bytes)
        if settings.max_workspace_diff_bytes
        else None
    )
    try:
        fingerprints = await asyncio.to_thread(
            snapshot,
            root,
            ignore=DEFAULT_IGNORE_GLOBS,
            max_files=settings.max_tracked_files,
            max_bytes=settings.max_tracked_bytes,
            text_capture=text,
        )
        return _TurnSnapshot(fingerprints, text)
    except SnapshotLimitError as exc:
        return f"{exc}; change tracking skipped."
    except OSError as exc:
        return f"Workspace snapshot failed ({type(exc).__name__}); change tracking skipped."


async def _change_report(
    root: Path, settings: Settings, before: _TurnSnapshot | str
) -> WorkspaceChangeReport:
    after = before if isinstance(before, str) else await _snapshot_or_reason(root, settings)
    if isinstance(before, str) or isinstance(after, str):
        return WorkspaceChangeReport(status="unavailable", reason=str(after))
    changes = compare(before.fingerprints, after.fingerprints)
    diffs = (
        await asyncio.to_thread(
            review_diffs,
            changes,
            before.text,
            after.text,
            max_bytes=settings.max_workspace_diff_bytes,
        )
        if before.text is not None and after.text is not None
        else []
    )
    return WorkspaceChangeReport(
        status="tracked",
        added=list(changes.added),
        modified=list(changes.modified),
        deleted=list(changes.deleted),
        diffs=[
            WorkspaceFileDiff(path=item.path, diff=item.diff, reason=item.reason) for item in diffs
        ],
    )
