"""Transport-independent setup shared by the API and model smoke evaluations.

Tool simulation is an explicit callable transformation. It preserves public
schemas and descriptions while allowing evals to replace execution completely.
The low-level loop remains available for historical scripted contract tests.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from memory import FactStore
from providers.base import ChatMessage, ChatProvider
from tools import Tool, ToolRegistry
from tools.code import build_code_tools
from tools.memory import register_memory_tools
from tools.semantic import register_semantic_search_stub
from workspace import Workspace

from .config import Settings
from .grounding import Grounder
from .loop import run_turn
from .policy import is_out_of_scope_request
from .prompts import BASE_SYSTEM_PROMPT
from .state import Session, TurnResponse
from .stream import EventCallback


def build_registry(
    *,
    fact_store: FactStore,
    user_id: str,
    workspace_root: str | None,
    support_tools: Iterable[Tool] = (),
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
            code_registry.register(tool)
        register_semantic_search_stub(code_registry)
        for tool in code_registry:
            registry.register(code_tool_transform(tool) if code_tool_transform else tool)
    return registry


def refresh_system_message(session: Session, fact_store: FactStore, user_id: str) -> None:
    blocks = [BASE_SYSTEM_PROMPT]
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
) -> TurnResponse:
    if is_out_of_scope_request(message):
        return out_of_scope_response()
    refresh_system_message(session, fact_store, user_id)
    return await run_turn(
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
        on_event=on_event,
    )
