"""Read-only configuration drift and estimated context review, never resume permission."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from memory import FactStore
from providers.base import ChatMessage
from tools import ToolRegistry

from .config import Settings
from .context import BudgetStop, TokenBudget
from .execution_profile import ExecutionProfile, capture_execution_profile
from .prompts import coding_prompt
from .runtime import refresh_system_message
from .session_store import SessionArchive
from .transcript import TranscriptReview, inspect_transcript


class ProfileComparison(BaseModel):
    status: Literal["matching", "changed", "unknown", "not_recorded", "unavailable"]
    changed_fields: list[str] = Field(default_factory=list)
    unknown_fields: list[str] = Field(default_factory=list)
    reason: str | None = None


class ContextReview(BaseModel):
    status: Literal["estimated_fit", "blocked", "unbounded_context", "not_checked"]
    estimated_prompt_tokens: int | None = None
    includes_proposed_message: bool = False
    max_context_tokens: int | None = None
    request_output_allowance: int | None = None
    reason: str | None = None


class ConfigurationReview(BaseModel):
    status: Literal["reviewed", "busy", "unavailable"] = "reviewed"
    turn_id: str | None = None
    provider: str | None = None
    configuration: ProfileComparison | None = None
    transcript: TranscriptReview | None = None
    context: ContextReview | None = None
    reason: str | None = None
    read_only: Literal[True] = True
    resume_supported: Literal[False] = False
    scope_note: str = (
        "Compares captured configured fields and estimates a new turn with current facts, tools, "
        "and the proposed message. Matching is not backend/tool compatibility "
        "or resume permission. "
        "Workspace state, runtime/tool implementations, backend defaults and capabilities, and "
        "unsaved crash work are outside this review. No history repair, inference, or tools run."
    )


def compare_profiles(
    saved: ExecutionProfile | None, current: ExecutionProfile
) -> ProfileComparison:
    if saved is None:
        return ProfileComparison(status="not_recorded", reason="No saved execution configuration.")
    saved_fields = saved.model_dump(exclude={"scope_note"})
    current_fields = current.model_dump(exclude={"scope_note"})
    current_settings = current.settings.model_dump()
    changed: list[str] = []
    for field, value in saved_fields.items():
        if field == "settings":
            changed.extend(
                f"settings.{name}"
                for name, setting in saved.settings.model_dump().items()
                if setting != current_settings[name]
            )
        elif value != current_fields[field]:
            changed.append(field)
    unknown = [
        field
        for field in (
            "configured_model",
            "endpoint_sha256",
            "prompt_version",
            "memory_prompt_max_bytes",
            "memory_scope_version",
        )
        if saved_fields[field] is None or current_fields[field] is None
    ]
    return ProfileComparison(
        status="changed" if changed else "unknown" if unknown else "matching",
        changed_fields=sorted(changed),
        unknown_fields=unknown,
    )


def review_configuration(
    archive: SessionArchive,
    *,
    settings: Settings,
    provider: str,
    registry: ToolRegistry,
    fact_store: FactStore,
    user_id: str,
    message: str,
) -> ConfigurationReview:
    """Use detached history and the same prompt refresh/budget math as configured execution."""
    turns = archive.session.turns
    turn_id = turns[-1].turn_id if turns else None
    transcript = inspect_transcript(archive.session.messages)
    transcript.turn_id = turn_id
    review = ConfigurationReview(turn_id=turn_id, provider=provider, transcript=transcript)
    turn_ids = [turn.turn_id for turn in turns]
    if (
        turn_ids != [item.turn_id for item in archive.responses]
        or len(set(turn_ids)) != len(turn_ids)
        or any(turn.finished_at is None for turn in turns)
    ):
        review.configuration = ProfileComparison(
            status="unavailable", reason="Saved turn/response alignment is inconsistent."
        )
        review.context = ContextReview(status="not_checked", reason="Inconsistent archive.")
        return review
    saved = archive.responses[-1] if archive.responses else None
    profile = saved.execution_profile if saved else None
    if (
        profile is not None
        and saved is not None
        and (
            profile.provider != saved.response.provider
            or profile.configured_model != saved.configured_model
        )
    ):
        review.configuration = ProfileComparison(
            status="unavailable", reason="Saved provider/model identity is inconsistent."
        )
        review.context = ContextReview(status="not_checked", reason="Inconsistent archive.")
        return review
    current = capture_execution_profile(settings, provider, registry)
    review.configuration = compare_profiles(profile, current)
    if transcript.status == "invalid":
        review.context = ContextReview(status="not_checked", reason="Saved transcript is invalid.")
        return review
    detached = archive.session.model_copy(deep=True)
    refresh_system_message(
        detached,
        fact_store,
        user_id,
        project_check_argv=settings.project_check_argv,
        system_prompt=coding_prompt(settings.coding_toolset, settings.coding_prompt_profile),
    )
    if message:
        detached.messages.append(ChatMessage(role="user", content=message))
    specs = registry.as_tool_specs()
    budget = TokenBudget(
        max_output=settings.max_completion_tokens_per_turn,
        max_total=settings.max_total_tokens_per_turn,
        max_context=settings.max_context_tokens,
        min_request_output=settings.min_request_output_tokens,
    )
    allowance = budget.request_allowance(detached.messages, specs)
    review.context = ContextReview(
        status="blocked"
        if isinstance(allowance, BudgetStop)
        else "estimated_fit"
        if settings.max_context_tokens > 0
        else "unbounded_context",
        estimated_prompt_tokens=budget.estimate_prompt(detached.messages, specs),
        includes_proposed_message=bool(message),
        max_context_tokens=settings.max_context_tokens,
        request_output_allowance=None if isinstance(allowance, BudgetStop) else allowance,
        reason=allowance.reason if isinstance(allowance, BudgetStop) else None,
    )
    return review
