"""Allowlisted configured execution evidence, never a resumable model profile."""

from __future__ import annotations

import hashlib
import json
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from memory.store import MAX_MEMORY_PROMPT_BYTES
from providers import compatible_transport_options
from tools import ToolRegistry

from .config import CodingToolset, Settings
from .model_profiles import ModelProfileIdentity
from .prompts import coding_prompt, coding_prompt_version
from .providers import configured_model, provider_endpoint
from .repository_instructions import RepositoryInstructionReview


class CapturedRuntimeSettings(BaseModel):
    """Explicit allowlist: adding a Settings field never automatically persists it."""

    model_config = ConfigDict(extra="forbid")

    coding_toolset: CodingToolset
    load_root_agents: bool = False
    enable_support_tools: bool
    max_tool_iterations: int
    max_tool_calls_per_turn: int
    max_turn_wall_seconds: float
    max_completion_tokens_per_turn: int
    # Older configured runtimes did not have a per-request cap.
    max_request_output_tokens: int = Field(default=0, ge=0, le=1_000_000)
    max_total_tokens_per_turn: int
    max_context_tokens: int
    min_request_output_tokens: int
    # Before this field existed, the implementation always used three bytes per token.
    prompt_estimate_bytes_per_token: float = Field(default=3, ge=1, le=8, allow_inf_nan=False)
    max_identical_tool_calls: int
    max_completion_retries: int
    request_timeout_seconds: int
    require_verification_before_finish: bool
    require_plan_before_edit: bool
    max_files_touched_per_turn: int
    track_workspace_changes: bool
    max_tracked_files: int
    max_tracked_bytes: int
    max_workspace_diff_bytes: int
    max_diff_file_bytes: int
    max_diff_snapshot_bytes: int
    confidence_escalation_threshold: float


class ExecutionProfile(BaseModel):
    schema_version: Literal[1] = 1
    source: Literal["configured_settings"] = "configured_settings"
    provider: str
    configured_model: str | None
    model_profile: ModelProfileIdentity | None = None
    # Before inspection sessions existed, configured sessions used the standard policy.
    inspection_only: bool = False
    repository_instructions: RepositoryInstructionReview | None = None
    endpoint_sha256: str | None
    prompt_version: str | None
    base_prompt_sha256: str
    tool_specs_sha256: str
    tool_names: list[str]
    project_check_sha256: str | None
    provider_options_sha256: str | None = None
    memory_prompt_max_bytes: int | None = Field(default=None, ge=1)
    memory_scope_version: Literal["workspace-path-v1"] | None = None
    settings: CapturedRuntimeSettings
    context_limit_policy: Literal["configured-min-v1"] | None = None
    effective_max_context_tokens: int | None = Field(default=None, ge=0)
    ollama_num_ctx: int | None = None
    ollama_think: bool | None = None
    scope_note: str = (
        "Configured execution evidence, not backend attestation or resume permission. "
        "Excludes credentials, raw endpoints, check argv, injected facts, workspace contents, "
        "tool implementations, and provider/server defaults. Null identities are unknown."
        " A special compatible transport recipe hash records public configured options only; "
        "null has no recorded recipe. Conversation IDs and constructor overrides are excluded."
        " Memory prompt byte limits are recorded; null means the boundary was not recorded."
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _json_digest(value: object) -> str:
    return _digest(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def _compatible_recipe_digest(settings: Settings) -> str | None:
    options = compatible_transport_options(
        settings.openai_compatible_transport,
        replay_reasoning_content=settings.openai_compatible_replay_reasoning_content,
    )
    if settings.openai_compatible_replay_reasoning_content:
        options["continuation_policy"] = "openai-reasoning-content-v1:65536-utf8-bytes"
    return _json_digest(options) if options else None


def _endpoint_digest(provider: str, settings: Settings, model: str | None) -> str | None:
    if model is None:
        return None
    endpoint = provider_endpoint(provider, settings)
    if provider in {"ollama", "openai_compatible"}:
        try:
            parsed = urlsplit(endpoint)
            # Accessing port also rejects invalid/non-numeric/out-of-range ports.
            _ = parsed.port
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                return None
        except ValueError:
            return None
    return _digest(endpoint)


def capture_execution_profile(
    settings: Settings,
    provider: str,
    registry: ToolRegistry,
    *,
    system_prompt: str | None = None,
    repository_instructions: RepositoryInstructionReview | None = None,
    inspection_only: bool = False,
) -> ExecutionProfile:
    """Capture settings and exposed schemas; never serialize the full Settings object."""
    try:
        model = configured_model(provider, settings)
    except ValueError:
        model = None
    specs = registry.as_tool_specs()
    return ExecutionProfile(
        provider=provider,
        configured_model=model,
        model_profile=settings.profile_identity(),
        inspection_only=inspection_only,
        repository_instructions=repository_instructions,
        endpoint_sha256=_endpoint_digest(provider, settings, model),
        prompt_version=coding_prompt_version(
            settings.coding_toolset, settings.coding_prompt_profile
        )
        if system_prompt is None
        else None,
        base_prompt_sha256=_digest(
            coding_prompt(settings.coding_toolset, settings.coding_prompt_profile)
            if system_prompt is None
            else system_prompt
        ),
        tool_specs_sha256=_json_digest([spec.model_dump(mode="json") for spec in specs]),
        tool_names=[spec.name for spec in specs],
        project_check_sha256=None
        if settings.project_check_argv is None
        else _json_digest(settings.project_check_argv),
        provider_options_sha256=_compatible_recipe_digest(settings)
        if provider == "openai_compatible"
        else None,
        memory_prompt_max_bytes=MAX_MEMORY_PROMPT_BYTES,
        memory_scope_version="workspace-path-v1",
        settings=CapturedRuntimeSettings.model_validate(
            {name: getattr(settings, name) for name in CapturedRuntimeSettings.model_fields}
        ),
        context_limit_policy="configured-min-v1",
        effective_max_context_tokens=settings.context_limit_for(provider),
        ollama_num_ctx=settings.ollama_num_ctx if provider == "ollama" else None,
        ollama_think=settings.ollama_think if provider == "ollama" else None,
    )
