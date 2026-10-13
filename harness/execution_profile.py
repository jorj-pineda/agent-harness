"""Allowlisted configured execution evidence, never a resumable model profile."""

from __future__ import annotations

import hashlib
import json
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

from providers import compatible_transport_options
from tools import ToolRegistry

from .config import CodingToolset, Settings
from .prompts import coding_prompt, coding_prompt_version
from .providers import configured_model, provider_endpoint


class CapturedRuntimeSettings(BaseModel):
    """Explicit allowlist: adding a Settings field never automatically persists it."""

    model_config = ConfigDict(extra="forbid")

    context_management: Literal["full", "bounded_reads_v1"] = "full"
    coding_toolset: CodingToolset
    enable_support_tools: bool
    max_tool_iterations: int
    max_tool_calls_per_turn: int
    max_turn_wall_seconds: float
    max_completion_tokens_per_turn: int
    max_total_tokens_per_turn: int
    max_context_tokens: int
    min_request_output_tokens: int
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
    endpoint_sha256: str | None
    prompt_version: str | None
    base_prompt_sha256: str
    tool_specs_sha256: str
    tool_names: list[str]
    project_check_sha256: str | None
    provider_options_sha256: str | None = None
    settings: CapturedRuntimeSettings
    ollama_num_ctx: int | None = None
    ollama_think: bool | None = None
    scope_note: str = (
        "Configured execution evidence, not backend attestation or resume permission. "
        "Excludes credentials, raw endpoints, check argv, injected facts, workspace contents, "
        "tool implementations, and provider/server defaults. Null identities are unknown."
        " A special compatible transport recipe hash records public configured options only; "
        "null has no recorded recipe. Conversation IDs and constructor overrides are excluded."
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _json_digest(value: object) -> str:
    return _digest(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


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
        provider_options_sha256=_json_digest(
            compatible_transport_options(settings.openai_compatible_transport)
        )
        if provider == "openai_compatible" and settings.openai_compatible_transport != "generic"
        else None,
        settings=CapturedRuntimeSettings.model_validate(
            {name: getattr(settings, name) for name in CapturedRuntimeSettings.model_fields}
        ),
        ollama_num_ctx=settings.ollama_num_ctx if provider == "ollama" else None,
        ollama_think=settings.ollama_think if provider == "ollama" else None,
    )
