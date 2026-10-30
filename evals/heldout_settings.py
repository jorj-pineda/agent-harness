"""Evaluation-only validated settings pins; no credentials or launch authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from evals.heldout_guard import _read_regular
from harness.config import Settings
from providers.base import ChatMessage, ChatProvider, ProviderResponse, ToolSpec

if TYPE_CHECKING:
    from evals.heldout_comparison import ComparisonPlan

CREDENTIAL_FIELDS = frozenset({"anthropic_api_key", "openai_api_key", "openai_compatible_api_key"})
# Explicit classification prevents a future credential-bearing field from being
# silently included in artifacts merely because Settings gained a new field.
PUBLIC_FIELDS = frozenset(
    [
        "model_profiles",
        "model_profile",
        "openai_compatible_base_url",
        "ollama_host",
        "default_provider",
        "ollama_model",
        "ollama_embed_model",
        "ollama_num_ctx",
        "ollama_think",
        "anthropic_model",
        "openai_model",
        "openai_compatible_model",
        "openai_compatible_transport",
        "openai_compatible_replay_reasoning_content",
        "sqlite_db_path",
        "chroma_path",
        "memory_db_path",
        "session_db_path",
        "load_root_agents",
        "max_tool_iterations",
        "max_tool_calls_per_turn",
        "max_turn_wall_seconds",
        "max_completion_tokens_per_turn",
        "max_request_output_tokens",
        "max_total_tokens_per_turn",
        "max_context_tokens",
        "prompt_estimate_bytes_per_token",
        "min_request_output_tokens",
        "max_identical_tool_calls",
        "max_completion_retries",
        "request_timeout_seconds",
        "project_check_argv",
        "confidence_escalation_threshold",
        "coding_prompt_profile",
        "coding_toolset",
        "read_file_default_lines",
        "require_verification_before_finish",
        "enable_support_tools",
        "max_files_touched_per_turn",
        "require_plan_before_edit",
        "track_workspace_changes",
        "max_tracked_files",
        "max_tracked_bytes",
        "max_workspace_diff_bytes",
        "max_diff_file_bytes",
        "max_diff_snapshot_bytes",
        "default_workspace_root",
        "log_level",
    ]
)


class ProtocolSettings(Settings):
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Ignore every ambient source, including deliberately named prefix vars.
        return (init_settings,)


def protocol_settings(plan: ComparisonPlan) -> Settings:
    return ProtocolSettings(
        _env_file=None,
        default_provider="ollama",
        ollama_host="http://127.0.0.1:11447",
        ollama_model=plan.model,
        **plan.runtime.model_dump(),
    )


class SettingsPin(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    format: Literal["heldout-validated-settings-v1"] = "heldout-validated-settings-v1"
    credential_policy: Literal["all-unset-excluded"] = "all-unset-excluded"
    values: dict[str, Any] = Field(min_length=1, max_length=100)


def settings_bytes(settings: Settings) -> bytes:
    if (
        set(type(settings).model_fields) != PUBLIC_FIELDS | CREDENTIAL_FIELDS
        or type(settings).model_computed_fields
        or settings.model_extra
    ):
        raise ValueError("Settings schema changed; field classification must be reviewed")
    if any(getattr(settings, field) is not None for field in CREDENTIAL_FIELDS):
        raise ValueError("Held-out settings must not contain configured credentials")
    if (
        settings.ollama_host != "http://127.0.0.1:11447"
        or settings.default_provider != "ollama"
        or settings.openai_compatible_base_url is not None
        or settings.openai_compatible_model is not None
    ):
        raise ValueError("Held-out settings require the fixed local endpoint without hosted config")
    if settings.model_profiles or settings.model_profile is not None:
        raise ValueError("Held-out protocol does not admit model profile overrides")
    values = settings.model_dump(mode="json", exclude=set(CREDENTIAL_FIELDS))
    raw = json.dumps(SettingsPin(values=values).model_dump(), sort_keys=True).encode()
    if len(raw) > 64_000:
        raise ValueError("Settings pin exceeds byte bound")
    return raw


def load_settings_pin(path: Path, *, expected_sha256: str) -> bytes:
    raw, mode = _read_regular(path, 64_000)
    if mode != 0o600 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Settings pin bytes/mode changed")
    SettingsPin.model_validate_json(raw)
    return raw


def check_settings(settings: Settings, expected: bytes) -> None:
    if settings_bytes(settings) != expected:
        raise ValueError("Validated held-out settings differ from capture")


class SettingsPinnedProvider:
    """Compare the actual Settings object before every delegate request; latch failures."""

    def __init__(self, provider: ChatProvider, settings: Settings, expected: bytes) -> None:
        self.provider, self.settings, self.expected = provider, settings, expected
        self.name = provider.name
        self.failed = False

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        if self.failed:
            raise ValueError("Settings-pinned provider is latched closed")
        try:
            check_settings(self.settings, self.expected)
            return await self.provider.chat(
                messages, tools=tools, temperature=temperature, max_tokens=max_tokens
            )
        except BaseException:
            self.failed = True
            raise

    async def aclose(self) -> None:
        self.failed = True
        close = getattr(self.provider, "aclose", None)
        if close is not None:
            await close()
