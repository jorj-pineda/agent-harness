"""Shared application and evaluation settings. Secrets are loaded from .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .outcome import is_verification_command

CodingToolset = Literal["full", "whole_file"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Providers
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openai_compatible_api_key: str | None = None
    openai_compatible_base_url: str | None = None
    ollama_host: str = "http://localhost:11434"
    default_provider: str = "ollama"

    # Models
    ollama_model: str = "gemma4"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_num_ctx: int | None = Field(
        default=None,
        ge=256,
        le=1_048_576,
        description="Context length requested per Ollama chat; unset uses the server default.",
    )
    ollama_think: bool | None = Field(
        default=None,
        description="Enable or disable model thinking on Ollama; unset uses the model default.",
    )
    anthropic_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-4o-mini"
    openai_compatible_model: str | None = Field(default=None, validate_default=True)

    @field_validator("openai_compatible_base_url", "openai_compatible_model", mode="before")
    @classmethod
    def empty_compatible_setting(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("openai_compatible_base_url")
    @classmethod
    def validate_compatible_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "OpenAI-compatible base URL must be HTTP(S) without credentials or query"
            )
        return value.rstrip("/")

    @field_validator("openai_compatible_model")
    @classmethod
    def validate_compatible_pair(cls, value: str | None, info: ValidationInfo) -> str | None:
        if bool(info.data.get("openai_compatible_base_url")) != bool(value):
            raise ValueError("OpenAI-compatible base URL and model must be configured together")
        return value

    # Data paths
    sqlite_db_path: Path = Path("data/support.db")
    chroma_path: Path = Path("data/chroma")
    memory_db_path: Path = Path("data/memory.db")

    # Harness budgets
    max_tool_iterations: int = Field(default=8, ge=1, le=32)
    max_tool_calls_per_turn: int = Field(default=24, ge=1, le=256)
    max_turn_wall_seconds: float = Field(default=0, ge=0, le=3600)
    max_completion_tokens_per_turn: int = Field(default=0, ge=0, le=1000000)
    max_total_tokens_per_turn: int = Field(
        default=0,
        ge=0,
        le=100_000_000,
        description="Reported prompt + output tokens across a turn, checked with estimates.",
    )
    max_context_tokens: int = Field(
        default=0,
        ge=0,
        le=10_000_000,
        description="Model context window; estimated prompt + output must fit (0 disables).",
    )
    min_request_output_tokens: int = Field(
        default=256,
        ge=1,
        le=100_000,
        description="Stop instead of requesting fewer output tokens under context/total limits.",
    )
    max_identical_tool_calls: int = Field(default=2, ge=0, le=10)
    max_completion_retries: int = Field(default=1, ge=0, le=3)
    request_timeout_seconds: int = Field(default=60, ge=1, le=600)
    project_check_argv: list[str] | None = Field(
        default=None,
        description="Exact allowlisted command that must pass after the latest edit.",
    )

    @field_validator("project_check_argv")
    @classmethod
    def validate_project_check(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value or any(not token or "\n" in token or "\x00" in token for token in value):
            raise ValueError("Project check must be a non-empty single-line argv list")
        if not is_verification_command(value):
            raise ValueError("Project check must be a pytest, ruff check, or mypy invocation")
        return value

    # Grounding
    confidence_escalation_threshold: float = Field(default=0.55, ge=0.0, le=1.0)

    # Coding agent (Phase 4+)
    coding_toolset: CodingToolset = Field(
        default="full",
        description="whole_file omits replace_text; all other tools remain unchanged.",
    )
    require_verification_before_finish: bool = Field(
        default=False,
        description="Require the configured check after file-tool edits before completion.",
    )
    enable_support_tools: bool = Field(
        default=False,
        description="Open legacy SQL/RAG resources and tools (requires seeded support DB/corpus).",
    )
    max_files_touched_per_turn: int = Field(
        default=5,
        ge=0,
        le=50,
        description="Block edits to additional distinct files at this limit (0 disables).",
    )
    require_plan_before_edit: bool = Field(
        default=False,
        description="Block file-tool edits until emit_plan succeeds in this turn.",
    )
    track_workspace_changes: bool = Field(
        default=True,
        description="Snapshot the workspace around each turn to report actual file changes.",
    )
    max_tracked_files: int = Field(default=20_000, ge=1, le=1_000_000)
    max_tracked_bytes: int = Field(default=256_000_000, ge=1, le=10_000_000_000)
    max_workspace_diff_bytes: int = Field(
        default=64_000,
        ge=0,
        le=10_000_000,
        description="Maximum UTF-8 diff content per turn; 0 disables text capture.",
    )
    max_diff_file_bytes: int = Field(default=128_000, ge=1, le=10_000_000)
    max_diff_snapshot_bytes: int = Field(default=2_000_000, ge=1, le=100_000_000)
    default_workspace_root: Path | None = Field(
        default=None,
        description="Optional default repo sandbox when sessions omit workspace_root.",
    )

    # Logging
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor — pydantic-settings reads `.env` once and stays put."""
    return Settings()
