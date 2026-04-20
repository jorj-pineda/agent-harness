"""API-layer settings — loaded from `.env` via `pydantic-settings`.

The harness keeps its own narrower `harness/config.py:Settings`. The API
carries a few extra concerns (per-provider model overrides, the memory DB
path) and gets its own settings object so the HTTP surface can evolve
without rippling into the harness layer.

Secrets live in `.env` (gitignored). `.env.example` is the committed
template — copy it to `.env` and fill in.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Providers
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_host: str = "http://localhost:11434"
    default_provider: str = "ollama"

    # Models
    ollama_model: str = "gemma4"
    ollama_embed_model: str = "nomic-embed-text"
    anthropic_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-4o-mini"

    # Data paths
    sqlite_db_path: Path = Path("data/support.db")
    chroma_path: Path = Path("data/chroma")
    memory_db_path: Path = Path("data/memory.db")

    # Harness budgets
    max_tool_iterations: int = Field(default=8, ge=1, le=32)
    request_timeout_seconds: int = Field(default=60, ge=1, le=600)

    # Grounding
    confidence_escalation_threshold: float = Field(default=0.55, ge=0.0, le=1.0)

    # Logging
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor — pydantic-settings reads `.env` once and stays put."""
    return Settings()
