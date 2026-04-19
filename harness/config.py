from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    ollama_host: str = "http://localhost:11434"
    default_provider: str = "ollama"

    ollama_model: str = "gemma4"
    ollama_embed_model: str = "nomic-embed-text"

    sqlite_db_path: Path = Path("data/support.db")
    chroma_path: Path = Path("data/chroma")

    max_tool_iterations: int = Field(default=8, ge=1, le=32)
    request_timeout_seconds: int = Field(default=60, ge=1, le=600)

    confidence_escalation_threshold: float = Field(default=0.55, ge=0.0, le=1.0)

    log_level: str = "INFO"


def get_settings() -> Settings:
    return Settings()
