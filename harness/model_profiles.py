"""Explicit startup model/budget bindings; no credentials, discovery or capability claims."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

ProfileName = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")]
ProfileProvider = Literal["ollama", "anthropic", "openai", "openai_compatible"]
ModelName = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:/@+-]{0,127}$")]


class ModelProfileIdentity(BaseModel):
    schema_version: Literal[1] = 1
    name: ProfileName
    provider: ProfileProvider
    model: ModelName
    definition_sha256: str


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    provider: ProfileProvider
    model: ModelName
    max_context_tokens: int = Field(ge=256, le=1_048_576)
    max_request_output_tokens: int = Field(ge=1, le=1_000_000)
    prompt_estimate_bytes_per_token: float | None = Field(
        default=None, ge=1, le=8, allow_inf_nan=False
    )
    ollama_think: bool | None = None
    max_completion_retries: int | None = Field(default=None, ge=0, le=3)
    max_tool_iterations: int | None = Field(default=None, ge=1, le=32, strict=True)

    @model_validator(mode="after")
    def check_binding(self) -> ModelProfile:
        if self.max_request_output_tokens > self.max_context_tokens:
            raise ValueError("Profile request output cap exceeds its context window")
        if self.provider != "ollama" and self.ollama_think is not None:
            raise ValueError("Ollama thinking is supported only by an Ollama profile")
        return self

    def settings_overrides(self) -> dict[str, object]:
        overrides: dict[str, object] = {
            "default_provider": self.provider,
            f"{self.provider}_model": self.model,
            "max_context_tokens": self.max_context_tokens,
            "max_request_output_tokens": self.max_request_output_tokens,
        }
        if self.provider == "ollama":
            overrides["ollama_num_ctx"] = self.max_context_tokens
        for name in (
            "prompt_estimate_bytes_per_token",
            "ollama_think",
            "max_completion_retries",
            "max_tool_iterations",
        ):
            value = getattr(self, name)
            if value is not None:
                overrides[name] = value
        return overrides

    def identity(self, name: str) -> ModelProfileIdentity:
        canonical = json.dumps(
            self.model_dump(exclude_none=True), sort_keys=True, separators=(",", ":")
        )
        return ModelProfileIdentity(
            name=name,
            provider=self.provider,
            model=self.model,
            definition_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        )
