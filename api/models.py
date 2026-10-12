"""Request/response Pydantic models for the API layer.

Responses ship the rule-#5 envelope from `harness/state.py:TurnResponse`,
re-exported here as `ChatResponse` so the public API surface lives in one
module — callers don't need to know which inner layer owns the schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from harness.reverts import FileRevertInput
from harness.runs import RequestId
from harness.state import TurnResponse


class CreateSessionRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Stable per-user identifier.")
    checkpoint_turns: bool = Field(
        default=False,
        description="Opt in to bounded plaintext workspace captures before/after admitted turns.",
    )
    workspace_root: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional absolute path to the repo sandbox. Resolved and stored on the session; "
            "code tools (Phase 2+) jail all paths under this root."
        ),
    )


class CreateSessionResponse(BaseModel):
    session_id: str


class CreateCheckpointRequest(BaseModel):
    user_id: str = Field(min_length=1)


class RevertFileRequest(BaseModel):
    user_id: str = Field(min_length=1)
    path: str = Field(min_length=1, max_length=4096)
    request_id: RequestId

    @field_validator("path")
    @classmethod
    def canonical_path(cls, value: str) -> str:
        return FileRevertInput.canonical_relative_path(value)


class ChatRequest(BaseModel):
    request_id: RequestId | None = Field(
        default=None,
        description=(
            "Per-user idempotency key. Matching input joins/replays one run; "
            "conflicts return 409. Omit for a distinct run."
        ),
    )
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    provider: str | None = Field(
        default=None,
        description="Optional provider override; falls back to the configured default.",
    )


class SubmitRunRequest(ChatRequest):
    request_id: RequestId


class ConfigurationReviewRequest(BaseModel):
    user_id: str = Field(min_length=1)
    provider: str | None = None
    message: str = Field(
        default="",
        max_length=100_000,
        description="Proposed next input for estimation only; never submitted for execution.",
    )


class CancelSessionRequest(BaseModel):
    user_id: str = Field(..., min_length=1)


class CancelSessionResponse(BaseModel):
    status: Literal["requested"] = "requested"
    detail: str = "Cancellation requested; wait for final evidence while owned work settles."


ChatResponse = TurnResponse


__all__ = [
    "CancelSessionRequest",
    "CancelSessionResponse",
    "ChatRequest",
    "ChatResponse",
    "ConfigurationReviewRequest",
    "CreateCheckpointRequest",
    "CreateSessionRequest",
    "CreateSessionResponse",
    "RevertFileRequest",
    "SubmitRunRequest",
]
