"""Request/response Pydantic models for the API layer.

Responses ship the rule-#5 envelope from `harness/state.py:TurnResponse`,
re-exported here as `ChatResponse` so the public API surface lives in one
module — callers don't need to know which inner layer owns the schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from harness.runs import RequestId
from harness.state import TurnResponse


class CreateSessionRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Stable per-user identifier.")
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
    "CreateSessionRequest",
    "CreateSessionResponse",
    "SubmitRunRequest",
]
