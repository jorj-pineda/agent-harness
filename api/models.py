"""Request/response Pydantic models for the API layer.

Responses ship the rule-#5 envelope from `harness/state.py:TurnResponse`,
re-exported here as `ChatResponse` so the public API surface lives in one
module — callers don't need to know which inner layer owns the schema.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from harness.state import TurnResponse


class CreateSessionRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Stable per-user identifier.")


class CreateSessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    provider: str | None = Field(
        default=None,
        description="Optional provider override; falls back to the configured default.",
    )


ChatResponse = TurnResponse


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "CreateSessionRequest",
    "CreateSessionResponse",
]
