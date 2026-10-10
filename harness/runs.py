"""Validated durable run identity; execution remains in the shared runtime."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .state import TurnResponse

RequestId = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
]


class RunInput(BaseModel):
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    provider: str | None = None
    request_id: RequestId


class RunRecord(BaseModel):
    """Durable submission and final review result; never an executable checkpoint.

    Finished records can have incomplete, blocked, or failed checks. Inspect the
    response completion and verification fields to assess the task.
    """

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    request: RunInput
    provider: str
    configured_model: str | None = None
    status: Literal["running", "finished", "cancelled", "failed", "interrupted"] = "running"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    response: TurnResponse | None = None
    error: str | None = None


class RequestConflictError(ValueError):
    """The same user's request ID was already bound to different input."""
