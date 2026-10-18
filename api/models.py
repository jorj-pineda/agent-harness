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
from memory.review import MemoryNotePreview


class DeleteMemoryRequest(BaseModel):
    expected_workspace_root: str | None = Field(
        default=None,
        max_length=4096,
        description="Optional reviewed scope: omit for legacy content-only guard, null for global, "
        "or exact canonical workspace path. Scope is checked in the final mutation predicate.",
    )
    user_id: str = Field(min_length=1)
    expected_fact_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")


class DeleteMemoryResponse(BaseModel):
    status: Literal["deleted"] = "deleted"
    fact_id: int
    scope_note: str = (
        "Future memory use only; saved history and already sent prompts are unchanged."
    )


class CorrectMemoryRequest(DeleteMemoryRequest):
    replacement: str = Field(min_length=1, max_length=8192)


class CorrectMemoryResponse(BaseModel):
    status: Literal["corrected"] = "corrected"
    note: MemoryNotePreview
    scope_note: str = (
        "Future memory use only; saved history and already sent prompts are unchanged."
    )


class ReclassifyMemoryRequest(DeleteMemoryRequest):
    expected_workspace_root: str | None = Field(..., max_length=4096)
    workspace_root: str | None = Field(..., max_length=4096)


class ReclassifyMemoryResponse(BaseModel):
    status: Literal["reclassified"] = "reclassified"
    note: MemoryNotePreview
    scope_note: str = (
        "Future memory scope only; text, creation/correction metadata, saved history and "
        "already sent prompts are unchanged. No scope-change audit trail is recorded."
    )


class CreateSessionRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Stable per-user identifier.")
    inspection_only: bool = Field(
        default=False,
        strict=True,
        description="Restrict model tools to file inspection and memory recall. No commands, "
        "file edits or memory writes; not host process isolation. Fixed for this session.",
    )
    checkpoint_turns: bool = Field(
        default=False,
        description="Opt in to bounded plaintext workspace captures before/after admitted turns.",
    )
    workspace_root: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional absolute trusted workspace path, resolved and stored on the session. "
            "File-tool path validation is not process isolation for repository commands."
        ),
    )


class CreateSessionResponse(BaseModel):
    session_id: str
    inspection_only: bool = False


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


class ContinueSessionRequest(BaseModel):
    user_id: str = Field(min_length=1)
    request_id: RequestId
    expected_turn_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=100_000)
    provider: str | None = None
    acknowledge_uncaptured_state: Literal[True] = Field(
        description="Required explicit acknowledgment of excluded files, metadata, external "
        "writers and unrecorded backend/tool state. Guards are scoped evidence, not isolation."
    )


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
    "ContinueSessionRequest",
    "CorrectMemoryRequest",
    "CorrectMemoryResponse",
    "CreateCheckpointRequest",
    "CreateSessionRequest",
    "CreateSessionResponse",
    "DeleteMemoryRequest",
    "DeleteMemoryResponse",
    "ReclassifyMemoryRequest",
    "ReclassifyMemoryResponse",
    "RevertFileRequest",
    "SubmitRunRequest",
]
