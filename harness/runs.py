"""Validated durable run identity; execution remains in the shared runtime."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

from providers.base import ProviderFailure

from .budget_evidence import BudgetEvidence
from .execution_profile import ExecutionProfile
from .recovery_identity import RecoverySource
from .state import ContinuationSource, TurnResponse

RequestId = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
]


class RunInput(BaseModel):
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    provider: str | None = None
    request_id: RequestId
    continuation_source: ContinuationSource | None = None
    recovery_source: RecoverySource | None = Field(default=None, frozen=True)

    @model_serializer(mode="wrap")
    def serialize_input(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if self.recovery_source is None:
            data.pop("recovery_source", None)
        return data


class RunRecord(BaseModel):
    """Durable submission and final review result; never an executable checkpoint.

    Finished records can have incomplete, blocked, or failed checks. Inspect the
    response completion and verification fields to assess the task.
    """

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    request: RunInput
    provider: str
    configured_model: str | None = None
    before_checkpoint_id: str | None = None
    after_checkpoint_id: str | None = None
    execution_profile: ExecutionProfile | None = Field(
        default=None,
        description=(
            "Allowlisted configured execution evidence; null when not captured, "
            "including legacy/unadmitted work. "
            "Does not attest backend behavior or grant resume permission."
        ),
    )
    status: Literal["running", "finished", "cancelled", "failed", "interrupted"] = "running"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    response: TurnResponse | None = None
    budget_evidence: BudgetEvidence | None = Field(
        default=None,
        description=(
            "Bounded request/usage/token-stop metadata; null for legacy or unadmitted runs."
        ),
    )
    error: str | None = None
    provider_failure: ProviderFailure | None = Field(
        default=None,
        description=(
            "Allowlisted provider failure observation, without raw error text or payloads. "
            "Null for legacy records, non-provider failures or when not observed. "
            "Does not establish remote cancellation, token usage or retry safety."
        ),
    )


class RequestConflictError(ValueError):
    """The same user's request ID was already bound to different input."""
