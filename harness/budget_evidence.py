"""Bounded normalized request/budget observations, without model payloads."""

from typing import Any, Literal

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

from providers.base import ProviderCallObservation, TokenUsage

from .context import RequestBudgetBlock
from .trace import RequestIdentity

MAX_SAVED_BUDGET_EVENTS = 100
BudgetObservationKind = Literal[
    "request", "response", "request_failure", "pre_request_stop", "post_response_stop"
]


class BudgetObservation(BaseModel):
    request_block: RequestBudgetBlock | None = None

    kind: BudgetObservationKind
    iteration: int = Field(ge=0)
    message_count: int = Field(ge=0)
    estimated_prompt_tokens: int | None = Field(default=None, ge=0)
    heuristic_prompt_tokens: int | None = Field(default=None, ge=0)
    anchor_adjustment_tokens: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=0)
    identity: RequestIdentity | None = None
    usage: TokenUsage | None = None
    call_observation: ProviderCallObservation | None = None
    prompt_tokens_used: int | None = Field(default=None, ge=0)
    completion_tokens_used: int | None = Field(default=None, ge=0)

    @model_serializer(mode="wrap")
    def serialize_observation(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if self.request_block is None:
            data.pop("request_block", None)
        return data


class BudgetEvidence(BaseModel):
    events: list[BudgetObservation] = Field(
        default_factory=list, max_length=MAX_SAVED_BUDGET_EVENTS
    )
    observed_events: int = Field(default=0, ge=0)
    events_truncated: bool = False
    read_only: Literal[True] = True
    scope_note: str = (
        "Bounded prefix of normalized request attempts, reported response usage and token-gate "
        "stops. Request precedes delivery; response usage is a provider report, not backend "
        "attestation. Failure usage is unknown; a request without a response may have consumed "
        "tokens. Missing/truncated evidence is not zero usage or proven budget compliance. "
        "No messages, paths, call IDs, tool arguments/results or exception text are stored here. "
        "Fingerprints cover normalized inputs only; see the separate configured profile for "
        "limits/provider identity. Optional call observations describe adapter HTTP timing and "
        "allowlisted response fields, not backend causes or token compliance. Missing observations "
        "stay unknown, including cancellation, parsing failures and unsupported providers. "
        "Wall/tool/iteration stops are not token-gate events. "
        "Not completion, acceptance, replay or permission to resume."
    )
