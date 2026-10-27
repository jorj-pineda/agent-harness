"""Stable normalized recovery identities, independent of storage and execution policy."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from providers.base import ChatMessage


def recovery_messages_sha256(messages: Sequence[ChatMessage]) -> str:
    return hashlib.sha256("\n".join(m.model_dump_json() for m in messages).encode()).hexdigest()


# Deliberately storage-only: execution parses approval_json as RecoveryApproval again.
# Keeping that policy out of this module avoids a state/runtime/storage import cycle.
class RecoverySource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    draft_version: Literal["recovery-draft-v1"] = "recovery-draft-v1"
    source_run_id: str = Field(
        strict=True, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"
    )
    source_session_id: str = Field(
        strict=True, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"
    )
    source_turn_id: str = Field(
        strict=True, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"
    )
    source_record_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    source_archive_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    messages_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    tool_specs_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    request_output_allowance: int = Field(strict=True, ge=1, le=1_000_000)
    approval_json: str = Field(strict=True, min_length=1, max_length=65536)
    approval_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    budget_scope: Literal["separate-configured-turn"] = "separate-configured-turn"

    @model_validator(mode="after")
    def consistent_approval(self) -> RecoverySource:
        raw = self.approval_json.encode("utf-8")
        if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != self.approval_sha256:
            raise ValueError("Recovery approval size/hash mismatch")
        value = json.loads(self.approval_json)
        if (
            not isinstance(value, dict)
            or json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            != self.approval_json
        ):
            raise ValueError("Recovery approval must be canonical compact JSON")
        for field, expected in (
            ("expected_record_sha256", self.source_record_sha256),
            ("expected_archive_sha256", self.source_archive_sha256),
            ("expected_messages_sha256", self.messages_sha256),
            ("expected_tool_specs_sha256", self.tool_specs_sha256),
        ):
            if value.get(field) != expected:
                raise ValueError("Recovery approval pin mismatch")
        if (
            type(value.get("expected_request_output_allowance")) is not int
            or value["expected_request_output_allowance"] != self.request_output_allowance
        ):
            raise ValueError("Recovery approval allowance mismatch")
        if any(
            value.get(name) is not True
            for name in (
                "acknowledge_uncaptured_state",
                "acknowledge_omitted_context",
                "acknowledge_new_turn_budget",
            )
        ):
            raise ValueError("Recovery scope acknowledgment missing")
        return self
