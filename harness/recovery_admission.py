"""Fresh recovery preconditions and a one-run normalized first-request gate."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, field_validator

from memory import FactStore
from providers.base import (
    ChatMessage,
    ChatProvider,
    ConversationProvider,
    ProviderError,
    ProviderResponse,
    ToolSpec,
)
from tools import ToolRegistry

from .checkpoints import CheckpointRecord
from .config import Settings
from .execution_profile import tool_specs_digest
from .recovery_draft import RecoveryDraft, RecoveryDraftSelection, build_recovery_draft
from .recovery_identity import recovery_messages_sha256
from .runs import RunRecord
from .session_store import SessionArchive, SessionStore


class RecoveryApproval(RecoveryDraftSelection):
    expected_messages_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    expected_archive_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    expected_tool_specs_sha256: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    expected_request_output_allowance: int = Field(strict=True, ge=1, le=1_000_000)
    acknowledge_uncaptured_state: Literal[True]
    acknowledge_omitted_context: Literal[True]
    acknowledge_new_turn_budget: Literal[True]

    @field_validator(
        "acknowledge_uncaptured_state",
        "acknowledge_omitted_context",
        "acknowledge_new_turn_budget",
        mode="before",
    )
    @classmethod
    def explicit_true(cls, value: object) -> Literal[True]:
        if value is not True:
            raise ValueError("Explicit true acknowledgment required")
        return True


AdmissionCode = Literal[
    "source_unavailable",
    "run_active",
    "record_changed",
    "archive_changed",
    "checkpoint_unavailable",
    "workspace_changed",
    "newer_unfinished_work",
    "configuration_unusable",
    "context_unusable",
    "backend_context_unknown",
    "draft_changed",
]


class RecoveryAdmissionBlocked(ValueError):
    def __init__(self, code: AdmissionCode):
        self.code = code
        super().__init__(code)


def check_recovery_admission(
    store: SessionStore,
    record: RunRecord,
    archive: SessionArchive,
    current: CheckpointRecord,
    approval: RecoveryApproval,
    *,
    settings: Settings,
    provider: str,
    registry: ToolRegistry,
    fact_store: FactStore,
    user_id: str,
    excluding_run_id: str = "",
) -> RecoveryDraft:
    """Caller must own admission and supply a fresh captured workspace; never returns a lease."""
    if record.request.user_id != user_id or archive.session.user_id != user_id:
        raise RecoveryAdmissionBlocked("source_unavailable")
    if record.status == "running":
        raise RecoveryAdmissionBlocked("run_active")
    if (
        hashlib.sha256(record.model_dump_json().encode()).hexdigest()
        != approval.expected_record_sha256
    ):
        raise RecoveryAdmissionBlocked("record_changed")
    if (
        hashlib.sha256(archive.model_dump_json().encode()).hexdigest()
        != approval.expected_archive_sha256
    ):
        raise RecoveryAdmissionBlocked("archive_changed")
    expected = (
        store.get_checkpoint(record.after_checkpoint_id) if record.after_checkpoint_id else None
    )
    if (
        expected is None
        or expected.checkpoint_id != record.after_checkpoint_id
        or expected.kind != "turn_after"
        or expected.run_id != record.run_id
        or expected.session_id != record.request.session_id
        or expected.paired_before_id != record.before_checkpoint_id
        or record.before_checkpoint_id is None
        or archive.session.session_id != record.request.session_id
        or archive.session.workspace_root != expected.root
    ):
        raise RecoveryAdmissionBlocked("checkpoint_unavailable")
    if (expected.root, expected.root_identity, expected.files) != (
        current.root,
        current.root_identity,
        current.files,
    ):
        raise RecoveryAdmissionBlocked("workspace_changed")
    if store.has_newer_unfinished_work(
        user_id,
        record.request.session_id,
        record.created_at,
        excluding_run_id=excluding_run_id,
        workspace_root=expected.root,
    ):
        raise RecoveryAdmissionBlocked("newer_unfinished_work")
    draft = build_recovery_draft(
        store,
        record,
        archive,
        approval,
        settings=settings,
        provider=provider,
        registry=registry,
        fact_store=fact_store,
        user_id=user_id,
    )
    if draft.configuration.status != "matching":
        raise RecoveryAdmissionBlocked("configuration_unusable")
    if draft.context.status != "estimated_fit":
        raise RecoveryAdmissionBlocked("context_unusable")
    if provider == "ollama" and settings.ollama_num_ctx is None:
        raise RecoveryAdmissionBlocked("backend_context_unknown")
    if (
        draft.messages_sha256 != approval.expected_messages_sha256
        or draft.tool_specs_sha256 != approval.expected_tool_specs_sha256
        or draft.context.request_output_allowance != approval.expected_request_output_allowance
    ):
        raise RecoveryAdmissionBlocked("draft_changed")
    return draft


@dataclass
class _GuardState:
    dispatched: bool = False
    refused: bool = False


class RecoveryInputGuard:
    """One wrapper per run; underlying client ownership/cleanup stays with its caller."""

    def __init__(
        self,
        provider: ChatProvider,
        approval: RecoveryApproval,
        *,
        state: _GuardState | None = None,
    ):
        self.name = provider.name
        self._provider = provider
        self._approval = approval
        self._state = state or _GuardState()

    def for_conversation(self, conversation_id: str) -> ChatProvider:
        bound = (
            self._provider.for_conversation(conversation_id)
            if isinstance(self._provider, ConversationProvider)
            else self._provider
        )
        return RecoveryInputGuard(bound, self._approval, state=self._state)

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        if self._state.refused:
            raise ProviderError("Recovery input guard refused further dispatch.")
        if not self._state.dispatched:
            if (
                recovery_messages_sha256(messages) != self._approval.expected_messages_sha256
                or tool_specs_digest(tools or []) != self._approval.expected_tool_specs_sha256
                or type(max_tokens) is not int
                or max_tokens != self._approval.expected_request_output_allowance
            ):
                self._state.refused = True
                raise ProviderError(
                    "Recovery first input differs from reviewed input; dispatch refused."
                )
            self._state.dispatched = True
        try:
            return await self._provider.chat(
                messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except BaseException:
            self._state.refused = True
            raise
