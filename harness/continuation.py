"""Strict configured/context and scoped workspace checks for explicit continuation forks."""

from __future__ import annotations

from memory import FactStore
from tools import ToolRegistry

from .checkpoints import CheckpointRecord
from .config import Settings
from .configuration_review import review_configuration
from .session_store import SessionArchive


class ContinuationBlocked(ValueError):
    pass


def check_continuation(
    archive: SessionArchive,
    expected: CheckpointRecord,
    current: CheckpointRecord,
    *,
    settings: Settings,
    provider: str,
    registry: ToolRegistry,
    fact_store: FactStore,
    user_id: str,
    message: str,
) -> None:
    """Checks consume current captured evidence under admission; never a reusable lease."""
    if (expected.root, expected.root_identity, expected.files) != (
        current.root,
        current.root_identity,
        current.files,
    ):
        raise ContinuationBlocked(
            "Scoped workspace content/modes or root changed; start a fresh session."
        )
    review = review_configuration(
        archive,
        settings=settings,
        provider=provider,
        registry=registry,
        fact_store=fact_store,
        user_id=user_id,
        message=message,
    )
    if review.configuration is None or review.configuration.status != "matching":
        raise ContinuationBlocked(
            "Saved configuration is changed, unknown or inconsistent; start a fresh session."
        )
    if review.transcript is None or review.transcript.status != "valid":
        raise ContinuationBlocked(
            "Saved transcript is structurally invalid; no continuation or repair."
        )
    if review.context is None or review.context.status != "estimated_fit":
        raise ContinuationBlocked("Proposed continuation lacks bounded estimated context fit.")
    if provider == "ollama" and (
        settings.ollama_num_ctx is None or settings.ollama_num_ctx < settings.max_context_tokens
    ):
        raise ContinuationBlocked(
            "Configure Ollama context at least as large as the harness context limit."
        )
