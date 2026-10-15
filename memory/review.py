"""Bounded owner-facing note inspection, separate from model tool results."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel

from .store import Fact, FactStore

MAX_PREVIEW_BYTES = 1024


class MemoryNotePreview(BaseModel):
    id: int
    preview: str
    preview_truncated: bool
    fact_bytes: int
    fact_sha256: str
    created_at: str
    updated_at: str | None = None
    source_turn_id: str | None


def preview_note(fact: Fact) -> MemoryNotePreview:
    raw = fact.fact.encode("utf-8")
    return MemoryNotePreview(
        id=fact.id,
        preview=raw[:MAX_PREVIEW_BYTES].decode("utf-8", errors="ignore"),
        preview_truncated=len(raw) > MAX_PREVIEW_BYTES,
        fact_bytes=len(raw),
        fact_sha256=hashlib.sha256(raw).hexdigest(),
        created_at=fact.created_at,
        updated_at=fact.updated_at,
        source_turn_id=fact.source_turn_id,
    )


class MemoryPage(BaseModel):
    notes: list[MemoryNotePreview]
    next_before_id: int | None
    scope_note: str = (
        "User-scoped memory, not repository-scoped. Previews may omit note text. "
        "Deletion changes future recall/injection, not saved transcripts, run evidence or "
        "already sent prompts. It is not secure erasure. Source turn IDs may be unrecorded."
    )


def review_notes(
    store: FactStore, user_id: str, *, limit: int = 20, before_id: int | None = None
) -> MemoryPage:
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    facts = store.list(user_id, limit=limit + 1, before_id=before_id)
    selected = facts[:limit]
    return MemoryPage(
        notes=[preview_note(fact) for fact in selected],
        next_before_id=selected[-1].id if len(facts) > limit else None,
    )
