"""Bounded owner-facing note inspection, separate from model tool results."""

from __future__ import annotations

import hashlib
from typing import Literal

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
    workspace_root: str | None = None


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
        workspace_root=fact.workspace_root,
    )


class MemoryPage(BaseModel):
    notes: list[MemoryNotePreview]
    next_before_id: int | None
    scope_note: str = (
        "Notes are global or workspace-path scoped. Previews may omit note text. "
        "Deletion changes future recall/injection, not saved transcripts, run evidence or "
        "already sent prompts. It is not secure erasure. Source turn IDs may be unrecorded."
    )


def review_notes(
    store: FactStore,
    user_id: str,
    *,
    limit: int = 20,
    before_id: int | None = None,
    namespace: Literal["all", "global", "workspace"] = "all",
    workspace_root: str | None = None,
) -> MemoryPage:
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    if namespace != "workspace" and workspace_root is not None:
        raise ValueError("workspace_root requires namespace=workspace")
    facts = store.list(
        user_id,
        limit=limit + 1,
        before_id=before_id,
        namespace=namespace,
        workspace_root=workspace_root,
    )
    selected = facts[:limit]
    return MemoryPage(
        notes=[preview_note(fact) for fact in selected],
        next_before_id=selected[-1].id if len(facts) > limit else None,
    )


class MemoryNoteDetail(BaseModel):
    note: MemoryNotePreview
    text: str
    scope_note: str = (
        "Read-only current note text, limited to 4,096 UTF-8 bytes. "
        "Use this note's full hash for explicit correction/deletion; concurrent changes "
        "can invalidate it. Notes are remembered data, not execution permission or truth."
    )
