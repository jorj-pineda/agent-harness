"""Stable normalized recovery identities, independent of storage and execution policy."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from providers.base import ChatMessage


def recovery_messages_sha256(messages: Sequence[ChatMessage]) -> str:
    return hashlib.sha256("\n".join(m.model_dump_json() for m in messages).encode()).hexdigest()
