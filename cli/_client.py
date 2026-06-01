"""Thin synchronous HTTP wrapper around the agent-harness FastAPI server.

cli/ is a pure HTTP client — it never imports harness/, providers/, or tools/
directly. All agent logic stays in the server; the client calls /sessions and
/chat over HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx


@dataclass
class TurnSummary:
    """Flattened view of the TurnResponse envelope for CLI display."""

    answer: str
    confidence: float
    escalated: bool
    provider: str
    latency_ms: float
    files_touched: list[str] = field(default_factory=list)


class AgentClient:
    """Synchronous client for /sessions and /chat endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self._base = base_url.rstrip("/")

    def create_session(self, user_id: str, workspace_root: str | None = None) -> str:
        payload: dict[str, str] = {"user_id": user_id}
        if workspace_root is not None:
            payload["workspace_root"] = workspace_root
        resp = httpx.post(f"{self._base}/sessions", json=payload, timeout=30)
        resp.raise_for_status()
        return str(resp.json()["session_id"])

    def chat(
        self,
        user_id: str,
        session_id: str,
        message: str,
        provider: str | None = None,
    ) -> TurnSummary:
        payload: dict[str, str] = {
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
        }
        if provider is not None:
            payload["provider"] = provider
        resp = httpx.post(f"{self._base}/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return TurnSummary(
            answer=str(data.get("answer", "")),
            confidence=float(data.get("confidence", 0.0)),
            escalated=bool(data.get("escalated", False)),
            provider=str(data.get("provider", "")),
            latency_ms=float(data.get("latency_ms", 0.0)),
            files_touched=list(data.get("files_touched", [])),
        )
