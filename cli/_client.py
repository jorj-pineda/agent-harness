"""Thin synchronous HTTP wrapper around the agent-harness FastAPI server.

cli/ is a pure HTTP client — it never imports harness/, providers/, or tools/
directly. All agent logic stays in the server; the client calls /sessions and
/chat over HTTP.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class TurnSummary:
    """Flattened view of the TurnResponse envelope for CLI display."""

    answer: str
    confidence: float | None
    escalated: bool
    provider: str
    latency_ms: float
    files_touched: list[str] = field(default_factory=list)
    completion_status: str = "completed"
    completion_reason: str | None = None
    verification_status: str = "not_run"
    check_attempts: list[dict[str, object]] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)
    token_usage: dict[str, int | None] = field(default_factory=dict)
    workspace_changes: dict[str, object] = field(default_factory=dict)
    run_id: str | None = None
    workspace_review_ms: float | None = None


class AgentClient:
    """Synchronous client for /sessions and /chat endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self._base = base_url.rstrip("/")

    def create_session(
        self, user_id: str, workspace_root: str | None = None, *, checkpoint_turns: bool = False
    ) -> str:
        payload: dict[str, str | bool] = {"user_id": user_id}
        if workspace_root is not None:
            payload["workspace_root"] = workspace_root
        if checkpoint_turns:
            payload["checkpoint_turns"] = True
        resp = httpx.post(f"{self._base}/sessions", json=payload, timeout=30)
        resp.raise_for_status()
        return str(resp.json()["session_id"])

    def continue_session(
        self,
        user_id: str,
        session_id: str,
        expected_turn_id: str,
        message: str,
        request_id: str,
        *,
        provider: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "user_id": user_id,
            "expected_turn_id": expected_turn_id,
            "message": message,
            "request_id": request_id,
            "provider": provider,
            "acknowledge_uncaptured_state": True,
        }
        resp = httpx.post(f"{self._base}/sessions/{session_id}/continue", json=payload, timeout=30)
        resp.raise_for_status()
        record = dict(resp.json())
        request = record.get("request", {})
        expected_source = {
            "session_id": session_id,
            "turn_id": expected_turn_id,
            "acknowledge_uncaptured_state": True,
        }
        if (
            request.get("user_id") != user_id
            or request.get("request_id") != request_id
            or request.get("message") != message
            or request.get("provider") != provider
            or request.get("continuation_source") != expected_source
            or not request.get("session_id")
            or request.get("session_id") == session_id
            or not record.get("run_id")
        ):
            raise ValueError("Mismatched continuation response; inspect the keyed run.")
        return record

    def cancel(self, user_id: str, session_id: str) -> str:
        resp = httpx.post(
            f"{self._base}/sessions/{session_id}/cancel",
            json={"user_id": user_id},
            timeout=30,
        )
        resp.raise_for_status()
        return str(resp.json()["detail"])

    def chat(
        self,
        user_id: str,
        session_id: str,
        message: str,
        provider: str | None = None,
        *,
        request_id: str | None = None,
    ) -> TurnSummary:
        payload: dict[str, str] = {
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "request_id": request_id or uuid.uuid4().hex,
        }
        if provider is not None:
            payload["provider"] = provider
        resp = httpx.post(f"{self._base}/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return summarize(resp.json())

    def memory(
        self, user_id: str, *, limit: int = 20, before_id: int | None = None
    ) -> dict[str, Any]:
        params: dict[str, str | int] = {"user_id": user_id, "limit": limit}
        if before_id is not None:
            params["before_id"] = before_id
        resp = httpx.get(f"{self._base}/memory", params=params, timeout=30)
        resp.raise_for_status()
        return dict(resp.json())

    def delete_memory(
        self, user_id: str, fact_id: int, expected_fact_sha256: str
    ) -> dict[str, Any]:
        resp = httpx.request(
            "DELETE",
            f"{self._base}/memory/{fact_id}",
            json={"user_id": user_id, "expected_fact_sha256": expected_fact_sha256},
            timeout=30,
        )
        resp.raise_for_status()
        record = dict(resp.json())
        if record.get("status") != "deleted" or record.get("fact_id") != fact_id:
            raise ValueError("Mismatched memory deletion response; inspect memory.")
        return record

    def runs(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        params: dict[str, str | int] = {"user_id": user_id, "limit": limit, "offset": offset}
        if session_id is not None:
            params["session_id"] = session_id
        if request_id is not None:
            params["request_id"] = request_id
        resp = httpx.get(f"{self._base}/runs", params=params, timeout=30)
        resp.raise_for_status()
        return list(resp.json())

    def run(self, user_id: str, run_id: str) -> dict[str, Any]:
        resp = httpx.get(f"{self._base}/runs/{run_id}", params={"user_id": user_id}, timeout=30)
        resp.raise_for_status()
        return dict(resp.json())

    def tool_progress(self, user_id: str, run_id: str) -> dict[str, Any]:
        resp = httpx.get(
            f"{self._base}/runs/{run_id}/tool-progress",
            params={"user_id": user_id},
            timeout=30,
        )
        resp.raise_for_status()
        return dict(resp.json())

    def create_checkpoint(self, user_id: str, session_id: str) -> dict[str, Any]:
        resp = httpx.post(
            f"{self._base}/sessions/{session_id}/checkpoints",
            json={"user_id": user_id},
            timeout=120,
        )
        resp.raise_for_status()
        return dict(resp.json())

    def checkpoints(
        self, user_id: str, session_id: str, *, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        resp = httpx.get(
            f"{self._base}/sessions/{session_id}/checkpoints",
            params={"user_id": user_id, "limit": limit, "offset": offset},
            timeout=30,
        )
        resp.raise_for_status()
        return list(resp.json())

    def checkpoint(self, user_id: str, session_id: str, checkpoint_id: str) -> dict[str, Any]:
        resp = httpx.get(
            f"{self._base}/sessions/{session_id}/checkpoints/{checkpoint_id}",
            params={"user_id": user_id},
            timeout=30,
        )
        resp.raise_for_status()
        return dict(resp.json())

    def revert_file(self, user_id: str, run_id: str, path: str, request_id: str) -> dict[str, Any]:
        resp = httpx.post(
            f"{self._base}/runs/{run_id}/revert-file",
            json={"user_id": user_id, "path": path, "request_id": request_id},
            timeout=120,
        )
        data = resp.json()
        if isinstance(data, dict) and data.get("revert_id") is not None:
            expected = {
                "user_id": user_id,
                "run_id": run_id,
                "path": path,
                "request_id": request_id,
            }
            codes = {
                "applied": 200,
                "pending": 202,
                "rejected": 409,
                "uncertain": 503,
                "interrupted": 409,
            }
            if data.get("request") != expected or codes.get(data.get("status")) != resp.status_code:
                raise ValueError("Mismatched revert response; inspect the keyed journal.")
            return data
        resp.raise_for_status()
        raise ValueError("No revert record returned")

    def revert_status(self, user_id: str, request_id: str) -> dict[str, Any] | None:
        resp = httpx.get(
            f"{self._base}/file-reverts",
            params={"user_id": user_id, "request_id": request_id},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return dict(data) if data is not None else None

    def configuration_review(
        self, user_id: str, session_id: str, *, provider: str | None = None, message: str = ""
    ) -> dict[str, Any]:
        resp = httpx.post(
            f"{self._base}/sessions/{session_id}/configuration-review",
            json={"user_id": user_id, "provider": provider, "message": message},
            timeout=30,
        )
        resp.raise_for_status()
        return dict(resp.json())

    def transcript_review(
        self, user_id: str, session_id: str, *, max_issues: int = 100
    ) -> dict[str, Any]:
        resp = httpx.get(
            f"{self._base}/sessions/{session_id}/transcript-review",
            params={"user_id": user_id, "max_issues": max_issues},
            timeout=30,
        )
        resp.raise_for_status()
        return dict(resp.json())

    def workspace_review(
        self, user_id: str, session_id: str, *, max_paths: int = 100
    ) -> dict[str, Any]:
        resp = httpx.get(
            f"{self._base}/sessions/{session_id}/workspace-review",
            params={"user_id": user_id, "max_paths": max_paths},
            timeout=120,
        )
        resp.raise_for_status()
        return dict(resp.json())


def summarize(data: dict[str, Any]) -> TurnSummary:
    """Use the same review presentation for live and saved final envelopes."""
    return TurnSummary(
        run_id=data.get("run_id"),
        answer=str(data.get("answer", "")),
        confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
        escalated=bool(data.get("escalated", False)),
        provider=str(data.get("provider", "")),
        latency_ms=float(data.get("latency_ms", 0.0)),
        workspace_review_ms=float(data["workspace_review_ms"])
        if data.get("workspace_review_ms") is not None
        else None,
        files_touched=list(data.get("files_touched", [])),
        completion_status=str(data.get("completion_status", "completed")),
        completion_reason=data.get("completion_reason"),
        verification_status=str(data.get("verification_status", "not_run")),
        check_attempts=list(data.get("check_attempts", [])),
        tool_errors=list(data.get("tool_errors", [])),
        token_usage=dict(data.get("token_usage", {})),
        workspace_changes=dict(data.get("workspace_changes", {})),
    )
