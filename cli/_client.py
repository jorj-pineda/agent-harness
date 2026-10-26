"""Thin synchronous HTTP wrapper around the agent-harness FastAPI server.

cli/ is a pure HTTP client — it never imports harness/, providers/, or tools/
directly. All agent logic stays in the server; the client calls /sessions and
/chat over HTTP.
"""

from __future__ import annotations

import hashlib
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

    def __init__(
        self, base_url: str = "http://localhost:8000", *, response_timeout: int = 120
    ) -> None:
        if type(response_timeout) is not int or not 1 <= response_timeout <= 3600:
            raise ValueError("Response timeout must be an integer from 1 to 3600 seconds")
        self._base = base_url.rstrip("/")
        # Only /chat's read wait changes. Other transport phases and owner actions
        # keep their existing timeouts; this is not a server execution deadline.
        self._chat_timeout = httpx.Timeout(120, read=response_timeout)

    def create_session(
        self,
        user_id: str,
        workspace_root: str | None = None,
        *,
        checkpoint_turns: bool = False,
        inspection_only: bool = False,
    ) -> str:
        payload: dict[str, str | bool] = {"user_id": user_id}
        if workspace_root is not None:
            payload["workspace_root"] = workspace_root
        if checkpoint_turns:
            payload["checkpoint_turns"] = True
        if inspection_only:
            payload["inspection_only"] = True
        resp = httpx.post(f"{self._base}/sessions", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if inspection_only and (
            not isinstance(data, dict)
            or data.get("inspection_only") is not True
            or not isinstance(data.get("session_id"), str)
            or not data["session_id"].strip()
        ):
            raise ValueError(
                "Server did not acknowledge an inspection-only session. No chat was sent; "
                "use a server supporting this policy. No automatic retry."
            )
        return str(data["session_id"])

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
        resp = httpx.post(f"{self._base}/chat", json=payload, timeout=self._chat_timeout)
        resp.raise_for_status()
        return summarize(resp.json())

    def memory(
        self,
        user_id: str,
        *,
        limit: int = 20,
        before_id: int | None = None,
        namespace: str = "all",
        workspace_root: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, str | int] = {"user_id": user_id, "limit": limit}
        if before_id is not None:
            params["before_id"] = before_id
        if namespace != "all":
            params["namespace"] = namespace
        if workspace_root is not None:
            params["workspace_root"] = workspace_root
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

    def correct_memory(
        self, user_id: str, fact_id: int, expected_fact_sha256: str, replacement: str
    ) -> dict[str, Any]:
        resp = httpx.request(
            "PUT",
            f"{self._base}/memory/{fact_id}",
            json={
                "user_id": user_id,
                "expected_fact_sha256": expected_fact_sha256,
                "replacement": replacement,
            },
            timeout=30,
        )
        resp.raise_for_status()
        record = dict(resp.json())
        if record.get("status") != "corrected" or (record.get("note") or {}).get("id") != fact_id:
            raise ValueError("Mismatched memory correction response; inspect memory.")
        return record

    def reclassify_memory(
        self,
        user_id: str,
        fact_id: int,
        expected_fact_sha256: str,
        expected_workspace_root: str | None,
        workspace_root: str | None,
    ) -> dict[str, Any]:
        resp = httpx.request(
            "PUT",
            f"{self._base}/memory/{fact_id}/scope",
            json={
                "user_id": user_id,
                "expected_fact_sha256": expected_fact_sha256,
                "expected_workspace_root": expected_workspace_root,
                "workspace_root": workspace_root,
            },
            timeout=30,
        )
        resp.raise_for_status()
        record = dict(resp.json())
        note = record.get("note") or {}
        if (
            record.get("status") != "reclassified"
            or note.get("id") != fact_id
            or "workspace_root" not in note
            or note["workspace_root"] != workspace_root
            or note.get("fact_sha256") != expected_fact_sha256
        ):
            raise ValueError("Mismatched memory scope response; inspect memory.")
        return record

    def memory_note(self, user_id: str, fact_id: int) -> dict[str, Any]:
        resp = httpx.get(f"{self._base}/memory/{fact_id}", params={"user_id": user_id}, timeout=30)
        resp.raise_for_status()
        record = dict(resp.json())
        note = record.get("note") or {}
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError("Missing memory note text")
        encoded = text.encode("utf-8")
        if (
            note.get("id") != fact_id
            or len(encoded) > 4096
            or note.get("fact_bytes") != len(encoded)
            or note.get("fact_sha256") != hashlib.sha256(encoded).hexdigest()
        ):
            raise ValueError("Mismatched memory detail response; inspect memory.")
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

    def recovery_review(self, user_id: str, run_id: str) -> dict[str, Any]:
        resp = httpx.get(
            f"{self._base}/runs/{run_id}/recovery-review",
            params={"user_id": user_id},
            timeout=30,
        )
        resp.raise_for_status()
        if len(resp.content) > 32768:
            raise ValueError("Recovery review exceeds the supported wire size")
        record = resp.json()
        if (
            not isinstance(record, dict)
            or type(record.get("schema_version")) is not int
            or record["schema_version"] != 1
            or record.get("run_id") != run_id
            or record.get("run_status") not in {"finished", "cancelled", "failed", "interrupted"}
            or type(record.get("final_response_available")) is not bool
            or record.get("max_serialized_bytes") != 32768
            or not isinstance(record.get("source_record_sha256"), str)
            or not isinstance(record.get("scope_note"), str)
        ):
            raise ValueError("Unsupported recovery review response")
        for name in ("unresolved_checks", "recent_actions", "diffs"):
            items = record.get(name)
            if (
                not isinstance(items, list)
                or len(items) > 8
                or any(not isinstance(item, dict) for item in items)
            ):
                raise ValueError("Invalid recovery review items")
        for name in (
            "unresolved_check_count",
            "action_count",
            "diff_count",
            "omitted_unresolved_checks",
            "omitted_actions",
            "omitted_diffs",
        ):
            value = record.get(name)
            if name not in record or (value is not None and (type(value) is not int or value < 0)):
                raise ValueError("Invalid recovery review counts")
            if record["final_response_available"] and value is None:
                raise ValueError("Final-response recovery counts must be recorded")
            if not record["final_response_available"] and value is not None:
                raise ValueError("Missing-final recovery counts must remain unknown")
        if record["final_response_available"]:
            for items, count, omitted in (
                ("unresolved_checks", "unresolved_check_count", "omitted_unresolved_checks"),
                ("recent_actions", "action_count", "omitted_actions"),
                ("diffs", "diff_count", "omitted_diffs"),
            ):
                selected = len(record[items])
                if record[count] != selected + record[omitted] or selected != min(record[count], 8):
                    raise ValueError("Recovery review counts contradict selected evidence")
        if not record["final_response_available"] and (
            any(record[name] for name in ("unresolved_checks", "recent_actions", "diffs"))
            or any(
                record.get(name) is not None
                for name in ("completion_status", "verification_status", "workspace_status")
            )
        ):
            raise ValueError("Missing-final recovery evidence must remain unknown")
        if not isinstance(record.get("objective"), dict):
            raise ValueError("Missing recovery objective evidence")
        return record

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

    def check_suggestions(self, user_id: str, session_id: str) -> dict[str, Any]:
        resp = httpx.get(
            f"{self._base}/sessions/{session_id}/check-suggestions",
            params={"user_id": user_id},
            timeout=30,
        )
        resp.raise_for_status()
        record = dict(resp.json())
        if (
            record.get("session_id") != session_id
            or record.get("status")
            not in {"inspected", "partial", "unavailable", "busy", "no_workspace"}
            or record.get("read_only") is not True
            or record.get("commands_executed") is not False
            or record.get("configuration_changed") is not False
            or not isinstance(record.get("suggestions"), list)
            or not isinstance(record.get("issues"), list)
            or not isinstance(record.get("scope_note"), str)
            or (record.get("reason") is not None and not isinstance(record["reason"], str))
        ):
            raise ValueError("Inconsistent check suggestions response")
        for suggestion in record["suggestions"]:
            if not isinstance(suggestion, dict) or any(
                not isinstance(suggestion.get(key), list)
                or not suggestion[key]
                or not all(isinstance(value, str) for value in suggestion[key])
                for key in ("argv", "evidence")
            ):
                raise ValueError("Invalid check suggestion")
        for issue in record["issues"]:
            if not isinstance(issue, dict) or any(
                not isinstance(issue.get(key), str) for key in ("path", "code")
            ):
                raise ValueError("Invalid manifest issue")
        return record

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
