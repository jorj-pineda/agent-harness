"""Thin synchronous HTTP wrapper around the agent-harness FastAPI server.

cli/ is a pure HTTP client — it never imports harness/, providers/, or tools/
directly. All agent logic stays in the server; the client calls /sessions and
/chat over HTTP.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
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
            or request["continuation_source"].get("acknowledge_uncaptured_state") is not True
            or not request.get("session_id")
            or request.get("session_id") == session_id
            or not record.get("run_id")
        ):
            raise ValueError("Mismatched continuation response; inspect the keyed run.")
        return record

    @staticmethod
    def _validate_recovery_approval(approval: dict[str, Any]) -> None:
        expected = {
            "expected_record_sha256",
            "spans",
            "message",
            "expected_archive_sha256",
            "expected_messages_sha256",
            "expected_tool_specs_sha256",
            "expected_request_output_allowance",
        }
        if not isinstance(approval, dict) or set(approval) != expected:
            raise ValueError("Approval must contain only the reviewed pins, spans and message")
        AgentClient._validate_recovery_selection(
            {name: approval[name] for name in ("expected_record_sha256", "spans")}
        )
        for name in (
            "expected_archive_sha256",
            "expected_messages_sha256",
            "expected_tool_specs_sha256",
        ):
            if (
                not isinstance(approval[name], str)
                or re.fullmatch(r"[0-9a-f]{64}", approval[name]) is None
            ):
                raise ValueError("Invalid reviewed recovery digest")
        message, allowance = approval["message"], approval["expected_request_output_allowance"]
        if (
            not isinstance(message, str)
            or not message.strip()
            or len(message.encode()) > 4096
            or type(allowance) is not int
            or not 1 <= allowance <= 1_000_000
        ):
            raise ValueError("Invalid proposed message or reviewed allowance")
        if len(json.dumps(approval, ensure_ascii=False, separators=(",", ":")).encode()) > 65536:
            raise ValueError("Approval exceeds supported size")

    def recover_run(
        self,
        user_id: str,
        run_id: str,
        approval: dict[str, Any],
        request_id: str,
        *,
        acknowledge_uncaptured_state: bool,
        acknowledge_omitted_context: bool,
        acknowledge_new_turn_budget: bool,
        provider: str | None = None,
    ) -> dict[str, Any]:
        """One explicit submission; no preview refresh, polling, retry or inferred permission."""
        self._validate_recovery_approval(approval)

        def identifier(value: object) -> bool:
            return (
                isinstance(value, str)
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value) is not None
            )

        if (
            not identifier(run_id)
            or not identifier(request_id)
            or not isinstance(user_id, str)
            or not user_id
        ):
            raise ValueError("Invalid recovery submission identity")
        acks = {
            "acknowledge_uncaptured_state": acknowledge_uncaptured_state,
            "acknowledge_omitted_context": acknowledge_omitted_context,
            "acknowledge_new_turn_budget": acknowledge_new_turn_budget,
        }
        if any(value is not True for value in acks.values()):
            raise ValueError("Three explicit recovery acknowledgments required")
        approved = {**approval, **acks}
        resp = httpx.post(
            f"{self._base}/runs/{run_id}/recover",
            json={"user_id": user_id, "request_id": request_id, "provider": provider, **approved},
            timeout=30,
        )
        resp.raise_for_status()
        if len(resp.content) > 4 * 1024 * 1024:
            raise ValueError("Recovery result exceeds supported wire size; inspect keyed run")
        record = resp.json()
        request = record.get("request") if isinstance(record, dict) else None
        origin = request.get("recovery_source") if isinstance(request, dict) else None
        if not isinstance(origin, dict):
            raise ValueError("Recovery response lacks provenance; inspect keyed run")
        raw = origin.get("approval_json")
        if not isinstance(raw, str) or len(raw.encode()) > 65536:
            raise ValueError("Invalid saved recovery approval")
        saved = json.loads(raw)
        if isinstance(saved, dict):
            self._validate_recovery_approval({k: v for k, v in saved.items() if k not in acks})
        if (
            not isinstance(saved, dict)
            or saved != approved
            or json.dumps(saved, ensure_ascii=False, separators=(",", ":"), allow_nan=False) != raw
            or origin.get("approval_sha256") != hashlib.sha256(raw.encode()).hexdigest()
            or type(saved.get("expected_request_output_allowance")) is not int
            or any(saved.get(name) is not True for name in acks)
        ):
            raise ValueError("Recovery response approval differs; inspect keyed run")
        fork_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            json.dumps(
                [
                    "agent-harness-recovery",
                    user_id,
                    run_id,
                    request_id,
                ]
            ),
        ).hex
        expected_source = {
            "session_id": origin.get("source_session_id"),
            "turn_id": origin.get("source_turn_id"),
            "acknowledge_uncaptured_state": True,
        }
        if (
            type(origin.get("schema_version")) is not int
            or origin["schema_version"] != 1
            or origin.get("draft_version") != "recovery-draft-v1"
            or origin.get("source_run_id") != run_id
            or not identifier(origin.get("source_session_id"))
            or origin["source_session_id"] == fork_id
            or not identifier(origin.get("source_turn_id"))
            or origin.get("budget_scope") != "separate-configured-turn"
            or origin.get("source_record_sha256") != approval["expected_record_sha256"]
            or origin.get("source_archive_sha256") != approval["expected_archive_sha256"]
            or origin.get("messages_sha256") != approval["expected_messages_sha256"]
            or origin.get("tool_specs_sha256") != approval["expected_tool_specs_sha256"]
            or type(origin.get("request_output_allowance")) is not int
            or origin["request_output_allowance"] != approval["expected_request_output_allowance"]
            or request.get("user_id") != user_id
            or request.get("request_id") != request_id
            or request.get("provider") != provider
            or request.get("message") != approval["message"]
            or request.get("session_id") != fork_id
            or request.get("continuation_source") != expected_source
            or request["continuation_source"].get("acknowledge_uncaptured_state") is not True
            or not identifier(record.get("run_id"))
            or record.get("run_id") == run_id
            or record.get("status")
            not in {"running", "finished", "cancelled", "failed", "interrupted"}
            or (record.get("response") is not None and not isinstance(record["response"], dict))
        ):
            raise ValueError("Mismatched recovery response identity; inspect keyed run")
        final = record.get("response")
        if (record["status"] in {"finished", "cancelled"} and final is None) or (
            final is not None
            and (
                final.get("run_id") != record["run_id"]
                or final.get("completion_status")
                not in {"completed", "incomplete", "blocked", "budget_exhausted", "cancelled"}
                or final.get("verification_status")
                not in {"not_run", "passed", "failed", "unavailable", "stale"}
            )
        ):
            raise ValueError("Contradictory recovery result; inspect keyed run")
        return dict(record)

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
        return self._validate_recovery_review(resp.json(), run_id)

    @staticmethod
    def _validate_recovery_review(record: Any, run_id: str) -> dict[str, Any]:
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

    def recovery_context(
        self, user_id: str, run_id: str, selection: dict[str, Any]
    ) -> dict[str, Any]:
        self._validate_recovery_selection(selection)
        resp = httpx.post(
            f"{self._base}/runs/{run_id}/recovery-context",
            json={"user_id": user_id, **selection},
            timeout=30,
        )
        resp.raise_for_status()
        if len(resp.content) > 65536:
            raise ValueError("Recovery context exceeds supported wire size")
        return self._validate_recovery_context(resp.json(), run_id, selection)

    @staticmethod
    def _validate_recovery_selection(selection: dict[str, Any]) -> None:
        def digest(value: object) -> bool:
            return (
                isinstance(value, str)
                and len(value) == 64
                and all(char in "0123456789abcdef" for char in value)
            )

        if (
            set(selection) != {"expected_record_sha256", "spans"}
            or not digest(selection.get("expected_record_sha256"))
            or not isinstance(selection.get("spans"), list)
            or len(selection["spans"]) > 4
        ):
            raise ValueError("Invalid recovery context selection")
        requested = selection["spans"]
        for i, item in enumerate(requested):
            if (
                not isinstance(item, dict)
                or set(item) != {"path", "expected_sha256", "start_line", "end_line"}
                or not isinstance(item["path"], str)
                or not 1 <= len(item["path"]) <= 1024
                or not digest(item["expected_sha256"])
                or type(item["start_line"]) is not int
                or type(item["end_line"]) is not int
                or not 1 <= item["start_line"] <= item["end_line"] <= 1_048_576
                or item["end_line"] - item["start_line"] + 1 > 200
            ):
                raise ValueError("Invalid recovery context span selection")
            for previous in requested[:i]:
                if item["path"] == previous["path"] and (
                    item["expected_sha256"] != previous["expected_sha256"]
                    or max(item["start_line"], previous["start_line"])
                    <= min(item["end_line"], previous["end_line"])
                ):
                    raise ValueError("Conflicting recovery context selections")

    def _validate_recovery_context(
        self, data: Any, run_id: str, selection: dict[str, Any]
    ) -> dict[str, Any]:
        def digest(value: object) -> bool:
            return (
                isinstance(value, str)
                and len(value) == 64
                and all(char in "0123456789abcdef" for char in value)
            )

        requested = selection["spans"]
        if (
            not isinstance(data, dict)
            or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1
            or data.get("read_only") is not True
            or data.get("execution_supported") is not False
            or type(data.get("max_serialized_bytes")) is not int
            or data["max_serialized_bytes"] != 65536
            or not isinstance(data.get("scope_note"), str)
            or not digest(data.get("evidence_sha256"))
            or not isinstance(data.get("spans"), list)
            or len(data["spans"]) != len(requested)
            or type(data.get("selected_source_bytes")) is not int
            or not 0 <= data["selected_source_bytes"] <= 16384
        ):
            raise ValueError("Unsupported recovery context response")
        review = self._validate_recovery_review(data.get("review"), run_id)
        if review["source_record_sha256"] != selection["expected_record_sha256"]:
            raise ValueError("Mismatched recovery context run hash")
        source_bytes = 0
        checkpoint_identity = None
        file_metadata: dict[str, tuple[Any, ...]] = {}
        for span, item in zip(data["spans"], requested, strict=True):
            if (
                not isinstance(span, dict)
                or span.get("checkpoint_kind") != "turn_after"
                or not isinstance(span.get("session_id"), str)
                or not span["session_id"]
                or not isinstance(span.get("checkpoint_id"), str)
                or not span["checkpoint_id"]
            ):
                raise ValueError("Invalid final-checkpoint source identity")
            self._validate_checkpoint_span(
                span,
                span["session_id"],
                span["checkpoint_id"],
                item["path"],
                item["expected_sha256"],
                item["start_line"],
                item["end_line"],
            )
            identity = (span["session_id"], span["checkpoint_id"], span["checkpoint_record_sha256"])
            if checkpoint_identity is not None and identity != checkpoint_identity:
                raise ValueError("Mixed final-checkpoint source identities")
            checkpoint_identity = identity
            metadata = (span["file_sha256"], span["file_bytes"], span["total_lines"], span["mode"])
            if span["path"] in file_metadata and metadata != file_metadata[span["path"]]:
                raise ValueError("Contradictory saved file metadata across ranges")
            file_metadata[span["path"]] = metadata
            source_bytes += span["selected_bytes"]
        if source_bytes != data["selected_source_bytes"]:
            raise ValueError("Contradictory recovery context source totals")

        def compact(value: object) -> str:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

        evidence = compact(review) + "\n" + "\n".join(compact(s) for s in data["spans"])
        if hashlib.sha256(evidence.encode()).hexdigest() != data["evidence_sha256"]:
            raise ValueError("Recovery context evidence hash mismatch")
        return data

    def recovery_draft(
        self,
        user_id: str,
        run_id: str,
        selection: dict[str, Any],
        message: str,
        *,
        provider: str | None = None,
    ) -> dict[str, Any]:
        self._validate_recovery_selection(selection)
        if not isinstance(message, str) or not message.strip() or len(message.encode()) > 4096:
            raise ValueError("Use a nonblank proposed message of at most 4096 UTF-8 bytes")
        payload = {"user_id": user_id, **selection, "message": message}
        if provider is not None:
            payload["provider"] = provider
        resp = httpx.post(f"{self._base}/runs/{run_id}/recovery-draft", json=payload, timeout=30)
        resp.raise_for_status()
        if len(resp.content) > 131072:
            raise ValueError("Recovery draft exceeds supported wire size")
        data = resp.json()
        if (
            not isinstance(data, dict)
            or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1
            or data.get("draft_version") != "recovery-draft-v1"
            or data.get("run_id") != run_id
            or data.get("source_record_sha256") != selection["expected_record_sha256"]
            or data.get("budget_scope") != "hypothetical-new-turn"
            or data.get("read_only") is not True
            or data.get("execution_supported") is not False
            or type(data.get("max_serialized_bytes")) is not int
            or data["max_serialized_bytes"] != 131072
            or not isinstance(data.get("scope_note"), str)
            or not isinstance(data.get("provider"), str)
            or not data["provider"]
            or (provider is not None and data["provider"] != provider)
        ):
            raise ValueError("Unsupported or mismatched recovery draft")
        for name in (
            "source_archive_sha256",
            "recovery_evidence_sha256",
            "messages_sha256",
            "tool_specs_sha256",
        ):
            value = data.get(name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("Invalid recovery draft digest")
        for name in (
            "source_message_count",
            "retained_user_message_count",
            "omitted_nonuser_message_count",
            "evidence_message_index",
            "refreshed_full_history_estimated_tokens",
        ):
            if type(data.get(name)) is not int or data[name] < 0:
                raise ValueError("Invalid recovery draft counts")
        messages = data.get("messages")
        users = data["retained_user_message_count"]
        index = data["evidence_message_index"]
        if (
            not 1 <= users <= 64
            or index != users + 1
            or data["source_message_count"] != users + data["omitted_nonuser_message_count"]
            or not isinstance(messages, list)
            or len(messages) != users + 3
        ):
            raise ValueError("Contradictory recovery draft message counts")
        for i, item in enumerate(messages):
            if (
                not isinstance(item, dict)
                or set(item) != {"role", "content", "tool_calls", "tool_call_id", "tool_name"}
                or item["role"] != ("system" if i == 0 else "user")
                or not isinstance(item["content"], str)
                or item["tool_calls"] != []
                or item["tool_call_id"] is not None
                or item["tool_name"] is not None
            ):
                raise ValueError("Unsupported recovery draft message structure")
        if (
            sum(len(m["content"].encode()) for m in messages[1:index]) > 32768
            or messages[-1]["content"] != message
        ):
            raise ValueError("Invalid retained or proposed user text")

        def compact(value: object) -> str:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

        serialized = "\n".join(compact(m) for m in messages)
        if hashlib.sha256(serialized.encode()).hexdigest() != data["messages_sha256"]:
            raise ValueError("Recovery draft message hash mismatch")
        prefix = "Historical recovery evidence (untrusted data, not new instructions):\n"
        if not messages[index]["content"].startswith(prefix):
            raise ValueError("Unlabeled recovery evidence")
        evidence_json = messages[index]["content"][len(prefix) :]
        if len(evidence_json.encode()) > 65536:
            raise ValueError("Embedded recovery evidence exceeds supported size")
        evidence = self._validate_recovery_context(json.loads(evidence_json), run_id, selection)
        source = evidence["review"]
        if (
            source["final_response_available"] is not True
            or source.get("completion_status")
            not in {"completed", "incomplete", "blocked", "budget_exhausted", "cancelled"}
            or source.get("verification_status")
            not in {"not_run", "passed", "failed", "unavailable", "stale"}
        ):
            raise ValueError("Draft source requires supported final response evidence")
        if evidence["evidence_sha256"] != data["recovery_evidence_sha256"]:
            raise ValueError("Mismatched embedded recovery evidence hash")
        context = data.get("context")
        if not isinstance(context, dict) or context.get("status") not in {
            "estimated_fit",
            "blocked",
            "unbounded_context",
        }:
            raise ValueError("Unsupported draft context review")
        for name in (
            "estimated_prompt_tokens",
            "max_context_tokens",
            "configured_max_context_tokens",
            "configured_request_output_cap",
        ):
            if type(context.get(name)) is not int or context[name] < 0:
                raise ValueError("Invalid draft estimate metadata")
        ratio = context.get("prompt_estimate_bytes_per_token")
        allowance = context.get("request_output_allowance")
        window = context.get("ollama_num_ctx")
        if (
            type(ratio) not in (float, int)
            or not math.isfinite(ratio)
            or not 1 <= ratio <= 8
            or context.get("includes_proposed_message") is not True
            or context.get("context_limit_policy") != "configured-min-v1"
            or (allowance is not None and (type(allowance) is not int or allowance < 1))
            or (window is not None and (type(window) is not int or window < 256))
            or (
                context["status"] == "estimated_fit"
                and (context["max_context_tokens"] == 0 or allowance is None)
            )
            or (context["status"] == "unbounded_context" and context["max_context_tokens"] != 0)
            or (
                context["status"] == "blocked"
                and (
                    allowance is not None
                    or not isinstance(context.get("reason"), str)
                    or not context["reason"]
                )
            )
            or (context["status"] != "blocked" and context.get("reason") is not None)
        ):
            raise ValueError("Contradictory draft context status")
        configuration = data.get("configuration")
        if not isinstance(configuration, dict) or configuration.get("status") not in {
            "matching",
            "changed",
            "unknown",
            "not_recorded",
            "unavailable",
        }:
            raise ValueError("Unsupported draft configuration review")
        for name in ("changed_fields", "unknown_fields"):
            if not isinstance(configuration.get(name), list) or any(
                not isinstance(v, str) for v in configuration[name]
            ):
                raise ValueError("Invalid draft configuration fields")
        if (
            (configuration["status"] == "changed") != bool(configuration["changed_fields"])
            or (configuration["status"] == "matching" and configuration["unknown_fields"])
            or (configuration["status"] == "unknown" and not configuration["unknown_fields"])
        ):
            raise ValueError("Contradictory draft configuration status")
        instructions = data.get("repository_instructions")
        if (
            not isinstance(instructions, dict)
            or instructions.get("policy") != "root-agents-v1"
            or instructions.get("path") != "AGENTS.md"
            or type(instructions.get("max_bytes")) is not int
            or instructions["max_bytes"] != 8192
            or instructions.get("issue") is not None
            or instructions.get("status") not in {"loaded", "disabled", "no_workspace", "missing"}
        ):
            raise ValueError("Unsupported draft root instructions")
        if instructions["status"] == "loaded":
            size, digest = instructions.get("content_bytes"), instructions.get("content_sha256")
            if (
                type(size) is not int
                or not 0 <= size <= 8192
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("Invalid loaded instruction identity")
        elif (
            instructions.get("content_bytes") is not None
            or instructions.get("content_sha256") is not None
        ):
            raise ValueError("Contradictory unloaded instructions")
        return data

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

    def checkpoint_span(
        self,
        user_id: str,
        session_id: str,
        checkpoint_id: str,
        *,
        path: str,
        expected_sha256: str,
        start_line: int,
        end_line: int,
    ) -> dict[str, Any]:
        def digest(value: object) -> bool:
            return (
                isinstance(value, str)
                and len(value) == 64
                and all(char in "0123456789abcdef" for char in value)
            )

        if (
            not digest(expected_sha256)
            or type(start_line) is not int
            or type(end_line) is not int
            or not 1 <= start_line <= end_line <= 1_048_576
            or end_line - start_line + 1 > 200
        ):
            raise ValueError("Use a saved file SHA-256 and an explicit range of at most 200 lines")
        resp = httpx.get(
            f"{self._base}/sessions/{session_id}/checkpoints/{checkpoint_id}/span",
            params={
                "user_id": user_id,
                "path": path,
                "expected_sha256": expected_sha256,
                "start_line": start_line,
                "end_line": end_line,
            },
            timeout=30,
        )
        resp.raise_for_status()
        if len(resp.content) > 32768:
            raise ValueError("Checkpoint span response exceeds the supported size")
        return self._validate_checkpoint_span(
            resp.json(), session_id, checkpoint_id, path, expected_sha256, start_line, end_line
        )

    @staticmethod
    def _validate_checkpoint_span(
        data: Any,
        session_id: str,
        checkpoint_id: str,
        path: str,
        expected_sha256: str,
        start_line: int,
        end_line: int,
    ) -> dict[str, Any]:
        def digest(value: object) -> bool:
            return (
                isinstance(value, str)
                and len(value) == 64
                and all(char in "0123456789abcdef" for char in value)
            )

        if (
            not isinstance(data, dict)
            or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1
            or data.get("session_id") != session_id
            or data.get("checkpoint_id") != checkpoint_id
            or data.get("path") != path
            or data.get("file_sha256") != expected_sha256
            or data.get("start_line") != start_line
            or data.get("end_line") != end_line
            or data.get("checkpoint_kind") not in {"manual", "turn_before", "turn_after"}
            or data.get("read_only") is not True
            or type(data.get("has_more")) is not bool
            or not digest(data.get("checkpoint_record_sha256"))
            or not isinstance(data.get("scope_note"), str)
            or not isinstance(data.get("content"), str)
        ):
            raise ValueError("Unsupported or mismatched checkpoint span response")
        for name, minimum, maximum in (
            ("file_bytes", 1, 1_048_576),
            ("selected_bytes", 1, 8192),
            ("total_lines", end_line, 1_048_576),
            ("mode", 0, 0o7777),
            ("start_line", 1, 1_048_576),
            ("end_line", 1, 1_048_576),
        ):
            value = data.get(name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError("Invalid checkpoint span counts or range")
        selected = data["content"].encode("utf-8")
        line_count = selected.count(b"\n") + (not selected.endswith(b"\n"))
        if (
            data["selected_bytes"] != len(selected)
            or data.get("content_sha256") != hashlib.sha256(selected).hexdigest()
            or line_count != end_line - start_line + 1
            or data["selected_bytes"] > data["file_bytes"]
            or data["total_lines"] > data["file_bytes"]
            or data["has_more"] != (end_line < data["total_lines"])
            or (data["has_more"] and not selected.endswith(b"\n"))
            or (
                start_line == 1
                and end_line == data["total_lines"]
                and (
                    len(selected) != data["file_bytes"]
                    or data["content_sha256"] != data["file_sha256"]
                )
            )
        ):
            raise ValueError("Checkpoint span content/counts contradict saved metadata")
        return data

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
