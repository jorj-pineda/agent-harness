"""Typer CLI entry point for agent-harness.

Commands
--------
serve   Start the FastAPI server (uvicorn wrapper).
chat    Interactive REPL against a running server.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
import uuid
from pathlib import Path
from typing import Any

import typer

from cli._client import AgentClient, TurnSummary, summarize

app = typer.Typer(
    name="agent-harness",
    help="Local coding agent — harness CLI.",
    no_args_is_help=True,
)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
    reload: bool = typer.Option(False, "--reload/--no-reload", help="Hot-reload on changes."),
) -> None:
    """Start the FastAPI server (uvicorn wrapper)."""
    import uvicorn  # late import so --help works without spinning up uvicorn

    uvicorn.run("api.server:app", host=host, port=port, reload=reload)


@app.command()
def cancel(
    session_id: str = typer.Argument(..., help="Session with an active turn."),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    user_id: str = typer.Option("cli-user", "--user-id"),
) -> None:
    """Request cancellation; final review is saved after owned work settles."""
    try:
        detail = AgentClient(base_url).cancel(user_id, session_id)
    except Exception as exc:
        typer.echo(f"Cancellation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(detail)


@app.command()
def chat(
    base_url: str = typer.Option("http://localhost:8000", "--base-url", help="API base URL."),
    user_id: str = typer.Option("cli-user", "--user-id", help="Stable user identifier."),
    checkpoints: bool = typer.Option(
        False, "--checkpoints", help="Save bounded plaintext workspace captures before/after turns."
    ),
    inspection_only: bool = typer.Option(
        False,
        "--inspection-only",
        help="Restrict model tools to file inspection and memory recall.",
    ),
    workspace: str | None = typer.Option(
        None, "--workspace", help="Absolute path to repo root (sets workspace_root on session)."
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Provider override: ollama | anthropic | openai | openai_compatible.",
    ),
    response_timeout: int = typer.Option(
        120,
        "--response-timeout",
        min=1,
        max=3600,
        help="Seconds of HTTP read inactivity allowed for chat. Does not change server budgets.",
    ),
) -> None:
    """Interactive REPL — connect to a running agent-harness server and chat."""
    client = AgentClient(base_url, response_timeout=response_timeout)
    try:
        options = {}
        if checkpoints:
            options["checkpoint_turns"] = True
        if inspection_only:
            options["inspection_only"] = True
        session_id = client.create_session(user_id, workspace, **options)
    except Exception as exc:
        typer.echo(f"Error: could not create session — {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Connected  session={session_id}")
    if inspection_only:
        typer.echo(
            "Tool policy: inspection only (server acknowledged). No model commands, file edits "
            "or memory writes. Not host process isolation; checks may remain not_run."
        )
    typer.echo(
        "Type your message and press Enter. /check and /retry recover a lost response; "
        "/quit exits.\n"
    )

    pending: tuple[str, str] | None = None
    while True:
        typer.echo("> ", nl=False)
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            typer.echo("\nBye.")
            break

        if not line:  # EOF (Ctrl-D)
            typer.echo("\nBye.")
            break

        message = line.rstrip("\n").strip()
        if message == "/quit":
            typer.echo("Bye.")
            break
        if not message:
            continue

        recovering = pending is not None
        if message == "/check":
            if pending is None:
                typer.echo("No uncertain submission.")
                continue
            try:
                records = client.runs(user_id, request_id=pending[0])
                if not records:
                    typer.echo("No saved run found. /retry uses the same submission key.")
                    continue
                show_run(records[0])
                if records[0]["status"] != "running":
                    typer.echo("Review only. Start a new session after inspecting the workspace.")
                    break
            except Exception as exc:
                typer.echo(f"Run lookup failed: {exc}", err=True)
            continue
        if pending is not None:
            if message != "/retry":
                typer.echo(
                    "Outcome unknown. Use /check, /retry, or /quit before starting new work."
                )
                continue
            request_id, message = pending
        else:
            if message == "/retry":
                typer.echo("No uncertain submission.")
                continue
            request_id = uuid.uuid4().hex
            pending = (request_id, message)
        typer.echo(f"  request_id={request_id} (retain this key if the response is lost)")
        try:
            result = client.chat(user_id, session_id, message, provider, request_id=request_id)
        except KeyboardInterrupt:
            try:
                typer.echo(client.cancel(user_id, session_id))
            except Exception as exc:
                typer.echo(f"Cancellation not confirmed: {exc}", err=True)
            typer.echo("Inspect saved session review after cleanup. Bye.")
            break
        except Exception as exc:
            typer.echo(f"  error: {exc}", err=True)
            typer.echo(
                "Outcome unknown. /check reads saved state; "
                "/retry repeats only this keyed submission."
            )
            continue

        pending = None
        show_turn(result)
        if recovering:
            typer.echo(
                "Recovered result is for review. "
                "Start a new session after inspecting the workspace."
            )
            break
        if result.completion_status == "cancelled":
            typer.echo("Cancelled session is read-only. Start a new session after review.")
            break


def show_ignored_entries(data: dict[str, Any] | None) -> None:
    summary = data or {}
    typer.echo(f"  ignored top-level metadata: {summary.get('status', 'not_recorded')}")
    for kind in ("added", "modified", "deleted"):
        if summary.get(kind) or summary.get(kind + "_count"):
            typer.echo(
                f"    {kind}: {summary.get(kind, [])} "
                f"(total {summary.get(kind + '_count', 'unknown')})"
            )
    if summary.get("paths_truncated"):
        typer.echo("    Ignored-entry list truncated; counts cover the full metadata comparison.")
    typer.echo(summary.get("scope_note", "Metadata only; ignored contents are not compared."))


def show_turn(result: TurnSummary) -> None:
    if result.run_id:
        typer.echo(f"  run_id={result.run_id}")
    typer.echo(f"\n{result.answer}\n")
    typer.echo(f"  status={result.completion_status} checks={result.verification_status}")
    if result.completion_reason:
        typer.echo(f"  reason={result.completion_reason}")
    typer.echo(
        f"  escalated={result.escalated}"
        f"  provider={result.provider}"
        f"  latency={result.latency_ms:.0f}ms"
    )
    typer.echo(
        f"  workspace_review={result.workspace_review_ms:.0f}ms (separate from loop latency)"
        if result.workspace_review_ms is not None
        else "  workspace_review=unmeasured (legacy/low-level/blocked response)"
    )
    typer.echo("  Completion records how the run ended; passed checks cover only the checks run.")
    if result.files_touched:
        typer.echo(f"  files_touched={result.files_touched}")
    changes = result.workspace_changes
    if changes.get("status") == "tracked":
        for kind in ("added", "modified", "deleted"):
            if changes.get(kind):
                typer.echo(f"  workspace_{kind}={changes[kind]}")
    elif changes.get("status") == "unavailable":
        typer.echo(f"  workspace_changes=unavailable ({changes.get('reason')})")
    if changes.get("status") != "not_tracked":
        show_ignored_entries(changes.get("ignored_entries"))
    for entry in changes.get("diffs", []):
        typer.echo(f"  turn diff: {entry['path']}")
        if entry.get("reason"):
            typer.echo(f"  diff omitted: {entry['reason']}")
        else:
            typer.echo(entry.get("diff", ""), nl=False)
    if result.token_usage:
        typer.echo(f"  token_usage={result.token_usage}")
    for check in result.check_attempts:
        typer.echo(
            f"  check={check.get('argv')} status={check.get('status')} "
            f"exit={check.get('exit_code')} relevant={check.get('relevant')} "
            f"stale={check.get('superseded_by_edit')}"
        )
        if check.get("error"):
            typer.echo(f"  check_error={check['error']}")
    for error in result.tool_errors:
        typer.echo(f"  tool_error={error}")
    score = "n/a" if result.confidence is None else f"{result.confidence:.2f}"
    typer.echo(f"  legacy_evidence_score={score} (tool heuristic; not patch correctness)")
    typer.echo()


def show_profile_selection(selection: object) -> None:
    """Current server selection only, including explicit direct/unknown distinctions."""
    if selection is None:
        typer.echo("current_model_profile=not_checked_or_legacy_response")
    elif (
        isinstance(selection, dict)
        and selection.get("status") == "direct_settings"
        and selection.get("identity") is None
    ):
        typer.echo("current_model_profile=direct_settings (no named profile selected)")
    else:
        identity = selection.get("identity") if isinstance(selection, dict) else None
        valid = (
            isinstance(selection, dict)
            and selection.get("status") == "selected"
            and isinstance(identity, dict)
            and identity.get("schema_version") == 1
            and all(
                isinstance(identity.get(key), str)
                for key in ("name", "provider", "model", "definition_sha256")
            )
            and len(identity["definition_sha256"]) == 64
            and all(char in "0123456789abcdef" for char in identity["definition_sha256"])
        )
        if valid:
            typer.echo("current_model_profile=selected")
            for key in ("name", "provider", "model", "definition_sha256"):
                typer.echo(f"  {key}={json.dumps(identity[key])}")
        else:
            typer.echo("current_model_profile=unknown_or_inconsistent")
    typer.echo(
        "Current operator configuration only; no switching, inference or settings changes. "
        "Not backend/capability evidence or continuation permission."
    )


def show_repository_instructions(evidence: object, *, saved: bool = False) -> None:
    """Render only known instruction metadata; contents and policy stay server-owned."""
    prefix = "saved" if saved else "current"
    label = f"{prefix}_repository_instructions"
    if evidence is None:
        typer.echo(f"{label}={'unrecorded' if saved else 'not_checked_or_legacy_response'}")
    else:
        status = evidence.get("status") if isinstance(evidence, dict) else None
        valid = (
            isinstance(evidence, dict)
            and evidence.get("policy") == "root-agents-v1"
            and evidence.get("path") == "AGENTS.md"
            and evidence.get("max_bytes") == 8192
            and isinstance(status, str)
            and status in {"disabled", "no_workspace", "missing", "loaded", "unavailable"}
        )
        if valid:
            assert isinstance(evidence, dict)
            size = evidence.get("content_bytes")
            digest = evidence.get("content_sha256")
            issue = evidence.get("issue")
            if status == "loaded":
                valid = (
                    type(size) is int
                    and 0 <= size <= 8192
                    and isinstance(digest, str)
                    and re.fullmatch(r"[a-f0-9]{64}", digest) is not None
                    and issue is None
                )
            else:
                valid = (
                    size is None
                    and digest is None
                    and (
                        isinstance(issue, str)
                        and issue
                        in {"unsafe_or_unreadable", "too_large", "changed", "invalid_utf8"}
                        if status == "unavailable"
                        else issue is None
                    )
                )
        if valid:
            assert isinstance(evidence, dict)
            typer.echo(f"{label}={status} path=AGENTS.md policy=root-agents-v1 max_bytes=8192")
            if status == "loaded":
                typer.echo(
                    f"  content_bytes={evidence['content_bytes']} "
                    f"content_sha256={evidence['content_sha256']}"
                )
            elif status == "unavailable":
                typer.echo(f"  issue={evidence['issue']}")
        else:
            typer.echo(f"{label}=unknown_or_inconsistent")
    typer.echo(
        f"{'Saved run evidence' if saved else 'Current root review'} only; "
        "no parent/nested discovery, instruction contents or settings changes. "
        "No commands, inference or continuation permission. "
        "Enabled runs retain injected text in transcripts/traces."
    )


def show_context_limits(context: dict[str, Any]) -> None:
    """Render server evidence; never derive or apply a budget in the client."""

    def value(name: str) -> str:
        item = context.get(name)
        return "unrecorded" if item is None else json.dumps(item)

    if context.get("context_limit_policy") == "configured-min-v1":
        typer.echo(
            f"effective_context_limit={value('max_context_tokens')} (0 = uncapped by harness)"
        )
    else:
        typer.echo(
            "effective_context_limit=unknown (policy unrecorded or unsupported); "
            f"reported_context_limit={value('max_context_tokens')}"
        )
    typer.echo(
        f"configured_harness_context_limit={value('configured_max_context_tokens')} "
        "(0 disables only this cap)"
    )
    typer.echo(
        f"ollama_requested_context_limit={value('ollama_num_ctx')} "
        "(unrecorded: unset, not applicable or unknown)"
    )
    typer.echo(
        f"configured_request_output_cap={value('configured_request_output_cap')} "
        "(0 disables only this cap)"
    )
    typer.echo(f"context_limit_policy={value('context_limit_policy')}")
    typer.echo(f"prompt_estimate_bytes_per_token={value('prompt_estimate_bytes_per_token')}")
    typer.echo(
        "Manual byte heuristic; higher ratios estimate fewer tokens. "
        "Estimated fit is not backend-attested/tokenizer fit or permission to continue."
    )


def show_run(record: dict) -> None:
    typer.echo(f"run_id={record['run_id']} status={record['status']}")
    if record.get("before_checkpoint_id") or record.get("after_checkpoint_id"):
        typer.echo(
            f"before_checkpoint={record.get('before_checkpoint_id') or 'unavailable'} "
            f"after_checkpoint={record.get('after_checkpoint_id') or 'unavailable'} "
            "(scoped content captures; whole-checkpoint restore unsupported)"
        )
    typer.echo(f"task={record['request'].get('message', '')}")
    typer.echo(
        f"session={record['request']['session_id']} request_id={record['request']['request_id']}"
    )
    typer.echo(
        f"provider={record['provider']} model={record.get('configured_model') or 'unrecorded'}"
    )
    profile = record.get("execution_profile")
    show_tool_policy(profile.get("inspection_only") if isinstance(profile, dict) else None)
    if profile is None:
        typer.echo("execution_profile=unrecorded (configuration not captured in this record)")
    else:
        limits = profile["settings"]
        show_repository_instructions(profile.get("repository_instructions"), saved=True)
        typer.echo(
            f"execution_profile=v{profile['schema_version']} "
            f"prompt={profile.get('prompt_version') or 'custom'} "
            f"toolset={limits['coding_toolset']} tools={len(profile['tool_names'])}"
        )
        typer.echo(
            f"  configured_context_limit={limits['max_context_tokens']} "
            f"output_budget={limits['max_completion_tokens_per_turn']} "
            f"tool_budget={limits['max_tool_calls_per_turn']} "
            "(0 disables the corresponding configured cap)"
        )
        show_context_limits(
            {
                "max_context_tokens": profile.get("effective_max_context_tokens"),
                "configured_max_context_tokens": limits.get("max_context_tokens"),
                "ollama_num_ctx": profile.get("ollama_num_ctx"),
                "context_limit_policy": profile.get("context_limit_policy"),
                "prompt_estimate_bytes_per_token": limits.get("prompt_estimate_bytes_per_token"),
                "configured_request_output_cap": limits.get("max_request_output_tokens"),
            }
        )
        typer.echo("  Configured settings only; inspect --json for hashes and full limits.")
    if record.get("error"):
        typer.echo(record["error"])
    if record.get("response") is not None:
        show_turn(summarize(record["response"]))
    else:
        typer.echo("No saved final response. Workspace edits may exist; inspect before new work.")


def show_tool_policy(value: object) -> None:
    """Render captured policy without interpreting missing or malformed legacy evidence."""
    if value is True:
        typer.echo(
            "saved_tool_policy=inspection_only (model file inspection and recall; "
            "no commands, file edits or memory writes)"
        )
    elif value is False:
        typer.echo("saved_tool_policy=standard")
    else:
        typer.echo("saved_tool_policy=unrecorded_or_unsupported")
    typer.echo("Historical tool policy only; not host isolation or permission to continue.")


@app.command()
def memory(
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    user_id: str = typer.Option("cli-user", "--user-id"),
    limit: int = typer.Option(20, "--limit", min=1, max=50),
    before_id: int | None = typer.Option(None, "--before-id", min=1),
    namespace: str = typer.Option("all", "--namespace", help="all, global or workspace"),
    workspace_root: str | None = typer.Option(None, "--workspace-root"),
    raw: bool = typer.Option(False, "--json"),
) -> None:
    """Inspect bounded user-memory previews; never submits a model turn."""
    if namespace not in {"all", "global", "workspace"}:
        raise typer.BadParameter("--namespace must be all, global or workspace.")
    if (namespace == "workspace") != (workspace_root is not None):
        raise typer.BadParameter("--workspace-root is required only with --namespace workspace.")
    try:
        page = AgentClient(base_url).memory(
            user_id,
            limit=limit,
            before_id=before_id,
            namespace=namespace,
            workspace_root=workspace_root,
        )
    except Exception as exc:
        typer.echo(f"Memory lookup failed ({type(exc).__name__}).", err=True)
        raise typer.Exit(code=1) from exc
    if raw:
        typer.echo(json.dumps(page, indent=2))
        return
    typer.echo(f"Namespace: {namespace}; workspace={json.dumps(workspace_root)}")
    for note in page["notes"]:
        typer.echo(
            f"id={note['id']} workspace={json.dumps(note.get('workspace_root'))} "
            f"bytes={note['fact_bytes']} truncated={note['preview_truncated']} "
            f"sha256={note['fact_sha256']}"
        )
        # Quote previews to prevent remembered control characters from driving a terminal.
        typer.echo(json.dumps(note["preview"]))
        typer.echo(
            f"created={json.dumps(note['created_at'])} "
            f"updated={json.dumps(note.get('updated_at'))} "
            f"source_turn={json.dumps(note.get('source_turn_id'))}"
        )
    if not page["notes"]:
        typer.echo("No remembered notes found.")
    if page.get("next_before_id") is not None:
        typer.echo(
            f"Next page: --before-id {page['next_before_id']} "
            "(keep the same namespace/root filters)"
        )
    typer.echo(page["scope_note"])


@app.command()
def memory_scope(
    fact_id: int = typer.Argument(..., min=1),
    expected_sha256: str = typer.Option(..., "--expected-sha256"),
    from_global: bool = typer.Option(False, "--from-global"),
    from_workspace: str | None = typer.Option(None, "--from-workspace"),
    to_global: bool = typer.Option(False, "--to-global"),
    to_workspace: str | None = typer.Option(None, "--to-workspace"),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    user_id: str = typer.Option("cli-user", "--user-id"),
) -> None:
    """Reclassify one reviewed note; select explicit source and destination scopes."""
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise typer.BadParameter("Use the 64-character lowercase hash from memory inspection.")
    if from_global == (from_workspace is not None):
        raise typer.BadParameter("Choose exactly one of --from-global or --from-workspace.")
    if to_global == (to_workspace is not None):
        raise typer.BadParameter("Choose exactly one of --to-global or --to-workspace.")
    try:
        record = AgentClient(base_url).reclassify_memory(
            user_id, fact_id, expected_sha256, from_workspace, to_workspace
        )
    except Exception as exc:
        typer.echo(
            f"Memory reclassification failed ({type(exc).__name__}). Outcome may be unknown; "
            "inspect memory before retrying. No automatic retry.",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Memory note {fact_id} scope={json.dumps(record['note']['workspace_root'])}. "
        f"{record['scope_note']}"
    )


@app.command()
def memory_note(
    fact_id: int = typer.Argument(..., min=1),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    user_id: str = typer.Option("cli-user", "--user-id"),
    raw: bool = typer.Option(False, "--json"),
) -> None:
    """Inspect one complete note up to 4,096 UTF-8 bytes, without submitting work."""
    try:
        record = AgentClient(base_url).memory_note(user_id, fact_id)
    except Exception as exc:
        typer.echo(
            f"Memory detail lookup failed ({type(exc).__name__}). "
            "Use memory for bounded previews, including oversized legacy notes. "
            "No automatic retry.",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    if raw:
        typer.echo(json.dumps(record, indent=2))
        return
    note = record["note"]
    typer.echo(
        f"id={note['id']} workspace={json.dumps(note.get('workspace_root'))} "
        f"bytes={note['fact_bytes']} sha256={note['fact_sha256']}"
    )
    typer.echo(json.dumps(record["text"]))
    typer.echo(
        f"created={json.dumps(note['created_at'])} "
        f"updated={json.dumps(note.get('updated_at'))} "
        f"source_turn={json.dumps(note.get('source_turn_id'))}"
    )
    typer.echo(record["scope_note"])


@app.command()
def memory_delete(
    fact_id: int = typer.Argument(..., min=1),
    expected_sha256: str = typer.Option(..., "--expected-sha256"),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    user_id: str = typer.Option("cli-user", "--user-id"),
) -> None:
    """Delete a whole note using its reviewed full-content hash, even if preview is truncated."""
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise typer.BadParameter("Use the 64-character lowercase hash from memory inspection.")
    try:
        record = AgentClient(base_url).delete_memory(user_id, fact_id, expected_sha256)
    except Exception as exc:
        typer.echo(
            f"Memory deletion failed ({type(exc).__name__}). Outcome may be unknown; "
            "inspect memory before retrying. No automatic retry.",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(f"Deleted memory note {record['fact_id']}. {record['scope_note']}")


@app.command()
def memory_correct(
    fact_id: int = typer.Argument(..., min=1),
    replacement: str | None = typer.Argument(None),
    replacement_file: Path | None = typer.Option(None, "--replacement-file"),
    expected_sha256: str = typer.Option(..., "--expected-sha256"),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    user_id: str = typer.Option("cli-user", "--user-id"),
) -> None:
    """Replace a whole reviewed note; server enforces UTF-8 size/conflict guards."""
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise typer.BadParameter("Use the 64-character lowercase hash from memory inspection.")
    if (replacement is None) == (replacement_file is None):
        raise typer.BadParameter("Supply exactly one replacement argument or --replacement-file.")
    if replacement_file is not None:
        try:
            fd = os.open(replacement_file, os.O_RDONLY | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as source:
                if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                    raise ValueError("Replacement input must be a regular file.")
                raw = source.read(8193)
            if len(raw) > 8192:
                raise ValueError("Replacement file exceeds 8,192 bytes.")
            replacement = raw.decode("utf-8")
        except (OSError, UnicodeError, ValueError) as exc:
            raise typer.BadParameter(
                "Cannot read replacement file: use a regular UTF-8 file of at most 8,192 bytes."
            ) from exc
    assert replacement is not None
    try:
        record = AgentClient(base_url).correct_memory(
            user_id, fact_id, expected_sha256, replacement
        )
    except Exception as exc:
        typer.echo(
            f"Memory correction failed ({type(exc).__name__}). Outcome may be unknown; "
            "inspect memory before retrying. No automatic retry.",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(f"Corrected memory note {record['note']['id']}. {record['scope_note']}")


@app.command()
def runs(
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    user_id: str = typer.Option("cli-user", "--user-id"),
    session_id: str | None = typer.Option(None, "--session-id"),
    request_id: str | None = typer.Option(None, "--request-id"),
    limit: int = typer.Option(50, "--limit", min=1, max=100),
    offset: int = typer.Option(0, "--offset", min=0),
) -> None:
    """List saved run identities; this never submits work or resumes a session."""
    try:
        records = AgentClient(base_url).runs(
            user_id, session_id=session_id, request_id=request_id, limit=limit, offset=offset
        )
    except Exception as exc:
        typer.echo(f"Run lookup failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for record in records:
        response = record.get("response") or {}
        typer.echo(
            f"{record['run_id']} {record['status']} "
            f"completion={response.get('completion_status', 'unknown')} "
            f"checks={response.get('verification_status', 'unknown')} "
            f"request_id={record['request']['request_id']} "
            f"session={record['request']['session_id']}"
        )
    if not records:
        typer.echo("No saved runs found.")


@app.command()
def run(
    run_id: str = typer.Argument(...),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    user_id: str = typer.Option("cli-user", "--user-id"),
    raw: bool = typer.Option(False, "--json", help="Print the complete saved record."),
) -> None:
    """Inspect a saved run and bounded review evidence without executing tools."""
    try:
        record = AgentClient(base_url).run(user_id, run_id)
    except Exception as exc:
        typer.echo(f"Run lookup failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if raw:
        typer.echo(json.dumps(record, indent=2))
    else:
        show_run(record)


@app.command()
def tool_progress(
    run_id: str = typer.Argument(...),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    user_id: str = typer.Option("cli-user", "--user-id"),
    raw: bool = typer.Option(False, "--json", help="Print the bounded metadata journal."),
) -> None:
    """Inspect saved tool progress; never replays tools or continues a session."""
    try:
        progress = AgentClient(base_url).tool_progress(user_id, run_id)
    except Exception as exc:
        typer.echo(f"Tool progress lookup failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if raw:
        typer.echo(json.dumps(progress, indent=2))
        return
    typer.echo(
        f"run={progress['run_id']} coverage={progress['coverage']} "
        f"observed_events={progress['observed_events']}"
    )
    for event in progress["events"]:
        typer.echo(
            f"{event['sequence']}: {event['type']} {event['tool']} "
            f"error_observed={event.get('error_observed')} "
            f"latency_ms={event.get('latency_ms')} at={event['observed_at']}"
        )
    if progress["events_truncated"]:
        typer.echo("Saved event prefix truncated; count includes later observed events.")
    typer.echo(progress["scope_note"])


def show_checkpoint(record: dict[str, Any]) -> None:
    typer.echo(
        f"checkpoint={record['checkpoint_id']} session={record['session_id']} "
        f"files={record['file_count']} bytes={record['byte_count']} "
        f"restore_supported={record.get('restore_supported', 'unknown')}"
    )
    typer.echo(f"root={record['root']} captured_at={record['captured_at']}")
    typer.echo(
        record.get("scope_note", "Checkpoint scope unavailable; do not infer restore support.")
    )


@app.command()
def continue_session(
    session_id: str = typer.Argument(..., help="Source saved session; it stays read-only."),
    message: str = typer.Argument(..., help="New bounded coding request."),
    expected_turn_id: str = typer.Option(..., "--expected-turn-id"),
    request_id: str = typer.Option(..., "--request-id", help="Stable key for this exact request."),
    acknowledge_uncaptured_state: bool = typer.Option(
        False,
        "--acknowledge-uncaptured-state",
        help="Accept excluded files, external writers and unrecorded backend/tool state risks.",
    ),
    provider: str | None = typer.Option(None, "--provider"),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    user_id: str = typer.Option("cli-user", "--user-id"),
    raw: bool = typer.Option(False, "--json"),
) -> None:
    """Explicitly fork saved history and submit one guarded turn; never polls or retries."""
    if not acknowledge_uncaptured_state:
        typer.echo(
            "Explicit --acknowledge-uncaptured-state is required. Captures exclude .env, "
            "ignored paths, directory/ownership metadata and backend/tool state. "
            "Checks do not isolate external writers or prove model context fit.",
            err=True,
        )
        raise typer.Exit(code=2)
    try:
        record = AgentClient(base_url).continue_session(
            user_id, session_id, expected_turn_id, message, request_id, provider=provider
        )
    except Exception as exc:
        typer.echo(
            f"Continuation request failed ({type(exc).__name__}); request_id={request_id}. "
            "Submission outcome may be unknown. Inspect runs with this key before an explicit "
            "retry of the same continuation input. No automatic retry was made.",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    if raw:
        typer.echo(json.dumps(record, indent=2))
    else:
        show_run(record)
        typer.echo(
            "Submitted fork; inspect run status before further work. Source remains read-only."
        )


@app.command()
def checkpoint_create(
    session_id: str = typer.Argument(...),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    user_id: str = typer.Option("cli-user", "--user-id"),
    raw: bool = typer.Option(False, "--json", help="Print saved checkpoint metadata only."),
) -> None:
    """Save current eligible source bytes in server-local SQLite; never restores files."""
    try:
        record = AgentClient(base_url).create_checkpoint(user_id, session_id)
    except Exception as exc:
        typer.echo(
            f"Checkpoint capture request failed ({type(exc).__name__}). "
            "Outcome may be unknown; inspect checkpoint history before explicitly retrying. "
            "No automatic retry was made.",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    if raw:
        typer.echo(json.dumps(record, indent=2))
    else:
        show_checkpoint(record)


@app.command()
def checkpoints(
    session_id: str = typer.Argument(...),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    user_id: str = typer.Option("cli-user", "--user-id"),
    limit: int = typer.Option(50, "--limit", min=1, max=100),
    offset: int = typer.Option(0, "--offset", min=0),
    raw: bool = typer.Option(False, "--json"),
) -> None:
    """Read a page of checkpoint metadata; never submits a capture or coding turn."""
    try:
        records = AgentClient(base_url).checkpoints(user_id, session_id, limit=limit, offset=offset)
    except Exception as exc:
        typer.echo(f"Checkpoint lookup failed ({type(exc).__name__}).", err=True)
        raise typer.Exit(code=1) from exc
    if raw:
        typer.echo(json.dumps(records, indent=2))
    else:
        for record in records:
            show_checkpoint(record)
        if not records:
            typer.echo("No saved checkpoints in this page.")


@app.command()
def checkpoint(
    session_id: str = typer.Argument(...),
    checkpoint_id: str = typer.Argument(...),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    user_id: str = typer.Option("cli-user", "--user-id"),
    raw: bool = typer.Option(False, "--json", help="Print the full metadata manifest, not source."),
) -> None:
    """Read saved checkpoint metadata after restart; no resume or restore permission."""
    try:
        record = AgentClient(base_url).checkpoint(user_id, session_id, checkpoint_id)
    except Exception as exc:
        typer.echo(f"Checkpoint lookup failed ({type(exc).__name__}).", err=True)
        raise typer.Exit(code=1) from exc
    if raw:
        typer.echo(json.dumps(record, indent=2))
    else:
        show_checkpoint(record)


@app.command()
def check_suggestions(
    session_id: str = typer.Argument(...),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    user_id: str = typer.Option("cli-user", "--user-id"),
    raw: bool = typer.Option(False, "--json"),
) -> None:
    """Inspect static project-check hints; never runs or configures a command."""
    try:
        review = AgentClient(base_url).check_suggestions(user_id, session_id)
    except Exception as exc:
        typer.echo(
            f"Check suggestions unavailable ({type(exc).__name__}). No automatic retry.", err=True
        )
        raise typer.Exit(code=1) from exc
    if raw:
        typer.echo(json.dumps(review, indent=2))
        return
    typer.echo(f"check_suggestions={review['status']} (read-only static hints)")
    typer.echo("No command ran and no configuration changed. Inspect setup before execution.")
    for suggestion in review["suggestions"]:
        typer.echo(f"argv={json.dumps(suggestion['argv'])}")
        typer.echo(f"evidence={json.dumps(suggestion['evidence'])}")
    if not review["suggestions"]:
        typer.echo("No suggestions returned within the supported root-manifest scope.")
    for issue in review["issues"]:
        typer.echo(f"issue={json.dumps(issue['path'])}: {json.dumps(issue['code'])}")
    if review.get("reason"):
        typer.echo(json.dumps(review["reason"]))
    typer.echo(json.dumps(review["scope_note"]))


@app.command()
def workspace_review(
    session_id: str = typer.Argument(...),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    user_id: str = typer.Option("cli-user", "--user-id"),
    max_paths: int = typer.Option(100, "--max-paths", min=1, max=1000),
) -> None:
    """Compare the current workspace to its saved final baseline; never resumes work."""
    try:
        review = AgentClient(base_url).workspace_review(user_id, session_id, max_paths=max_paths)
    except Exception as exc:
        typer.echo(f"Workspace review failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"workspace={review['status']} baseline_turn={review.get('turn_id')}")
    for kind in ("added", "modified", "deleted"):
        typer.echo(f"{kind}: {review.get(kind, [])} (total {review.get(kind + '_count', 0)})")
    if review.get("paths_truncated"):
        typer.echo("Path list truncated; counts cover the full comparison.")
    if review.get("reason"):
        typer.echo(review["reason"])
    show_ignored_entries(review.get("ignored_entries"))
    typer.echo(f"Regular-file mode coverage: {review.get('mode_coverage', 'not_recorded')}")
    typer.echo(f"Excluded patterns: {review.get('excluded_patterns', [])}")
    typer.echo(review.get("scope_note", "Read-only review, not permission to resume or revert."))


@app.command()
def transcript_review(
    session_id: str = typer.Argument(...),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    user_id: str = typer.Option("cli-user", "--user-id"),
    max_issues: int = typer.Option(100, "--max-issues", min=1, max=1000),
) -> None:
    """Inspect saved transcript structure without repairing or continuing work."""
    try:
        review = AgentClient(base_url).transcript_review(user_id, session_id, max_issues=max_issues)
    except Exception as exc:
        typer.echo(f"Transcript review failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"transcript={review['status']} messages={review['message_count']} "
        f"issues={review['issue_count']}"
    )
    for issue in review.get("issues", []):
        typer.echo(f"message[{issue['message_index']}]: {issue['code']}")
    if review.get("issues_truncated"):
        typer.echo("Issue list truncated; count covers the full transcript.")
    typer.echo(review["scope_note"])


@app.command()
def configuration_review(
    session_id: str = typer.Argument(...),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    user_id: str = typer.Option("cli-user", "--user-id"),
    provider: str | None = typer.Option(None, "--provider"),
    message: str = typer.Option("", "--message", help="Proposed input for estimation only."),
) -> None:
    """Review saved configuration and estimated context without starting work."""
    try:
        review = AgentClient(base_url).configuration_review(
            user_id, session_id, provider=provider, message=message
        )
    except Exception as exc:
        typer.echo(
            f"Configuration review unavailable ({type(exc).__name__}). No automatic retry.",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    typer.echo(f"review={review['status']} provider={review.get('provider')}")
    show_profile_selection(review.get("current_model_profile"))
    show_repository_instructions(review.get("repository_instructions"))
    if review.get("reason"):
        typer.echo(review["reason"])
    if review.get("configuration"):
        config = review["configuration"]
        typer.echo(f"configuration={config['status']}")
        typer.echo(
            f"changed_fields={config['changed_fields']} unknown_fields={config['unknown_fields']}"
        )
        if config.get("reason"):
            typer.echo(config["reason"])
    if review.get("transcript"):
        typer.echo(f"transcript={review['transcript']['status']}")
    if review.get("context"):
        context = review["context"]
        typer.echo(
            f"context={context['status']} "
            f"estimated_prompt_tokens={context['estimated_prompt_tokens']} "
            f"includes_proposed_message={context['includes_proposed_message']}"
        )
        show_context_limits(context)
        allowance = context.get("request_output_allowance")
        label = (
            json.dumps(allowance)
            if allowance is not None
            else "uncapped_by_harness"
            if context.get("status") in {"estimated_fit", "unbounded_context"}
            else "unavailable"
        )
        typer.echo(f"request_output_allowance={label}")
        if context.get("reason"):
            typer.echo(json.dumps(context["reason"]))
    typer.echo(review["scope_note"])


def show_revert(record: dict[str, Any]) -> None:
    typer.echo(
        f"revert={record['revert_id']} status={record['status']} "
        f"run={record['request']['run_id']} path={record['request']['path']} "
        f"request_id={record['request']['request_id']}"
    )
    if record.get("reason"):
        typer.echo(record["reason"])
    typer.echo(record["scope_note"])


@app.command()
def revert_file(
    run_id: str = typer.Argument(...),
    path: str = typer.Argument(...),
    request_id: str = typer.Option(
        ..., "--request-id", help="Stable key for this exact file action."
    ),
    user_id: str = typer.Option("cli-user", "--user-id"),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    raw: bool = typer.Option(False, "--json"),
) -> None:
    """Explicitly reverse one recorded file change if its after state still matches."""
    try:
        record = AgentClient(base_url).revert_file(user_id, run_id, path, request_id)
    except Exception as exc:
        typer.echo(
            f"Revert request failed ({type(exc).__name__}); outcome may be unknown. "
            f"Inspect revert-status --request-id {request_id} before choosing any further action. "
            "No automatic retry was made.",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    if raw:
        typer.echo(json.dumps(record, indent=2))
    else:
        show_revert(record)
    if record["status"] != "applied":
        raise typer.Exit(code=2 if record["status"] == "pending" else 1)


@app.command()
def revert_status(
    request_id: str = typer.Option(..., "--request-id"),
    user_id: str = typer.Option("cli-user", "--user-id"),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    raw: bool = typer.Option(False, "--json"),
) -> None:
    """Read one keyed file-action journal record; never performs or retries a file operation."""
    try:
        record = AgentClient(base_url).revert_status(user_id, request_id)
    except Exception as exc:
        typer.echo(f"Revert lookup failed ({type(exc).__name__}).", err=True)
        raise typer.Exit(code=1) from exc
    if raw:
        typer.echo(json.dumps(record, indent=2))
    elif record is None:
        typer.echo("No saved revert request found; no file action was submitted by this lookup.")
    else:
        show_revert(record)
