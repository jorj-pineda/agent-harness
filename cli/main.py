"""Typer CLI entry point for agent-harness.

Commands
--------
serve   Start the FastAPI server (uvicorn wrapper).
chat    Interactive REPL against a running server.
"""

from __future__ import annotations

import json
import sys
import uuid

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
    workspace: str | None = typer.Option(
        None, "--workspace", help="Absolute path to repo root (sets workspace_root on session)."
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Provider override: ollama | anthropic | openai | openai_compatible.",
    ),
) -> None:
    """Interactive REPL — connect to a running agent-harness server and chat."""
    client = AgentClient(base_url)
    try:
        session_id = client.create_session(user_id, workspace)
    except Exception as exc:
        typer.echo(f"Error: could not create session — {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Connected  session={session_id}")
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


def show_run(record: dict) -> None:
    typer.echo(f"run_id={record['run_id']} status={record['status']}")
    typer.echo(f"task={record['request'].get('message', '')}")
    typer.echo(
        f"session={record['request']['session_id']} request_id={record['request']['request_id']}"
    )
    typer.echo(
        f"provider={record['provider']} model={record.get('configured_model') or 'unrecorded'}"
    )
    if record.get("error"):
        typer.echo(record["error"])
    if record.get("response") is not None:
        show_turn(summarize(record["response"]))
    else:
        typer.echo("No saved final response. Workspace edits may exist; inspect before new work.")


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
    typer.echo(f"Excluded patterns: {review.get('excluded_patterns', [])}")
    typer.echo(review.get("scope_note", "Read-only review, not permission to resume or revert."))
