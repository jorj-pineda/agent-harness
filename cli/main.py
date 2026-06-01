"""Typer CLI entry point for agent-harness.

Commands
--------
serve   Start the FastAPI server (uvicorn wrapper).
chat    Interactive REPL against a running server.
"""

from __future__ import annotations

import sys

import typer

from cli._client import AgentClient

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
def chat(
    base_url: str = typer.Option("http://localhost:8000", "--base-url", help="API base URL."),
    user_id: str = typer.Option("cli-user", "--user-id", help="Stable user identifier."),
    workspace: str | None = typer.Option(
        None, "--workspace", help="Absolute path to repo root (sets workspace_root on session)."
    ),
    provider: str | None = typer.Option(
        None, "--provider", help="Provider override: ollama | anthropic | openai."
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
    typer.echo("Type your message and press Enter.  /quit or Ctrl-C to exit.\n")

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

        try:
            result = client.chat(user_id, session_id, message, provider)
        except Exception as exc:
            typer.echo(f"  error: {exc}", err=True)
            continue

        typer.echo(f"\n{result.answer}\n")
        confidence_label = (
            "low" if result.confidence < 0.5 else "mid" if result.confidence < 0.7 else "high"
        )
        typer.echo(
            f"  conf={result.confidence:.2f}({confidence_label})"
            f"  escalated={result.escalated}"
            f"  provider={result.provider}"
            f"  latency={result.latency_ms:.0f}ms"
        )
        if result.files_touched:
            typer.echo(f"  files_touched={result.files_touched}")
        typer.echo()
