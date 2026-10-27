"""Local HTTP fixture: real application lifecycle/tools, scripted responses only."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sqlite3
import sys
from pathlib import Path

import uvicorn
from fastapi.routing import APIRoute

from api.server import Components, create_app
from harness.config import Settings
from harness.grounding import Grounder
from harness.router import ProviderRouter
from memory import FactStore
from providers.base import ProviderError, ProviderResponse


def main() -> None:
    configuration = json.loads(Path(sys.argv[1]).read_text())
    trace_path = Path(sys.argv[2])
    port = int(sys.argv[3])
    calls = []
    replies = list(configuration["responses"])

    class Scripted:
        name = "ollama"
        model = "gemma4:12b"

        async def chat(self, messages, *, tools=None, temperature=0.0, max_tokens=None):
            calls.append(
                {
                    "messages": [m.model_dump(mode="json") for m in messages],
                    "tools": [t.model_dump(mode="json") for t in tools or []],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            if not replies:
                raise AssertionError("Offline fixture has no remaining responses")
            reply = replies.pop(0)
            if reply == {"hang": True}:
                await asyncio.Event().wait()
            if reply == {"failure": True}:
                raise ProviderError("Scripted provider failure, no inference")
            return ProviderResponse.model_validate(reply)

    settings = Settings(_env_file=None, **configuration["settings"])
    provider = Scripted()

    def components_factory(_):
        return Components(
            providers={"ollama": provider},
            router=ProviderRouter({"ollama": provider}, default="ollama"),
            embedder=None,
            collection=None,
            fact_store=FactStore(settings.memory_db_path),
            grounder=Grounder(escalation_threshold=0.55),
        )

    app = create_app(settings=settings, components_factory=components_factory)

    def fixture_identity():
        return {"pid": os.getpid(), "calls": len(calls)}

    # Precede the application's catch-all static mount, only in this fixture.
    app.router.routes.insert(0, APIRoute("/__preflight", fixture_identity, methods=["GET"]))

    # Uvicorn restores/re-raises captured SIGTERM after graceful lifespan shutdown.
    # Keep that final notification harmless so the owned fixture reports exit zero.
    signal.signal(signal.SIGTERM, lambda *_: None)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    components = app.state.components
    closed = []
    for connection in (components.fact_store._conn, components.session_store._conn):
        try:
            connection.execute("SELECT 1")
        except sqlite3.ProgrammingError as exc:
            closed.append("closed database" in str(exc))
        else:
            closed.append(False)
    trace = {
        "kind": "scripted-contract-local-http",
        "inference_ran": False,
        "calls": calls,
        "closing": components.closing,
        "unfinished_tasks": sum(not task.done() for task in components.tasks),
        "stores_closed": all(closed),
    }
    descriptor = os.open(trace_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(trace, stream)


if __name__ == "__main__":
    main()
