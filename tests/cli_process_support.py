"""Owned loopback API and real CLI processes; only the scripted fixture provider."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def environment():
    return {
        key: os.environ[key]
        for key in ("PATH", "VIRTUAL_ENV", "LANG", "TMPDIR")
        if key in os.environ
    }


def cli(url, arguments, *, input=None):
    return subprocess.run(
        [str(Path(sys.executable).parent / "agent-harness"), *arguments, "--base-url", url],
        input=input,
        text=True,
        capture_output=True,
        timeout=20,
        cwd=ROOT,
        env=environment(),
    )


@contextmanager
def server(directory, stores, responses, *, use_cli_server=False):
    directory.mkdir(mode=0o700)
    stores.mkdir(exist_ok=True)
    settings = {
        "default_provider": "ollama",
        "ollama_model": "gemma4:12b",
        "memory_db_path": str(stores / "facts.sqlite"),
        "session_db_path": str(stores / "sessions.sqlite"),
        "coding_toolset": "whole_file",
        "max_tool_iterations": 4,
    }
    config = directory / "configuration.json"
    config.write_text(
        json.dumps({"settings": settings, "responses": responses, "use_cli_server": use_cli_server})
    )
    config.chmod(0o600)
    trace = directory / "trace.json"
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with (directory / "server.log").open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "tests/fixtures/paired_recovery_server.py"),
                str(config),
                str(trace),
                str(port),
            ],
            cwd=ROOT,
            env=environment(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            with httpx.Client(base_url=url, timeout=2) as client:
                deadline = time.monotonic() + 10
                while True:
                    assert process.poll() is None, "Owned scripted API exited before readiness"
                    try:
                        ready = client.get("/__preflight")
                        if ready.status_code == 200 and ready.json()["pid"] == process.pid:
                            break
                    except httpx.ConnectError:
                        pass
                    assert time.monotonic() < deadline, "Owned scripted API readiness timed out"
                    time.sleep(0.025)
                yield url, client
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                raise AssertionError("Owned scripted API failed graceful shutdown") from None
    assert process.returncode == 0
    saved = json.loads(trace.read_text())
    assert saved["closing"] and saved["stores_closed"] and saved["unfinished_tasks"] == 0
    assert saved["inference_ran"] is False
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError("Owned API process group survived shutdown")
