"""Only explicitly created owned child processes are inspected/stopped here."""

import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress

import pytest

from evals.projects.duodoro.recovery_lifecycle import OwnedAPIGate


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def child(tmp_path, *, descendant=False, separate=True):
    ready = tmp_path / "ready"
    code = (
        "import signal,sys,time,subprocess\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))\n"
    )
    if descendant:
        code += f'subprocess.Popen([{sys.executable!r},"-c","import time; time.sleep(30)"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n'
    code += f'Path({str(ready)!r}).write_text("ready")\ntime.sleep(30)\n'
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        start_new_session=separate,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 5
    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists()
    return process


def test_active_owned_api_then_clean_exit_then_occupied_port(tmp_path):
    process = child(tmp_path)
    port = unused_port()
    gate = OwnedAPIGate.register(process, port)
    try:
        with pytest.raises(ValueError, match="active"):
            gate()
        process.terminate()
        assert process.wait(timeout=5) == 0
        gate()
        with socket.socket() as other:
            other.bind(("127.0.0.1", port))
            other.listen()
            with pytest.raises(ValueError, match="port"):
                gate()
            assert other.fileno() >= 0  # No unknown listener was closed or signaled.
        gate()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_surviving_owned_group_worker_refuses_even_after_api_exits_zero(tmp_path):
    process = child(tmp_path, descendant=True)
    gate = OwnedAPIGate.register(process, unused_port())
    try:
        process.terminate()
        assert process.wait(timeout=5) == 0
        with pytest.raises(ValueError, match="surviving workers"):
            gate()
    finally:
        # This group was created by this test; never signal a user/service group.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def test_registration_refuses_unowned_group_and_abnormal_exit(tmp_path):
    process = child(tmp_path, separate=False)
    try:
        with pytest.raises(ValueError, match="own process group"):
            OwnedAPIGate.register(process, unused_port())
    finally:
        process.terminate()
        process.wait(timeout=5)
    other = tmp_path / "other"
    other.mkdir()
    process = child(other)
    gate = OwnedAPIGate.register(process, unused_port())
    process.kill()
    process.wait(timeout=5)
    with pytest.raises(ValueError, match="did not exit cleanly"):
        gate()


def test_ownership_set_does_not_reuse_old_settlement_after_new_api_starts(tmp_path):
    from evals.projects.duodoro.recovery_lifecycle import OwnedAPIs

    resources = OwnedAPIs()
    first = child(tmp_path)
    resources.register(first, unused_port())
    first.terminate()
    first.wait(timeout=5)
    resources()
    later = tmp_path / "later"
    later.mkdir()
    second = child(later)
    try:
        resources.register(second, unused_port())
        with pytest.raises(ValueError, match="active"):
            resources()
    finally:
        second.terminate()
        second.wait(timeout=5)
    resources()
