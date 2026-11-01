import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest

from evals import heldout_backend_rehearsal as backend
from evals import heldout_model_copy as copy
from evals.heldout_comparison import validate_plan
from evals.heldout_model_image import MANIFEST


@pytest.fixture
def owned(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "blobs").mkdir(parents=True)
    raw = b"scripted-model"
    digest = hashlib.sha256(raw).hexdigest()
    (source / "blobs" / f"sha256-{digest}").write_bytes(raw)
    blob = {"digest": f"sha256:{digest}", "size": len(raw)}
    manifest = source / MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schemaVersion": 2, "config": blob, "layers": [blob]}))
    # Keep a synthetic config and model blob; no product model is loaded.
    config = b"scripted-config"
    cdigest = hashlib.sha256(config).hexdigest()
    (source / "blobs" / f"sha256-{cdigest}").write_bytes(config)
    manifest.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "config": {"digest": f"sha256:{cdigest}", "size": len(config)},
                "layers": [blob],
            }
        )
    )
    mdigest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(copy, "_clone_file", shutil.copyfile)
    image = tmp_path / "image"
    copy.prepare_image(source, image, expected_manifest_sha256=mdigest)
    monkeypatch.setattr(backend, "validate_plan", lambda: SimpleNamespace(model_digest=mdigest))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setattr(backend, "PORT", port)
    monkeypatch.setattr(backend, "ENDPOINT", f"http://127.0.0.1:{port}")
    directory = tmp_path / "backend"

    def listener_pids():
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return {json.loads((directory / "process.json").read_bytes())["pid"]}
        except (OSError, FileNotFoundError):
            return set()

    # CI need not install lsof; real child/group/listener lifecycle still executes.
    monkeypatch.setattr(backend, "_listener_pids", listener_pids)
    return image, directory, tmp_path / "fake-ollama"


def executable(path, *, fault="none"):
    plan = validate_plan()
    path.write_text(f"""#!{sys.executable}
import http.server, json, os, signal, sys
from pathlib import Path
assert sys.argv[1:] == ["serve"]
assert os.environ["OLLAMA_NO_CLOUD"] == "1"
assert "UNRELATED_SECRET" not in os.environ
signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        with open("requests.jsonl", "a") as out: out.write(json.dumps([self.command, self.path])+"\\n")
        if self.path == "/api/version": body = {{"version": {plan.expected_ollama_version!r} if {fault!r} != "version" else "wrong"}}
        elif self.path == "/api/tags": body = {{"models": [{{"name": {plan.model!r}, "model": {plan.model!r}, "digest": {plan.model_digest!r}}}]}}
        elif self.path == "/api/ps": body = {{"models": [] if {fault!r} != "loaded" else [{{"name": "unexpected"}}]}}
        else: self.send_error(405); return
        raw = json.dumps(body).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
with http.server.HTTPServer(("127.0.0.1", int(os.environ["OLLAMA_HOST"].split(":")[-1])), Handler) as server:
    server.serve_forever()
""")
    path.chmod(0o700)


def assert_settled(directory):
    pid = json.loads((directory / "process.json").read_bytes())["pid"]
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)
    backend._free_port()


def test_real_owned_child_only_gets_metadata_then_settles(owned, monkeypatch):
    image, directory, binary = owned
    executable(binary)
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-inherit")
    result = backend.rehearse(image, directory, binary)
    assert result["cleanup"]["settled"] and not result["inference_authorized"]
    assert result["generation_requests"] == 0
    assert [
        json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()
    ] == [["GET", "/api/version"], ["GET", "/api/tags"], ["GET", "/api/ps"]]
    env = json.loads((directory / "environment.json").read_bytes())
    assert env["HOME"] == str(directory / "home")
    assert env["OLLAMA_MODELS"] == str(image / "models")
    assert_settled(directory)


@pytest.mark.parametrize("fault", ["version", "loaded"])
def test_bad_metadata_is_retained_without_result_and_child_stops(owned, fault):
    image, directory, binary = owned
    executable(binary, fault=fault)
    with pytest.raises(ValueError, match="metadata differs"):
        backend.rehearse(image, directory, binary)
    assert json.loads((directory / "cleanup.json").read_bytes())["settled"]
    assert not (directory / "result.json").exists()
    assert json.loads((directory / "failure.json").read_bytes())["error_type"] == "ValueError"
    assert_settled(directory)


def test_interrupt_still_settles_owned_child(owned, monkeypatch):
    image, directory, binary = owned
    executable(binary)

    async def interrupted():
        raise KeyboardInterrupt

    monkeypatch.setattr(backend, "_metadata", interrupted)
    with pytest.raises(KeyboardInterrupt):
        backend.rehearse(image, directory, binary)
    assert_settled(directory)
    assert not (directory / "result.json").exists()


def test_foreign_listener_refuses_before_metadata(owned, monkeypatch):
    image, directory, binary = owned
    executable(binary)
    original = backend._listener_pids
    monkeypatch.setattr(backend, "_listener_pids", lambda: {999999} if original() else set())
    with pytest.raises(ValueError, match="Listener does not belong"):
        backend.rehearse(image, directory, binary)
    assert not (directory / "requests.jsonl").exists()
    assert_settled(directory)


@pytest.mark.parametrize("fault", ["occupied", "image-change", "extra-image", "extra-container"])
def test_preconditions_refuse_before_process_creation(owned, monkeypatch, fault):
    image, directory, binary = owned
    executable(binary)
    if fault == "occupied":
        directory.mkdir()
        (directory / "preserve").write_bytes(b"keep")
    elif fault == "image-change":
        (image / "models" / MANIFEST).chmod(0o600)
    elif fault == "extra-image":
        (image / "models" / "unexpected").write_bytes(b"keep")
    else:
        (image / "unexpected").write_bytes(b"keep")
    monkeypatch.setattr(backend.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned"))
    with pytest.raises((ValueError, FileExistsError)):
        backend.rehearse(image, directory, binary)


def test_occupied_port_refuses_without_touching_listener(owned):
    image, directory, binary = owned
    executable(binary)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", backend.PORT))
        listener.listen()
        with pytest.raises(OSError):
            backend.rehearse(image, directory, binary)
        assert listener.fileno() >= 0
        assert not directory.exists()


def test_forced_cleanup_cannot_pass(monkeypatch):
    signals = []
    process = SimpleNamespace(pid=12345, poll=lambda: None, returncode=0)
    attempts = []

    def wait(timeout):
        attempts.append(timeout)
        if len(attempts) == 1:
            raise subprocess.TimeoutExpired("fake", timeout)

    process.wait = wait
    monkeypatch.setattr(backend.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(backend.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(ValueError, match="Forced backend cleanup"):
        backend._stop(process, lambda: None)
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
    assert attempts == [20, 5]


def test_get_only_readiness_timeout_is_retained_before_success(owned, monkeypatch):
    image, directory, binary = owned
    executable(binary)
    original = backend._metadata
    attempts = []

    async def delayed():
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ReadTimeout("scripted readiness")
        return await original()

    monkeypatch.setattr(backend, "_metadata", delayed)
    result = backend.rehearse(image, directory, binary)
    assert result["readiness_errors"] == ["ReadTimeout"]
    assert json.loads((directory / "readiness-01.json").read_bytes()) == {
        "error_type": "ReadTimeout"
    }
    assert_settled(directory)


def test_post_spawn_recording_failure_cleans_up_child(owned, monkeypatch):
    image, directory, binary = owned
    executable(binary)
    save = backend._save
    spawn = backend.subprocess.Popen
    children = []

    def record_spawn(*args, **kwargs):
        process = spawn(*args, **kwargs)
        children.append(process)
        return process

    def fail_process_record(path, raw):
        if path.name == "process.json":
            raise OSError("scripted recording failure")
        save(path, raw)

    monkeypatch.setattr(backend.subprocess, "Popen", record_spawn)
    monkeypatch.setattr(backend, "_save", fail_process_record)
    with pytest.raises((ValueError, OSError)):
        backend.rehearse(image, directory, binary)
    assert len(children) == 1 and children[0].poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(children[0].pid, 0)
    backend._free_port()
    assert (directory / "cleanup.json").exists()
    assert not (directory / "result.json").exists()


@pytest.mark.parametrize("output,code", [(b"123\n", 0), (b"", 1)])
def test_lsof_pid_observation_is_parsed_without_service_requests(monkeypatch, output, code):
    def run(command, **kwargs):
        assert command == [
            "/usr/sbin/lsof",
            "-nP",
            "-a",
            f"-iTCP:{backend.PORT}",
            "-sTCP:LISTEN",
            "-t",
        ]
        assert kwargs["timeout"] == 5
        return SimpleNamespace(stdout=output, returncode=code)

    monkeypatch.setattr(backend.subprocess, "run", run)
    assert backend._listener_pids() == ({123} if output else set())


@pytest.mark.parametrize("output,code", [(b"not-a-pid", 0), (b"", 2), (b"1" * 4097, 0)])
def test_unknown_lsof_evidence_refuses(monkeypatch, output, code):
    monkeypatch.setattr(
        backend.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=output, returncode=code)
    )
    with pytest.raises(ValueError):
        backend._listener_pids()


@pytest.mark.parametrize("fault", ["poll", "group", "wait", "signal"])
def test_cleanup_observation_failures_still_check_and_record_settlement(monkeypatch, fault):
    calls = []
    process = SimpleNamespace(pid=12345, returncode=0)

    def fail():
        raise OSError("scripted cleanup observation failure")

    process.poll = fail if fault == "poll" else lambda: None
    process.wait = (
        (lambda timeout: fail()) if fault == "wait" else lambda timeout: calls.append("wait")
    )
    monkeypatch.setattr(
        backend.os, "getpgid", (lambda pid: fail()) if fault == "group" else lambda pid: pid
    )
    monkeypatch.setattr(
        backend.os,
        "killpg",
        (lambda *args: fail()) if fault == "signal" else lambda pid, sig: calls.append(sig),
    )
    with pytest.raises(backend.BackendCleanupError) as caught:
        backend._stop(process, lambda: calls.append("gate"))
    assert calls[-1] == "gate"
    assert caught.value.evidence["settled"] is True
    assert caught.value.evidence["passed"] is False
    stage = {"poll": "poll", "group": "terminate", "wait": "wait", "signal": "terminate"}[fault]
    assert caught.value.evidence["errors"][stage] == "OSError"
    if fault == "group":
        assert signal.SIGTERM not in calls


def test_unverified_group_and_failed_gate_remain_unknown_without_signal(monkeypatch):
    signals = []
    checked = []
    process = SimpleNamespace(
        pid=12345, returncode=None, poll=lambda: None, wait=lambda timeout: None
    )
    monkeypatch.setattr(backend.os, "getpgid", lambda pid: 99999)
    monkeypatch.setattr(backend.os, "killpg", lambda *args: signals.append(args))

    def gate():
        checked.append(True)
        raise ValueError("unknown settlement")

    with pytest.raises(backend.BackendCleanupError) as caught:
        backend._stop(process, gate)
    assert checked == [True] and signals == []
    assert caught.value.evidence["settled"] is False
    assert caught.value.evidence["errors"] == {
        "terminate": "ValueError",
        "settlement_gate": "ValueError",
    }


def test_force_kill_rechecks_group_identity(monkeypatch):
    signals = []
    groups = iter([12345, 99999])
    process = SimpleNamespace(pid=12345, returncode=None, poll=lambda: None)

    def wait(timeout):
        raise subprocess.TimeoutExpired("fake", timeout)

    process.wait = wait
    monkeypatch.setattr(backend.os, "getpgid", lambda pid: next(groups))
    monkeypatch.setattr(backend.os, "killpg", lambda pid, sig: signals.append(sig))
    with pytest.raises(backend.BackendCleanupError) as caught:
        backend._stop(process, lambda: None)
    assert signals == [signal.SIGTERM]
    assert caught.value.evidence["forced"] is True
    assert caught.value.evidence["errors"]["kill"] == "ValueError"


def test_cleanup_error_receipt_keeps_settlement_separate_from_pass(owned, monkeypatch):
    image, directory, binary = owned
    executable(binary)
    stop = backend._stop

    def observed_error(process, gate):
        evidence = stop(process, gate)
        evidence["passed"] = False
        evidence["errors"] = {"poll": "OSError"}
        raise backend.BackendCleanupError(evidence)

    monkeypatch.setattr(backend, "_stop", observed_error)
    with pytest.raises(backend.BackendCleanupError):
        backend.rehearse(image, directory, binary)
    cleanup = json.loads((directory / "cleanup.json").read_bytes())
    assert cleanup["settled"] is True and cleanup["passed"] is False
    assert cleanup["errors"] == {"poll": "OSError"}
    assert not (directory / "result.json").exists()
    assert_settled(directory)


def test_session_checks_ownership_only_while_active_and_settles(owned):
    image, directory, binary = owned
    executable(binary)
    with backend.owned_backend(image, directory, binary) as session:
        assert session.active and not session.evidence["inference_authorized"]
        session.check_owned()
        assert not (directory / "session.json").exists()
    assert not session.active and session.evidence["cleanup"]["passed"]
    with pytest.raises(ValueError, match="not active"):
        session.check_owned()
    assert_settled(directory)
    data = json.loads((directory / "session.json").read_bytes())
    assert "generation_requests" not in data
    assert not (directory / "result.json").exists()


def test_session_body_failure_retains_failure_and_settles_without_success_receipt(owned):
    image, directory, binary = owned
    executable(binary)
    with (
        pytest.raises(RuntimeError, match="caller failure"),
        backend.owned_backend(image, directory, binary) as session,
    ):
        session.check_owned()
        raise RuntimeError("caller failure")
    assert not session.active
    assert_settled(directory)
    assert json.loads((directory / "failure.json").read_bytes())["error_type"] == "RuntimeError"
    assert json.loads((directory / "cleanup.json").read_bytes())["passed"]
    assert not (directory / "session.json").exists()


def test_session_rejects_foreign_ownership_during_body_and_still_stops_child(owned, monkeypatch):
    image, directory, binary = owned
    executable(binary)
    with (
        pytest.raises(ValueError, match="Listener does not belong"),
        backend.owned_backend(image, directory, binary) as session,
    ):
        monkeypatch.setattr(backend, "_listener_pids", lambda: {999999})
        session.check_owned()
    assert not session.active
    assert_settled(directory)
    assert not (directory / "session.json").exists()


async def test_async_entry_refuses_before_image_access_or_process_creation(owned, monkeypatch):
    image, directory, binary = owned
    monkeypatch.setattr(backend, "_prepared_models", lambda *args: pytest.fail("image accessed"))
    with (
        pytest.raises(ValueError, match="outside an active event loop"),
        backend.owned_backend(image, directory, binary),
    ):
        pytest.fail("entered")
    assert not directory.exists()
