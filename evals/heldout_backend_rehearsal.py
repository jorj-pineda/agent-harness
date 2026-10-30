"""Start and settle one owned metadata-only Ollama child; never generate."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import stat
import subprocess
import time
from itertools import islice
from pathlib import Path

import httpx

from evals.heldout_comparison import validate_plan
from evals.heldout_environment import _get, observe
from evals.heldout_guard import _read_regular
from evals.heldout_inputs import _save
from evals.heldout_model_image import MANIFEST, _hash_blob, verify_image
from evals.heldout_workspaces import _owned_location
from evals.projects.duodoro.recovery_lifecycle import OwnedAPIGate

PORT = 11447  # Never manage the user's default 11434 endpoint.
ENDPOINT = f"http://127.0.0.1:{PORT}"


def _free_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", PORT))


def _listener_pids() -> set[int]:
    result = subprocess.run(
        ["/usr/sbin/lsof", "-nP", "-a", f"-iTCP:{PORT}", "-sTCP:LISTEN", "-t"],
        capture_output=True,
        timeout=5,
        check=False,
    )
    if result.returncode not in (0, 1) or len(result.stdout) > 4096:
        raise ValueError("Cannot establish owned listener identity")
    return {int(line) for line in result.stdout.splitlines()}


def _owned_listener(process: subprocess.Popen) -> None:
    if (
        process.poll() is not None
        or os.getpgid(process.pid) != process.pid
        or _listener_pids() != {process.pid}
    ):
        raise ValueError("Listener does not belong exclusively to the owned child")


def _prepared_models(directory: Path) -> tuple[Path, dict]:
    container = _owned_location(directory)
    if container.is_symlink() or {p.name for p in islice(container.iterdir(), 3)} != {
        "models",
        "prepared-image.json",
    }:
        raise ValueError("Prepared model container inventory changed")
    raw, mode = _read_regular(container / "prepared-image.json", 64_000)
    receipt = json.loads(raw)
    models = container / "models"
    verified = verify_image(models, expected_manifest_sha256=validate_plan().model_digest)
    if (
        mode != 0o600
        or receipt.get("format") != "prepared-heldout-model-image-v1"
        or receipt.get("models") != str(models)
        or receipt.get("copy_method") != "macos-clonefile"
        or receipt.get("prepared_image") != verified
        or verified["manifest_mode"] != 0o400
        or any(pin["mode"] != 0o400 for pin in verified["files"].values())
    ):
        raise ValueError("Prepared model receipt or image changed")
    # Refuse unknown files/directories without recursively following an unknown tree.
    names = {str(MANIFEST), *verified["files"]}
    directories = {parent for name in names for parent in Path(name).parents}
    for relative in directories:
        expected = {
            path.name
            for path in {*(Path(name) for name in names), *directories}
            if path != Path(".") and path.parent == relative
        }
        folder = models / relative
        if (
            folder.is_symlink()
            or {p.name for p in islice(folder.iterdir(), len(expected) + 1)} != expected
        ):
            raise ValueError("Prepared model image inventory changed")
    return models, verified


async def _metadata() -> dict:
    observation = await observe(ENDPOINT)
    async with httpx.AsyncClient(
        base_url=ENDPOINT, timeout=5, trust_env=False, follow_redirects=False
    ) as client:
        loaded = json.loads(await _get(client, "/api/ps"))
    if not observation["matches_protocol"] or loaded != {"models": []}:
        raise ValueError("Owned metadata differs from protocol or a model is loaded")
    return {"observation": observation, "loaded_models": []}


class BackendCleanupError(ValueError):
    def __init__(self, evidence: dict):
        self.evidence = evidence
        super().__init__(
            "Forced backend cleanup; rehearsal cannot pass"
            if evidence["forced"]
            else "Backend cleanup observations failed; rehearsal cannot pass"
        )


def _stop(process: subprocess.Popen, gate: OwnedAPIGate) -> dict:
    errors = {}
    forced = False

    def signal_owned(sig: signal.Signals, stage: str) -> bool:
        try:
            if os.getpgid(process.pid) != process.pid:
                raise ValueError("Owned child process group changed; refusing signal")
            os.killpg(process.pid, sig)
        except BaseException as error:
            errors[stage] = type(error).__name__
            return False
        return True

    try:
        active = process.poll() is None
    except BaseException as error:
        errors["poll"] = type(error).__name__
        # A failed poll is not proof of exit. A separate fresh group check must
        # succeed before signaling; otherwise leave the unknown group untouched.
        active = True
    if active:
        signal_owned(signal.SIGTERM, "terminate")
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        forced = True
        # Recheck ownership even if it was known before graceful termination.
        if signal_owned(signal.SIGKILL, "kill"):
            try:
                process.wait(timeout=5)
            except BaseException as error:
                errors["forced_wait"] = type(error).__name__
    except BaseException as error:
        errors["wait"] = type(error).__name__
    # Independently observe settlement even when poll, ownership, signals or wait
    # failed. The gate is read-only: it never supplies authority to signal.
    settled = False
    try:
        gate()
        settled = True
    except BaseException as error:
        errors["settlement_gate"] = type(error).__name__
    evidence = {
        "settled": settled,
        "passed": settled and not errors and not forced,
        "forced": forced,
        "returncode": process.returncode,
        "errors": errors,
    }
    if not evidence["passed"]:
        raise BackendCleanupError(evidence)
    return evidence


def rehearse(image_directory: Path, directory: Path, binary: Path) -> dict:
    models, initial_image = _prepared_models(image_directory)
    destination = _owned_location(directory)
    image_container = models.parent
    if destination.is_relative_to(image_container) or image_container.is_relative_to(destination):
        raise ValueError("Backend directory overlaps prepared image")
    binary = binary.resolve(strict=True)
    binary_stat = binary.stat()
    if not stat.S_ISREG(binary_stat.st_mode) or not 1 <= binary_stat.st_size <= 256_000_000:
        raise ValueError("Invalid Ollama executable")
    binary_pin = _hash_blob(binary, binary_stat.st_size)
    _free_port()  # Sends no request to an occupied endpoint; does not reserve it.
    destination.mkdir(mode=0o700, exist_ok=False)
    for name in ("home", "tmp"):
        (destination / name).mkdir(mode=0o700)
    environment = {
        "HOME": str(destination / "home"),
        "TMPDIR": str(destination / "tmp"),
        "PATH": f"{binary.parent}:/usr/bin:/bin:/usr/sbin:/sbin",
        "OLLAMA_HOST": f"127.0.0.1:{PORT}",
        "OLLAMA_MODELS": str(models),
        "OLLAMA_NO_CLOUD": "1",
        "OLLAMA_NOPRUNE": "1",
        "OLLAMA_NOHISTORY": "1",
        "OLLAMA_KEEP_ALIVE": "0",
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_QUEUE": "1",
        "OLLAMA_CONTEXT_LENGTH": "16384",
    }
    _save(destination / "environment.json", json.dumps(environment, indent=2).encode())
    log_fd = os.open(destination / "server.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(log_fd, "wb") as log:
        process = subprocess.Popen(
            [str(binary), "serve"],
            env=environment,
            cwd=destination,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        # Enter cleanup before any post-spawn recording/registration can fail.
        gate = OwnedAPIGate(process, PORT)
        try:
            OwnedAPIGate.register(process, PORT)
            _save(
                destination / "process.json",
                json.dumps({"pid": process.pid, "pgid": process.pid, "port": PORT}).encode(),
            )
            deadline = time.monotonic() + 30
            readiness_errors = []
            while True:
                if process.poll() is not None:
                    raise ValueError("Owned backend exited before metadata readiness")
                pids = _listener_pids()
                if pids:
                    _owned_listener(process)  # Never send metadata to a foreign listener.
                    try:
                        metadata = asyncio.run(_metadata())
                    except (httpx.ReadTimeout, httpx.ConnectError) as error:
                        # GET-only readiness can precede Ollama GPU discovery. This
                        # is not a generation retry or comparison attempt.
                        readiness_errors.append(type(error).__name__)
                        if len(readiness_errors) > 16:
                            raise TimeoutError(
                                "Owned metadata readiness retry bound reached"
                            ) from error
                        _save(
                            destination / f"readiness-{len(readiness_errors):02d}.json",
                            json.dumps({"error_type": type(error).__name__}).encode(),
                        )
                    else:
                        break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Owned backend listener did not become ready")
                time.sleep(0.1)
            _owned_listener(process)
            _save(destination / "metadata.json", json.dumps(metadata, indent=2).encode())
        except BaseException as error:
            _save(
                destination / "failure.json",
                json.dumps({"status": "failed", "error_type": type(error).__name__}).encode(),
            )
            raise
        finally:
            try:
                cleanup = _stop(process, gate)
            except BaseException as error:
                _save(
                    destination / "cleanup.json",
                    json.dumps(
                        error.evidence
                        if isinstance(error, BackendCleanupError)
                        else {"settled": False, "passed": False, "error_type": type(error).__name__}
                    ).encode(),
                )
                raise
            else:
                _save(destination / "cleanup.json", json.dumps(cleanup).encode())
    _, final_image = _prepared_models(image_directory)
    if initial_image != final_image or binary_pin != _hash_blob(binary, binary_stat.st_size):
        raise ValueError("Prepared image or executable changed during rehearsal")
    result = {
        "format": "heldout-owned-backend-rehearsal-v1",
        "endpoint": ENDPOINT,
        "binary": str(binary),
        "binary_pin": binary_pin,
        "image": initial_image,
        "metadata": metadata,
        "readiness_errors": readiness_errors,
        "cleanup": cleanup,
        "generation_requests": 0,
        "inference_authorized": False,
    }
    _save(destination / "result.json", json.dumps(result, indent=2).encode())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    rehearse(args.image_directory, args.directory, args.binary)
    print("Owned metadata-only child settled; no inference or launch approval.")


if __name__ == "__main__":
    main()
