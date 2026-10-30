"""Read-only bounded verification of the protocol's local Ollama model files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evals.heldout_comparison import validate_plan
from evals.heldout_guard import _read_regular
from evals.heldout_inputs import _save

MAX_IMAGE_BYTES = 16_000_000_000
MANIFEST = Path("manifests/registry.ollama.ai/library/gemma4/12b")


class Blob(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    size: int = Field(ge=1, le=MAX_IMAGE_BYTES)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    schemaVersion: Literal[2]
    config: Blob
    layers: list[Blob] = Field(min_length=1, max_length=32)


def _hash_blob(path: Path, size: int) -> dict:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != size:
            raise ValueError("Model blob size/type changed")
        digest = hashlib.sha256()
        remaining = size
        while remaining:
            chunk = stream.read(min(1_048_576, remaining))
            if not chunk:
                raise ValueError("Model blob ended early")
            digest.update(chunk)
            remaining -= len(chunk)
        if stream.read(1):
            raise ValueError("Model blob grew during verification")
        after = os.fstat(stream.fileno())
    named = path.lstat()

    def identity(s):
        return (s.st_dev, s.st_ino, s.st_size, s.st_mode, s.st_mtime_ns, s.st_ctime_ns)

    if identity(before) != identity(after) or identity(named) != identity(before):
        raise ValueError("Model blob changed during verification")
    return {"sha256": digest.hexdigest(), "size": size, "mode": stat.S_IMODE(before.st_mode)}


def verify_image(root: Path, *, expected_manifest_sha256: str) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Invalid model-image root")
    for relative in (MANIFEST.parent, Path("blobs")):
        current = root
        for component in relative.parts:
            current = current / component
            if current.is_symlink() or not current.is_dir():
                raise ValueError("Model-image path contains link or missing directory")
    raw, manifest_mode = _read_regular(root / MANIFEST, 64_000)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_manifest_sha256:
        raise ValueError("Model manifest differs from reviewed digest")
    manifest = Manifest.model_validate_json(raw)
    entries = [manifest.config, *manifest.layers]
    if sum(entry.size for entry in entries) > MAX_IMAGE_BYTES:
        raise ValueError("Model image exceeds byte bound")
    files = {}
    for entry in entries:
        name = "blobs/" + entry.digest.replace(":", "-")
        observation = _hash_blob(root / name, entry.size)
        if observation["sha256"] != entry.digest.removeprefix("sha256:"):
            raise ValueError("Model blob digest mismatch")
        files[name] = observation
    again, again_mode = _read_regular(root / MANIFEST, 64_000)
    if again != raw or again_mode != manifest_mode:
        raise ValueError("Model manifest changed during verification")
    return {
        "format": "heldout-model-image-v1",
        "manifest_sha256": digest,
        "manifest_mode": manifest_mode,
        "files": files,
        "verified_bytes": sum(entry.size for entry in entries),
        "service_verified": False,
        "ownership_verified": False,
        "inference_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.models.resolve() if not args.models.is_symlink() else args.models
    if args.report.resolve().is_relative_to(root.resolve()):
        raise ValueError("Report overlaps model image")
    if args.report.exists() or args.report.is_symlink():
        raise FileExistsError("Report already occupied")
    result = verify_image(root, expected_manifest_sha256=validate_plan().model_digest)
    _save(args.report, json.dumps(result, indent=2).encode())
    print("Model files verified read-only; service, ownership and launch remain unverified.")


if __name__ == "__main__":
    main()
