"""Prepare a fresh APFS model-file copy without starting Ollama."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import stat
import sys
from pathlib import Path

from evals.heldout_comparison import validate_plan
from evals.heldout_inputs import _save
from evals.heldout_model_image import MANIFEST, verify_image
from evals.heldout_workspaces import _owned_location


def _clone_file(source: Path, destination: Path) -> None:
    # clonefile creates a distinct file and refuses an occupied destination. There
    # is deliberately no hardlink, ordinary-copy or cross-filesystem fallback.
    if sys.platform != "darwin":
        raise OSError("APFS cloning requires macOS")
    library = ctypes.CDLL(None, use_errno=True)
    clone = library.clonefile
    clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    clone.restype = ctypes.c_int
    if clone(os.fsencode(source), os.fsencode(destination), 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def prepare_image(source: Path, directory: Path, *, expected_manifest_sha256: str) -> dict:
    """Retain partial failures; only a final receipt denotes completed preparation."""
    if source.is_symlink():
        raise ValueError("Source model-image root is a link")
    source = source.resolve(strict=True)
    destination = _owned_location(directory)
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination overlap")
    before = verify_image(source, expected_manifest_sha256=expected_manifest_sha256)
    destination.mkdir(mode=0o700, exist_ok=False)
    models = destination / "models"
    models.mkdir(mode=0o700)
    names = [str(MANIFEST), *before["files"]]
    for name in names:
        original, copied = source / name, models / name
        # Do not copy any other models, identity/config files or receipt contents.
        for parent in reversed(copied.relative_to(models).parents):
            if parent != Path("."):
                (models / parent).mkdir(mode=0o700, exist_ok=True)
        _clone_file(original, copied)
        original_stat, copy_stat = original.stat(), copied.lstat()
        if not stat.S_ISREG(copy_stat.st_mode):
            raise ValueError("Model copy is not a regular file")
        if (original_stat.st_dev, original_stat.st_ino) == (copy_stat.st_dev, copy_stat.st_ino):
            raise ValueError("Model copy shares source inode")
        # Check identity before chmod so a defective copy primitive cannot change
        # source permissions through a hardlink.
        copied.chmod(0o400, follow_symlinks=False)
    after = verify_image(source, expected_manifest_sha256=expected_manifest_sha256)
    if before != after:
        raise ValueError("Source model image changed during preparation")
    verified = verify_image(models, expected_manifest_sha256=expected_manifest_sha256)
    if verified["manifest_mode"] != 0o400 or any(
        pin["mode"] != 0o400 for pin in verified["files"].values()
    ):
        raise ValueError("Prepared image modes changed")
    # Both images independently match the reviewed manifest and every blob digest.
    # The receipt is written last, exclusively. No automatic rollback or retry.
    receipt = {
        "format": "prepared-heldout-model-image-v1",
        "source": str(source),
        "models": str(models),
        "copy_method": "macos-clonefile",
        "source_image": before,
        "prepared_image": verified,
        "service_started": False,
        "inference_authorized": False,
    }
    _save(destination / "prepared-image.json", json.dumps(receipt, indent=2).encode())
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-models", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    prepare_image(
        args.source_models,
        args.directory,
        expected_manifest_sha256=validate_plan().model_digest,
    )
    print("Fresh model image prepared; service lifecycle and launch gates remain incomplete.")


if __name__ == "__main__":
    main()
