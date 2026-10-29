"""Reviewed held-out data pins, not a model/runtime launch authorization.

Reads are bounded and fail closed. Observations are non-atomic; these checks do
not isolate host code or prevent external writers after a check.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field


class HeldoutDriftError(ValueError):
    """A held-out evaluation must stop without repairing or rebinding its data."""


class FilePin(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: int = Field(ge=0, le=0o7777)
    size: int = Field(ge=0, le=1_000_000)


class CatalogSeal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    format: str = Field(pattern=r"^heldout-tree-v1$")
    catalog_id: str = Field(min_length=1, max_length=64)
    files: dict[str, FilePin] = Field(min_length=1, max_length=2_000)


@dataclass(frozen=True)
class HeldoutRegistration:
    root: Path
    seal_sha256: str


REGISTRATIONS = (
    HeldoutRegistration(
        Path(__file__).parent / "heldout_tasks/v1",
        "dfda71293f2ba2569514fd3049c849e5c647e2725e937d85be174a760e4b9eee",
    ),
)


def registration_for(path: Path) -> HeldoutRegistration | None:
    for registration in REGISTRATIONS:
        if path.absolute().is_relative_to(
            registration.root.absolute()
        ) or path.resolve().is_relative_to(registration.root.resolve()):
            return registration
    return None


def _read_regular(path: Path, limit: int) -> tuple[bytes, int]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise HeldoutDriftError("Unbounded or nonregular held-out file")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    named = path.lstat()
    if (
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_mode,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_mode,
        )
        or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
        or len(raw) != before.st_size
        or len(raw) > limit
    ):
        raise HeldoutDriftError("Held-out file changed during reading")
    return raw, stat.S_IMODE(before.st_mode)


def verify_tree(
    root: Path, files: dict[str, FilePin], *, extra: frozenset[str] = frozenset()
) -> None:
    """Compare metadata inventory before reading any expected task contents."""
    directories = {"."}
    for name in files:
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or str(path) != name
            or ".." in path.parts
            or not path.parts
            or "\\" in name
            or any(part == ".env" or part.startswith(".env.") for part in path.parts)
        ):
            raise HeldoutDriftError("Invalid held-out pin path")
        directories.update(str(parent) for parent in path.parents)
    if (
        root.is_symlink()
        or not root.is_dir()
        or sum(pin.size for pin in files.values()) > 16_000_000
    ):
        raise HeldoutDriftError("Invalid held-out tree")
    actual_files = set()
    actual_dirs = {"."}

    def failed(error: OSError) -> None:
        raise error

    for parent, dirs, names in os.walk(root, followlinks=False, onerror=failed):
        for name in (*dirs, *names):
            path = Path(parent) / name
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                actual_dirs.add(relative)
            elif stat.S_ISREG(mode):
                actual_files.add(relative)
            else:
                raise HeldoutDriftError("Held-out tree contains a link or special file")
            if len(actual_files) + len(actual_dirs) > 2_000:
                raise HeldoutDriftError("Held-out inventory exceeds its bound")
    if actual_files != set(files) | extra or actual_dirs != directories:
        raise HeldoutDriftError("Held-out inventory changed")
    for name, pin in files.items():
        raw, mode = _read_regular(root / name, pin.size)
        if (
            mode != pin.mode
            or len(raw) != pin.size
            or hashlib.sha256(raw).hexdigest() != pin.sha256
        ):
            raise HeldoutDriftError("Held-out bytes or permissions changed")


def verify_catalog(registration: HeldoutRegistration) -> CatalogSeal:
    try:
        if registration.root.is_symlink() or not registration.root.is_dir():
            raise HeldoutDriftError("Invalid held-out root")
        raw, mode = _read_regular(registration.root / "seal.json", 64_000)
        if mode != 0o644 or hashlib.sha256(raw).hexdigest() != registration.seal_sha256:
            raise HeldoutDriftError("Held-out seal changed")
        seal = CatalogSeal.model_validate_json(raw)
        if "tasks.yaml" not in seal.files or "seal.json" in seal.files:
            raise HeldoutDriftError("Invalid held-out seal coverage")
        verify_tree(registration.root, seal.files, extra=frozenset({"seal.json"}))
        return seal
    except (OSError, ValueError, RuntimeError) as error:
        raise HeldoutDriftError("Held-out catalog preservation failed") from error


def verify_fixture(source: Path, copy_root: Path) -> None:
    try:
        registration = registration_for(source)
        if registration is None:
            raise HeldoutDriftError("Unknown held-out fixture")
        seal = verify_catalog(registration)
        prefix = f"fixtures/{source.name}/"
        files = {
            name.removeprefix(prefix): pin
            for name, pin in seal.files.items()
            if name.startswith(prefix)
        }
        if not files:
            raise HeldoutDriftError("Unknown held-out fixture")
        verify_tree(copy_root, files)
    except (OSError, ValueError, RuntimeError) as error:
        raise HeldoutDriftError("Held-out fixture copy preservation failed") from error
