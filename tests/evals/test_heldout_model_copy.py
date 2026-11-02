import hashlib
import json
import os
import shutil

import pytest

from evals import heldout_model_copy as copy
from evals.heldout_model_image import MANIFEST


@pytest.fixture
def model(tmp_path, monkeypatch):
    root = tmp_path / "source"
    (root / "blobs").mkdir(parents=True)
    entries = []
    for raw in (b"config", b"weights"):
        digest = hashlib.sha256(raw).hexdigest()
        (root / "blobs" / f"sha256-{digest}").write_bytes(raw)
        entries.append({"digest": f"sha256:{digest}", "size": len(raw)})
    manifest = root / MANIFEST
    manifest.parent.mkdir(parents=True)
    raw = json.dumps({"schemaVersion": 2, "config": entries[0], "layers": entries[1:]}).encode()
    manifest.write_bytes(raw)
    (root / "unrelated-private-file").write_bytes(b"do not copy")
    # Scripted copy primitive: this verifies orchestration, not APFS behavior.
    monkeypatch.setattr(copy, "_clone_file", shutil.copyfile)
    return root, hashlib.sha256(raw).hexdigest(), tmp_path / "prepared"


def prepare(model):
    root, digest, destination = model
    return copy.prepare_image(root, destination, expected_manifest_sha256=digest)


def snapshot(root):
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mode)
        for p in root.rglob("*")
        if p.is_file()
    }


def test_copy_is_independent_private_and_source_unchanged(model):
    root, _, destination = model
    before = snapshot(root)
    receipt = prepare(model)
    assert snapshot(root) == before
    assert not (destination / "models/unrelated-private-file").exists()
    assert (destination.stat().st_mode & 0o777) == 0o700
    assert (destination / "prepared-image.json").stat().st_mode & 0o777 == 0o600
    assert not receipt["service_started"] and not receipt["inference_authorized"]
    for name, (raw, mode) in snapshot(destination / "models").items():
        assert raw == (root / name).read_bytes()
        assert mode & 0o777 == 0o400
        assert (root / name).stat().st_ino != (destination / "models" / name).stat().st_ino


def test_occupied_destination_is_preserved(model):
    _, _, destination = model
    destination.mkdir()
    (destination / "user-work").write_bytes(b"keep")
    before = snapshot(destination)
    with pytest.raises(FileExistsError):
        prepare(model)
    assert snapshot(destination) == before


def test_hardlink_refuses_before_source_chmod(model, monkeypatch):
    root, _, destination = model
    before = snapshot(root)
    monkeypatch.setattr(copy, "_clone_file", os.link)
    with pytest.raises(ValueError, match="shares source inode"):
        prepare(model)
    assert snapshot(root) == before
    assert not (destination / "prepared-image.json").exists()


@pytest.mark.parametrize("fault", ["clone", "corrupt", "source-mode"])
def test_failed_or_changed_copy_retains_partial_without_receipt(model, monkeypatch, fault):
    root, _, destination = model

    def broken(source, target):
        if fault == "clone":
            raise OSError("clone unavailable")
        shutil.copyfile(source, target)
        if fault == "corrupt":
            target.write_bytes(b"bad")
        else:
            # Always change the mode, including fixtures created under umask 077.
            source.chmod(0o640 if source.stat().st_mode & 0o777 == 0o600 else 0o600)

    monkeypatch.setattr(copy, "_clone_file", broken)
    with pytest.raises((ValueError, OSError)):
        prepare(model)
    assert destination.exists() and not (destination / "prepared-image.json").exists()
    with pytest.raises(FileExistsError):
        prepare(model)
    assert root.exists()


@pytest.mark.parametrize("kind", ["overlap", "git", "link"])
def test_unsafe_destination_refuses_without_copy(model, monkeypatch, kind, tmp_path):
    root, digest, destination = model
    if kind == "overlap":
        destination = root / "child"
    elif kind == "git":
        (tmp_path / ".git").write_bytes(b"gitdir: somewhere")
    else:
        destination.symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(copy, "_clone_file", lambda *args: pytest.fail("copy attempted"))
    with pytest.raises((ValueError, FileExistsError)):
        copy.prepare_image(root, destination, expected_manifest_sha256=digest)


def test_bad_source_pin_refuses_before_destination_creation(model):
    root, _, destination = model
    with pytest.raises(ValueError, match="manifest differs"):
        copy.prepare_image(root, destination, expected_manifest_sha256="0" * 64)
    assert not destination.exists()


def test_link_copy_refuses_before_changing_target_mode(model, monkeypatch):
    root, _, destination = model
    before = snapshot(root)
    monkeypatch.setattr(copy, "_clone_file", lambda source, target: target.symlink_to(source))
    with pytest.raises(ValueError, match="not a regular file"):
        prepare(model)
    assert snapshot(root) == before
    assert not (destination / "prepared-image.json").exists()
