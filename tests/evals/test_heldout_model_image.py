import hashlib
import json

import pytest

from evals import heldout_model_image as image


@pytest.fixture
def model(tmp_path):
    root = tmp_path / "models"
    (root / "blobs").mkdir(parents=True)
    blobs = []
    for raw in (b"config", b"model-data"):
        digest = hashlib.sha256(raw).hexdigest()
        (root / "blobs" / f"sha256-{digest}").write_bytes(raw)
        blobs.append({"digest": "sha256:" + digest, "size": len(raw)})
    manifest = root / image.MANIFEST
    manifest.parent.mkdir(parents=True)
    raw = json.dumps({"schemaVersion": 2, "config": blobs[0], "layers": [blobs[1]]}).encode()
    manifest.write_bytes(raw)
    return root, hashlib.sha256(raw).hexdigest(), blobs


def test_all_declared_blobs_are_hashed_without_modification(model):
    root, digest, _ = model
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = image.verify_image(root, expected_manifest_sha256=digest)
    assert result["verified_bytes"] == len(b"configmodel-data")
    assert len(result["files"]) == 2 and result["manifest_sha256"] == digest
    assert not result["service_verified"] and not result["ownership_verified"]
    assert not result["inference_authorized"]
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "defect",
    ["manifest", "bytes", "size", "missing", "link", "directory-link", "root-link", "fifo"],
)
def test_changed_missing_or_nonregular_image_refuses(model, tmp_path, defect):
    root, digest, blobs = model
    blob = root / "blobs" / blobs[1]["digest"].replace(":", "-")
    if defect == "manifest":
        (root / image.MANIFEST).write_text("{}")
    elif defect == "bytes":
        blob.write_bytes(b"wrong-data")
    elif defect == "size":
        blob.write_bytes(b"x")
    elif defect == "missing":
        blob.unlink()
    elif defect == "link":
        outside = tmp_path / "outside"
        blob.rename(outside)
        blob.symlink_to(outside)
    elif defect == "directory-link":
        outside = tmp_path / "outside"
        (root / "blobs").rename(outside)
        (root / "blobs").symlink_to(outside, target_is_directory=True)
    elif defect == "root-link":
        outside = tmp_path / "outside"
        root.rename(outside)
        root.symlink_to(outside, target_is_directory=True)
    else:
        import os

        blob.unlink()
        os.mkfifo(blob)
    with pytest.raises((ValueError, OSError)):
        image.verify_image(root, expected_manifest_sha256=digest)


def test_wrong_manifest_digest_refuses_before_blob_reads(model, monkeypatch):
    root, _, _ = model
    monkeypatch.setattr(image, "_hash_blob", lambda *args: pytest.fail("blob read"))
    with pytest.raises(ValueError, match="manifest differs"):
        image.verify_image(root, expected_manifest_sha256="0" * 64)
