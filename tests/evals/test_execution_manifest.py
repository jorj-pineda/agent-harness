import hashlib
import json

import pytest

from evals.heldout_execution_manifest import load_execution
from evals.heldout_inputs import _save


@pytest.fixture
def selected(tmp_path):
    manifest = {
        "format": "heldout-execution-manifest-v1",
        "runtime_commit": "a" * 40,
        "inference_authorized": False,
        **dict.fromkeys(
            (
                "protocol_sha256",
                "capture_index_sha256",
                "runtime_inventory_sha256",
                "dependency_files_sha256",
                "image_receipt_sha256",
                "binary_sha256",
            ),
            "0" * 64,
        ),
        **{
            k: str(tmp_path / k)
            for k in (
                "staged_directory",
                "capture_directory",
                "output_directory",
                "image_directory",
                "ollama_binary",
            )
        },
        "command_environment": {
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    }
    mp = tmp_path / "manifest.json"
    mh = _save(mp, json.dumps(manifest).encode())
    approval = {
        "format": "heldout-parent-approval-v1",
        "approval_id": "scripted-not-real-approval",
        "manifest_sha256": mh,
        "runtime_commit": "a" * 40,
        "freeze_reviewed_head": "b" * 40,
        "freeze_merge_commit": "c" * 40,
        "exact_head_review_passed": True,
        "ci_passed": True,
        "merged": True,
        "inference_authorized": True,
    }
    ap = tmp_path / "approval.json"
    ah = _save(ap, json.dumps(approval).encode())
    return mp, mh, ap, ah


def load(selected):
    mp, mh, ap, ah = selected
    return load_execution(mp, manifest_sha256=mh, approval_path=ap, approval_sha256=ah)


def test_exact_binding_is_read_only_and_allocates_no_execution_output(selected):
    manifest, approval = load(selected)
    assert approval.manifest_sha256 == selected[1] and not manifest.inference_authorized
    assert not selected[0].parent.joinpath("output_directory").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("manifest_sha256", "1" * 64),
        ("runtime_commit", "b" * 40),
        ("inference_authorized", False),
        ("merged", False),
        ("ci_passed", False),
        ("exact_head_review_passed", False),
        ("extra", "unknown"),
    ],
)
def test_invalid_or_mismatched_parent_receipt_refuses(selected, field, value):
    mp, mh, ap, ah = selected
    data = json.loads(ap.read_bytes())
    data[field] = value
    ap.write_text(json.dumps(data))
    ah = hashlib.sha256(ap.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        load((mp, mh, ap, ah))


@pytest.mark.parametrize("fault", ["missing", "mode", "hash", "relative", "credentials", "flags"])
def test_manifest_preconditions_refuse_without_outputs(selected, fault):
    mp, mh, ap, ah = selected
    if fault == "missing":
        ap.unlink()
    elif fault == "mode":
        ap.chmod(0o644)
    elif fault == "hash":
        ah = "1" * 64
    else:
        data = json.loads(mp.read_bytes())
        if fault == "relative":
            data["output_directory"] = "relative"
        elif fault == "credentials":
            data["command_environment"]["API_KEY"] = "private-sentinel"
        else:
            data["command_environment"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "0"
        mp.write_text(json.dumps(data))
        mh = hashlib.sha256(mp.read_bytes()).hexdigest()
    with pytest.raises((ValueError, OSError)) as caught:
        load((mp, mh, ap, ah))
    assert "private-sentinel" not in str(caught.value)
    assert not mp.parent.joinpath("output_directory").exists()
