import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from evals import public_execution_admission as admission
from tests.evals import test_public_execution as execution_tests


@pytest.fixture
def artifacts(tmp_path):
    return execution_tests.selection(tmp_path)


@pytest.fixture
def admit(artifacts, monkeypatch):
    monkeypatch.setattr(admission, "preflight", lambda manifest: "scripted-capture")
    mp, mh, ap, ah = artifacts

    def run(verify_parent=lambda manifest, approval: None):
        # This callback is a scripted provenance fixture, never real approval.
        return admission.admit_public_execution(
            mp,
            manifest_sha256=mh,
            approval_path=ap,
            approval_sha256=ah,
            verify_parent=verify_parent,
        )

    return run


def test_exact_artifacts_preserved_privately_and_output_cannot_be_reused(admit, artifacts):
    result = admit()
    assert result.capture == "scripted-capture"
    assert (result.directory / "manifest.json").read_bytes() == artifacts[0].read_bytes()
    assert (result.directory / "parent-approval.json").read_bytes() == artifacts[2].read_bytes()
    record = json.loads((result.directory / "admission.json").read_bytes())
    assert record["status"] == "admitted" and not record["execution_completed"]
    assert not record["inference_performed_by_admission"]
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in result.directory.iterdir())
    with pytest.raises(FileExistsError):
        admit()


def test_parent_refusal_precedes_preflight_and_allocation(admit, artifacts, monkeypatch):
    monkeypatch.setattr(
        admission, "preflight", lambda m: pytest.fail("preflight before provenance")
    )

    def refuse(manifest, approval):
        raise ValueError("Unverified parent provenance")

    with pytest.raises(ValueError, match="provenance"):
        admit(refuse)
    assert not (artifacts[0].parent / "output_directory").exists()


def test_semantic_failure_leaves_no_output(admit, artifacts, monkeypatch):
    def refuse(manifest):
        raise ValueError("semantic drift")

    monkeypatch.setattr(admission, "preflight", refuse)
    with pytest.raises(ValueError, match="semantic"):
        admit()
    assert not (artifacts[0].parent / "output_directory").exists()


def test_callback_cannot_mutate_effective_environment(admit, artifacts):
    def mutate(manifest, approval):
        manifest.command_environment["PATH"] = "/changed"

    result = admit(mutate)
    assert result.manifest.command_environment["PATH"] == "/usr/bin:/bin"


def test_selected_source_drift_after_preflight_refuses_before_allocation(
    admit, artifacts, monkeypatch
):
    def drift(manifest):
        artifacts[2].write_bytes(b"changed")
        return "scripted-capture"

    monkeypatch.setattr(admission, "preflight", drift)
    with pytest.raises(ValueError, match="hash"):
        admit()
    assert not (artifacts[0].parent / "output_directory").exists()


@pytest.mark.parametrize("secondary_failure", [False, True])
def test_partial_preservation_retained_without_receipt_or_retry(
    admit, artifacts, monkeypatch, secondary_failure
):
    original = admission._save

    def fail(path, raw):
        if path.name == "parent-approval.json":
            path.write_bytes(b"partial")
            raise OSError("primary-preservation-failure")
        if path.name == "admission-failure.json" and secondary_failure:
            raise RuntimeError("secondary-recording-failure")
        return original(path, raw)

    monkeypatch.setattr(admission, "_save", fail)
    with pytest.raises(OSError, match="primary-preservation") as caught:
        admit()
    output = artifacts[0].parent / "output_directory"
    assert (output / "manifest.json").read_bytes() == artifacts[0].read_bytes()
    assert (output / "parent-approval.json").read_bytes() == b"partial"
    assert not (output / "admission.json").exists()
    if secondary_failure:
        assert "RuntimeError" in caught.value.__notes__[0]
    else:
        assert json.loads((output / "admission-failure.json").read_bytes())["status"] == "failed"
    with pytest.raises(FileExistsError):
        admit()


def test_racing_admissions_reserve_exactly_one_output(admit, monkeypatch):
    ready = Barrier(2)

    def preflight(manifest):
        ready.wait(timeout=10)
        return "scripted-capture"

    monkeypatch.setattr(admission, "preflight", preflight)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(admit) for _ in range(2)]
        successes, failures = [], []
        for future in futures:
            try:
                successes.append(future.result(timeout=10))
            except FileExistsError as error:
                failures.append(error)
    assert len(successes) == len(failures) == 1
    assert (successes[0].directory / "admission.json").is_file()


def test_selection_alias_refuses_before_callback(admit, artifacts, monkeypatch):
    original = artifacts[0].with_name("original.json")
    artifacts[0].rename(original)
    artifacts[0].symlink_to(original)
    with pytest.raises(OSError):
        admit(lambda *args: pytest.fail("verified a linked selection"))
    assert not (artifacts[0].parent / "output_directory").exists()


@pytest.mark.parametrize("marker", ["file", "directory", "broken-link"])
def test_withdrawn_selection_refuses_before_parent(admit, artifacts, marker):
    path = artifacts[0].parent / "WITHDRAWN.json"
    if marker == "directory":
        path.mkdir()
    elif marker == "broken-link":
        path.symlink_to(path.parent / "absent")
    else:
        path.write_text("do not read marker contents")
    with pytest.raises(ValueError, match="withdrawn"):
        admit(lambda *a: pytest.fail("parent invoked"))
    assert not (path.parent / "output_directory").exists()


@pytest.mark.parametrize("phase", ["parent", "preflight", "preservation"])
def test_mid_admission_withdrawal_never_writes_success(admit, artifacts, monkeypatch, phase):
    marker = artifacts[0].parent / "WITHDRAWN.json"

    def withdraw(*args):
        marker.write_text("withdrawn")
        return "scripted-capture"

    parent = withdraw if phase == "parent" else lambda *a: None
    if phase == "preflight":
        monkeypatch.setattr(admission, "preflight", withdraw)
    elif phase == "preservation":
        original = admission._save

        def save(path, raw):
            result = original(path, raw)
            if path.name == "parent-approval.json":
                withdraw()
            return result

        monkeypatch.setattr(admission, "_save", save)
    with pytest.raises(ValueError, match="withdrawn"):
        admit(parent)
    output = artifacts[0].parent / "output_directory"
    assert not (output / "admission.json").exists()
    assert output.exists() == (phase == "preservation")


@pytest.mark.parametrize(
    "defect", ["source", "approval", "saved-manifest", "saved-approval", "withdrawn"]
)
def test_ongoing_check_refuses_selected_or_preserved_drift(admit, artifacts, defect):
    result = admit()
    admission.check_public_admission(result)
    path = {
        "source": artifacts[0],
        "approval": artifacts[2],
        "saved-manifest": result.directory / "manifest.json",
        "saved-approval": result.directory / "parent-approval.json",
        "withdrawn": artifacts[0].parent / "WITHDRAWN.json",
    }[defect]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        admission.check_public_admission(result)
    assert (result.directory / "admission.json").exists()


selected = execution_tests.selected
prepared = execution_tests.prepared


def test_real_public_preflight_admits_without_consuming_cells(prepared, tmp_path):
    from evals.heldout_inputs import _save

    mp = tmp_path / "artifacts-manifest.json"
    mh = _save(mp, prepared.model_dump_json().encode())
    ap = tmp_path / "artifacts-approval.json"
    approval = {
        "format": "public-pilot-parent-approval-v1",
        "approval_id": "scripted-not-real",
        "manifest_sha256": mh,
        "runtime_commit": prepared.runtime_commit,
        "freeze_reviewed_head": "b" * 40,
        "freeze_merge_commit": "c" * 40,
        "exact_head_review_passed": True,
        "ci_passed": True,
        "merged": True,
        "inference_authorized": True,
    }
    ah = _save(ap, json.dumps(approval).encode())
    result = admission.admit_public_execution(
        mp, manifest_sha256=mh, approval_path=ap, approval_sha256=ah, verify_parent=lambda *a: None
    )
    assert len(result.capture.cells) == 6
    assert not list(tmp_path.glob("capture/cell-*/admitted.json"))
    admission.check_public_admission(result)
