import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from evals import heldout_execution_admission as admission
from tests.evals.test_execution_manifest import selected as selected


@pytest.fixture
def admit(selected, monkeypatch):
    monkeypatch.setattr(admission, "preflight", lambda manifest: "scripted-capture")
    mp, mh, ap, ah = selected

    def run(verify_parent=lambda manifest, approval: None):
        # This callback is a scripted provenance fixture, never real approval.
        return admission.admit_execution(
            mp,
            manifest_sha256=mh,
            approval_path=ap,
            approval_sha256=ah,
            verify_parent=verify_parent,
        )

    return run


def test_exact_artifacts_preserved_privately_and_output_cannot_be_reused(admit, selected):
    result = admit()
    assert result.capture == "scripted-capture"
    assert (result.directory / "manifest.json").read_bytes() == selected[0].read_bytes()
    assert (result.directory / "parent-approval.json").read_bytes() == selected[2].read_bytes()
    record = json.loads((result.directory / "admission.json").read_bytes())
    assert record["status"] == "admitted" and not record["execution_completed"]
    assert not record["inference_performed_by_admission"]
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in result.directory.iterdir())
    with pytest.raises(FileExistsError):
        admit()


def test_parent_refusal_precedes_preflight_and_allocation(admit, selected, monkeypatch):
    monkeypatch.setattr(
        admission, "preflight", lambda m: pytest.fail("preflight before provenance")
    )

    def refuse(manifest, approval):
        raise ValueError("Unverified parent provenance")

    with pytest.raises(ValueError, match="provenance"):
        admit(refuse)
    assert not (selected[0].parent / "output_directory").exists()


def test_semantic_failure_leaves_no_output(admit, selected, monkeypatch):
    def refuse(manifest):
        raise ValueError("semantic drift")

    monkeypatch.setattr(admission, "preflight", refuse)
    with pytest.raises(ValueError, match="semantic"):
        admit()
    assert not (selected[0].parent / "output_directory").exists()


def test_callback_cannot_mutate_effective_environment(admit, selected):
    def mutate(manifest, approval):
        manifest.command_environment["PATH"] = "/changed"

    result = admit(mutate)
    assert result.manifest.command_environment["PATH"] == "/usr/bin:/bin"


def test_selected_source_drift_after_preflight_refuses_before_allocation(
    admit, selected, monkeypatch
):
    def drift(manifest):
        selected[2].write_bytes(b"changed")
        return "scripted-capture"

    monkeypatch.setattr(admission, "preflight", drift)
    with pytest.raises(ValueError, match="hash"):
        admit()
    assert not (selected[0].parent / "output_directory").exists()


@pytest.mark.parametrize("secondary_failure", [False, True])
def test_partial_preservation_retained_without_receipt_or_retry(
    admit, selected, monkeypatch, secondary_failure
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
    output = selected[0].parent / "output_directory"
    assert (output / "manifest.json").read_bytes() == selected[0].read_bytes()
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


def test_selection_alias_refuses_before_callback(admit, selected, monkeypatch):
    original = selected[0].with_name("original.json")
    selected[0].rename(original)
    selected[0].symlink_to(original)
    with pytest.raises(OSError):
        admit(lambda *args: pytest.fail("verified a linked selection"))
    assert not (selected[0].parent / "output_directory").exists()
