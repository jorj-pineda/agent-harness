"""Permission coverage is explicit and legacy records remain scoped."""

from pathlib import Path

from harness.workspace_review import WorkspaceReviewBaseline, compare_workspace, review_fingerprints
from workspace import snapshot


def test_legacy_baseline_does_not_invent_mode_coverage(tmp_path: Path):
    path = tmp_path / "script"
    path.write_text("echo hi\n")
    path.chmod(0o600)
    identity = tmp_path.stat()
    baseline = WorkspaceReviewBaseline.model_validate(
        {
            "status": "recorded",
            "root": str(tmp_path.resolve()),
            "root_identity": [identity.st_dev, identity.st_ino],
            "fingerprints": review_fingerprints(snapshot(tmp_path, ignore=(), strict=True)),
        }
    )
    path.chmod(0o700)
    review = compare_workspace(baseline, max_files=10, max_bytes=100)
    assert review.status == "unchanged"
    assert review.mode_coverage == "not_recorded"
    assert baseline.file_modes is None


def test_modes_and_content_share_modified_path_without_double_counting(tmp_path: Path):
    path = tmp_path / "script"
    path.write_text("before")
    path.chmod(0o600)
    modes = {}
    fingerprints = snapshot(tmp_path, ignore=(), strict=True, file_modes=modes)
    identity = tmp_path.stat()
    baseline = WorkspaceReviewBaseline(
        status="recorded",
        root=str(tmp_path.resolve()),
        root_identity=(identity.st_dev, identity.st_ino),
        fingerprints=review_fingerprints(fingerprints),
        file_modes=modes,
    )
    path.write_text("after")
    path.chmod(0o700)
    review = compare_workspace(baseline, max_files=10, max_bytes=100)
    assert review.modified == ["script"]
    assert review.modified_count == 1
    assert review.mode_coverage == "compared"
    path.unlink()
    review = compare_workspace(baseline, max_files=10, max_bytes=100)
    assert review.deleted == ["script"]
    assert review.modified_count == 0
