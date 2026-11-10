import shutil

import pytest

import evals.public_feedback_staging as staging
from tests.evals import test_public_feedback_archive as archive_tests

captured = archive_tests.captured
sha = archive_tests.sha


@pytest.fixture(scope="module")
def template(tmp_path_factory, captured):
    path = tmp_path_factory.mktemp("feedback-staging-template") / "capture"
    shutil.copytree(captured, path)
    return path


@pytest.fixture
def source(captured, template):
    # Restore only this test-owned synthetic fixture between isolated cases.
    shutil.rmtree(captured)
    shutil.copytree(template, captured)
    return captured


def test_preserves_complete_capture_and_stages_exact_fresh_paths(source, tmp_path):
    digest = sha(source / "index.json")
    before = {
        p.relative_to(source): (p.read_bytes(), p.stat().st_mode & 0o777)
        for p in source.rglob("*")
        if p.is_file()
    }
    archive = tmp_path / "archive"
    staging.stage_feedback_capture(source, archive, expected_index_sha256=digest)
    index = staging.verify_staged_feedback(source, archive, expected_index_sha256=digest)
    assert len(index.cells) == 8 and index.inference_authorized is False
    for name, (raw, mode) in before.items():
        path = archive / "captured" / name
        assert path.read_bytes() == raw and path.stat().st_mode & 0o777 == mode
    for row in index.cells:
        container = source / f"cell-{row.cell.ordinal:02d}"
        assert str(container / "workspace") == row.workspace_root
        assert not (container / "admitted.json").exists()
        assert (
            (container / "workspace/pagination.py")
            .read_text()
            .strip()
            .endswith("return values[:size]")
        )
    with pytest.raises((ValueError, FileNotFoundError)):
        staging.stage_feedback_capture(source, tmp_path / "retry", expected_index_sha256=digest)
    assert not (tmp_path / "retry").exists()


@pytest.mark.parametrize("defect", ["overlap", "occupied", "drift"])
def test_refusal_before_rename_preserves_original(source, tmp_path, defect):
    digest = sha(source / "index.json")
    target = source / "nested" if defect == "overlap" else tmp_path / "archive"
    if defect == "occupied":
        target.mkdir()
        (target / "sentinel").write_text("preserve\n")
    elif defect == "drift":
        (source / "01-normalized.json").write_text("changed")
    with pytest.raises((ValueError, FileExistsError)):
        staging.stage_feedback_capture(source, target, expected_index_sha256=digest)
    assert (source / "index.json").exists()
    assert not (target / "captured").exists()
    if defect == "occupied":
        assert (target / "sentinel").read_text() == "preserve\n"


def test_prepare_failure_keeps_archive_and_partial_staging(source, tmp_path, monkeypatch):
    digest = sha(source / "index.json")
    prepare = staging.prepare_feedback_cell

    def fail(ordinal, directory):
        if ordinal == 2:
            directory.mkdir(mode=0o700)
            raise ValueError("simulated preparation failure")
        return prepare(ordinal, directory)

    monkeypatch.setattr(staging, "prepare_feedback_cell", fail)
    archive = tmp_path / "archive"
    with pytest.raises(ValueError, match="simulated"):
        staging.stage_feedback_capture(source, archive, expected_index_sha256=digest)
    assert (archive / "captured/index.json").exists()
    assert (source / "cell-01/prepared.json").exists() and (source / "cell-02").exists()
    assert not (source / "cell-03").exists() and not (source / "staged.json").exists()
    with pytest.raises(FileNotFoundError):
        staging.stage_feedback_capture(source, tmp_path / "retry", expected_index_sha256=digest)
    assert not (tmp_path / "retry").exists()


@pytest.mark.parametrize(
    "defect", ["receipt", "private_mode", "admitted", "initial_test", "archive", "inventory"]
)
def test_staged_drift_refuses_without_reset(source, tmp_path, defect):
    digest = sha(source / "index.json")
    archive = tmp_path / "archive"
    staging.stage_feedback_capture(source, archive, expected_index_sha256=digest)
    if defect == "receipt":
        (source / "staged.json").write_text("{}")
    elif defect == "private_mode":
        (source / "staged.json").chmod(0o644)
    elif defect == "admitted":
        (source / "cell-01/admitted.json").touch()
    elif defect == "initial_test":
        (source / "cell-02/workspace/test_visible.py").write_text("changed")
    elif defect == "archive":
        (archive / "captured/01-normalized.json").write_text("changed")
    else:
        (source / "extra").touch()
    with pytest.raises(ValueError):
        staging.verify_staged_feedback(source, archive, expected_index_sha256=digest)
    assert (source / "staged.json").exists() and (archive / "captured/index.json").exists()
