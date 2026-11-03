import hashlib
import json

import pytest

from evals.heldout_inputs import capture_inputs
from evals.heldout_staging import (
    stage_residency_capture,
    verify_capture,
    verify_residency_capture,
)
from evals.heldout_workspaces import verify_prepared


@pytest.fixture
async def captured(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "evals.heldout_inputs._harness_revision", lambda: {"commit": "a" * 40, "dirty": False}
    )
    source = tmp_path / "inputs"
    await capture_inputs(source, prepared=True, residency_schedule=True)
    return source


def digest(source):
    return hashlib.sha256((source / "index.json").read_bytes()).hexdigest()


async def test_stage_preserves_residency_capture_and_fresh_exact_paths(captured, tmp_path):
    source = captured
    selected = digest(source)
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    archive = tmp_path / "archive"
    index = verify_residency_capture(source, expected_index_sha256=selected)
    stage_residency_capture(source, archive, expected_index_sha256=selected)
    assert before == {
        str(p.relative_to(archive / "captured")): p.read_bytes()
        for p in (archive / "captured").rglob("*")
        if p.is_file()
    }
    assert (
        verify_residency_capture(
            archive / "captured", expected_index_sha256=selected, original_directory=source
        )
        == index
    )
    for cell in index.cells:
        container = source / f"cell-{cell.ordinal:02d}"
        verify_prepared(container, task_id=cell.task_id)
        assert not (container / "admitted.json").exists()
    assert (
        json.loads((source / "staged.json").read_text())["format"] == "staged-residency-capture-v1"
    )
    with pytest.raises(ValueError):
        verify_capture(
            archive / "captured", expected_index_sha256=selected, original_directory=source
        )


@pytest.mark.parametrize("fault", ["arm", "schedule", "order", "wire", "extra", "fixture"])
async def test_invalid_residency_bundle_refuses_before_archive(captured, tmp_path, fault):
    source = captured
    doc = json.loads((source / "index.json").read_text())
    if fault == "arm":
        doc["cells"][0]["keep_alive_seconds"] = 300
    elif fault == "schedule":
        doc["residency_plan_sha256"] = "0" * 64
    elif fault == "order":
        doc["cells"].reverse()
    elif fault == "wire":
        (source / "01-wire.json").write_bytes(b"changed")
    elif fault == "extra":
        (source / "unknown").mkdir()
    else:
        (source / "cell-01/workspace/week_key.py").write_text("changed")
    (source / "index.json").write_text(json.dumps(doc))
    archive = tmp_path / "archive"
    with pytest.raises(ValueError):
        stage_residency_capture(source, archive, expected_index_sha256=digest(source))
    assert not archive.exists() and source.exists()
