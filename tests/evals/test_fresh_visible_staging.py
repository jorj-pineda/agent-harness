"""Real mocked capture preservation and fresh exact-path staging; no inference."""

import asyncio
import hashlib
import json
import shutil
import sys
from pathlib import Path

import httpx
import pytest

from evals import fresh_visible_archive as archive
from evals import fresh_visible_rehearsal as rehearsal
from evals import fresh_visible_staging as staging
from evals.heldout_dependency_files import record_files
from evals.heldout_runtime_inventory import record_runtime
from providers.ollama import OllamaProvider


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    root = tmp_path_factory.mktemp("fresh-exact-capture")
    record_runtime({"python": Path(sys.executable)}, root / "runtime.json")
    dependency = root / "dependency.py"
    dependency.write_text("value = 1\n")
    record_files([dependency], root / "dependencies.json")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rehearsal, "_harness_revision", lambda: {"commit": "a" * 40, "dirty": False})
        asyncio.run(
            rehearsal.rehearse_fresh_inputs(
                root / "capture",
                command_env={
                    "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                runtime_inventory=root / "runtime.json",
                runtime_sha256=sha(root / "runtime.json"),
                dependency_manifest=root / "dependencies.json",
                dependency_sha256=sha(root / "dependencies.json"),
            )
        )
    template = root / "template"
    shutil.copytree(root / "capture", template)
    return root / "capture", template


@pytest.fixture
def source(captured):
    path, template = captured
    # Restore only this module's test-owned synthetic capture at its recorded paths.
    shutil.rmtree(path)
    shutil.copytree(template, path)
    return path


def verify(source):
    return archive.verify_fresh_capture(source, expected_index_sha256=sha(source / "index.json"))


def rewrite(source, name, value):
    (source / name).write_text(json.dumps(value))
    index = json.loads((source / "index.json").read_bytes())
    index["cells"][0]["files"][name] = sha(source / name)
    (source / "index.json").write_text(json.dumps(index))


def test_read_only_archive_verification_constructs_no_provider_or_transport(source, monkeypatch):
    before = {
        p.relative_to(source): (p.read_bytes(), p.stat().st_mode & 0o777)
        for p in source.rglob("*")
        if p.is_file()
    }

    def forbidden(*args, **kwargs):
        pytest.fail("Read-only verifier constructed inference machinery")

    monkeypatch.setattr(OllamaProvider, "__init__", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "__init__", forbidden)
    result = verify(source)
    assert len(result.cells) == 12 and result.inference_authorized is False
    assert {
        p.relative_to(source): (p.read_bytes(), p.stat().st_mode & 0o777)
        for p in source.rglob("*")
        if p.is_file()
    } == before


def test_whole_capture_preserved_and_all_twelve_initial_paths_staged(source, tmp_path):
    digest = sha(source / "index.json")
    before = {
        p.relative_to(source): (p.read_bytes(), p.stat().st_mode & 0o777)
        for p in source.rglob("*")
        if p.is_file()
    }
    target = tmp_path / "archive"
    staging.stage_fresh_capture(source, target, expected_index_sha256=digest)
    index = staging.verify_staged_fresh(source, target, expected_index_sha256=digest)
    for name, (raw, mode) in before.items():
        p = target / "captured" / name
        assert p.read_bytes() == raw and p.stat().st_mode & 0o777 == mode
    for row in index.cells:
        container = source / f"cell-{row.cell.ordinal:02d}"
        assert str(container / "workspace") == row.workspace_root
        assert not (container / "admitted.json").exists()
        assert {p.name for p in container.iterdir()} == {"workspace", "prepared.json"}
    with pytest.raises((ValueError, FileNotFoundError)):
        staging.stage_fresh_capture(source, tmp_path / "retry", expected_index_sha256=digest)
    assert not (tmp_path / "retry").exists()


@pytest.mark.parametrize("defect", ["overlap", "occupied", "drift"])
def test_refusal_before_rename_preserves_source(source, tmp_path, defect):
    digest = sha(source / "index.json")
    target = source / "nested" if defect == "overlap" else tmp_path / "archive"
    if defect == "occupied":
        target.mkdir()
        (target / "sentinel").write_text("preserve")
    if defect == "drift":
        (source / "01-normalized.json").write_text("{}")
    with pytest.raises((ValueError, FileExistsError)):
        staging.stage_fresh_capture(source, target, expected_index_sha256=digest)
    assert (source / "index.json").exists() and not (target / "captured").exists()


def test_post_rename_failure_preserves_archive_and_partial_fixtures(source, tmp_path, monkeypatch):
    digest = sha(source / "index.json")
    prepare = staging.prepare_fresh_cell

    def fail(ordinal, path):
        if ordinal == 2:
            path.mkdir(mode=0o700)
            raise ValueError("injected preparation failure")
        return prepare(ordinal, path)

    monkeypatch.setattr(staging, "prepare_fresh_cell", fail)
    target = tmp_path / "archive"
    with pytest.raises(ValueError, match="injected"):
        staging.stage_fresh_capture(source, target, expected_index_sha256=digest)
    assert (target / "captured/index.json").exists()
    assert (source / "cell-01/prepared.json").exists() and (source / "cell-02").exists()
    assert not (source / "cell-03").exists() and not (source / "staged.json").exists()


@pytest.mark.parametrize("defect", ["receipt", "mode", "admitted", "test", "archive", "extra"])
def test_staged_drift_never_resets_or_rebinds(source, tmp_path, defect):
    digest = sha(source / "index.json")
    target = tmp_path / "archive"
    staging.stage_fresh_capture(source, target, expected_index_sha256=digest)
    if defect == "receipt":
        (source / "staged.json").write_text("{}")
    elif defect == "mode":
        (source / "staged.json").chmod(0o644)
    elif defect == "admitted":
        (source / "cell-01/admitted.json").touch()
    elif defect == "test":
        (source / "cell-02/workspace/test_visible.py").write_text("# changed")
    elif defect == "archive":
        (target / "captured/01-normalized.json").write_text("{}")
    else:
        (source / "extra").touch()
    with pytest.raises(ValueError):
        staging.verify_staged_fresh(source, target, expected_index_sha256=digest)
    assert (source / "staged.json").exists() and (target / "captured/index.json").exists()


@pytest.mark.parametrize(
    "defect",
    [
        "ordinal",
        "native_model",
        "native_usage",
        "normalized",
        "reference",
        "result_identity",
        "mode",
    ],
)
def test_archive_semantics_refuse_even_with_reselected_outer_hash(source, defect):
    if defect == "ordinal":
        index = json.loads((source / "index.json").read_bytes())
        index["cells"][0]["cell"]["repeat"] = 2
        (source / "index.json").write_text(json.dumps(index))
    elif defect == "mode":
        (source / "01-normalized.json").chmod(0o644)
    elif defect == "reference":
        rewrite(source, "01-final-implementation.py", {"wrong": "reference"})
    else:
        name = {
            "native_model": "native-01/request-01.response.bin",
            "native_usage": "native-01/request-01.response.bin",
            "normalized": "01-normalized.json",
            "result_identity": "01-result.json",
        }[defect]
        value = json.loads((source / name).read_bytes())
        if defect == "native_model":
            value["model"] = "wrong"
        elif defect == "native_usage":
            value["eval_count"] = -1
        elif defect == "normalized":
            value["messages"][1]["content"] = "different request"
        else:
            value["task_id"] = "fv_header_pair"
        rewrite(source, name, value)
    with pytest.raises(ValueError):
        verify(source)


def test_historical_dependency_file_need_not_remain_available(source):
    dependency = source.parent / "dependency.py"
    raw, mode = dependency.read_bytes(), dependency.stat().st_mode & 0o777
    dependency.unlink()  # This module's test-owned synthetic dependency only.
    try:
        assert len(verify(source).cells) == 12
    finally:
        dependency.write_bytes(raw)
        dependency.chmod(mode)
