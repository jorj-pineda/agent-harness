import hashlib
import json

import pytest

import evals.public_staging as staging
from evals.public_inputs import capture_public_inputs
from tests.evals import test_public_inputs as input_tests

selected = input_tests.selected


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_json(path, value):
    path.write_text(json.dumps(value))


async def captured(tmp_path, selected):
    source = tmp_path / "capture"
    index = await capture_public_inputs(source, **selected)
    return source, index, sha(source / "index.json")


async def test_preserves_capture_and_restages_fresh_exact_paths(tmp_path, selected):
    source, index, digest = await captured(tmp_path, selected)
    original = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    archive = tmp_path / "archive"
    staging.stage_public_capture(source, archive, expected_index_sha256=digest)
    assert staging.verify_staged_public(source, archive, expected_index_sha256=digest) == index
    for name, data in original.items():
        assert (archive / "captured" / name).read_bytes() == data
    for cell in index.cells:
        container = source / f"cell-{cell.ordinal:02d}"
        assert str(container / "workspace") == cell.workspace_root
        assert not (container / "admitted.json").exists()
    assert json.loads((source / "staged.json").read_bytes())["inference_authorized"] is False
    with pytest.raises((ValueError, FileNotFoundError)):
        staging.stage_public_capture(source, tmp_path / "retry", expected_index_sha256=digest)
    assert not (tmp_path / "retry").exists()


async def test_archival_verification_does_not_observe_historical_dependencies(tmp_path, selected):
    source, index, digest = await captured(tmp_path, selected)
    (tmp_path / "dependency.py").unlink()
    assert staging.verify_public_capture(source, expected_index_sha256=digest) == index
    archive = tmp_path / "archive"
    source.rename(archive)
    assert (
        staging.verify_public_capture(
            archive, expected_index_sha256=digest, original_directory=source
        )
        == index
    )


@pytest.mark.parametrize(
    "defect", ["hash", "extra", "mode", "fixture", "order", "file-map", "wire", "root", "receipt"]
)
async def test_drift_refuses_before_archive_allocation(tmp_path, selected, defect):
    source, _, digest = await captured(tmp_path, selected)
    index = json.loads((source / "index.json").read_bytes())
    if defect == "hash":
        digest = "0" * 64
    elif defect == "extra":
        (source / "unexpected").write_text("keep")
    elif defect == "mode":
        (source / "01-wire.json").chmod(0o644)
    elif defect == "fixture":
        next(p for p in (source / "cell-01/workspace").rglob("*") if p.is_file()).write_text(
            "changed"
        )
    elif defect == "order":
        index["cells"].reverse()
    elif defect == "file-map":
        index["cells"][0]["files"]["../outside"] = "0" * 64
    else:
        name = {
            "wire": "01-wire.json",
            "root": "01-normalized.json",
            "receipt": "cell-01/admitted.json",
        }[defect]
        path = source / name
        body = json.loads(path.read_bytes())
        if defect == "wire":
            body["options"]["num_ctx"] = 1
        elif defect == "root":
            body["messages"][0]["content"] = "Workspace root: /wrong"
        else:
            body["ordinal"] = 2
        replace_json(path, body)
        index["cells"][0]["files"][name] = sha(path)
    if defect in {"order", "file-map", "wire", "root", "receipt"}:
        replace_json(source / "index.json", index)
        digest = sha(source / "index.json")
    archive = tmp_path / "archive"
    with pytest.raises(ValueError):
        staging.stage_public_capture(source, archive, expected_index_sha256=digest)
    assert source.exists() and not archive.exists()


async def test_prepare_failure_retains_archive_and_partial_staging(tmp_path, selected, monkeypatch):
    source, _, digest = await captured(tmp_path, selected)
    original = staging.prepare_public_cell

    def fail(ordinal, directory):
        if ordinal == 2:
            directory.mkdir(mode=0o700)
            raise RuntimeError("injected preparation failure")
        return original(ordinal, directory)

    monkeypatch.setattr(staging, "prepare_public_cell", fail)
    archive = tmp_path / "archive"
    with pytest.raises(RuntimeError, match="injected"):
        staging.stage_public_capture(source, archive, expected_index_sha256=digest)
    assert (source / "cell-01/prepared.json").exists()
    assert (source / "cell-02").exists() and not (source / "staged.json").exists()
    assert staging.verify_public_capture(
        archive / "captured", expected_index_sha256=digest, original_directory=source
    )


@pytest.mark.parametrize("defect", ["occupied", "overlap"])
async def test_archive_destination_refusals_preserve_capture(tmp_path, selected, defect):
    source, _, digest = await captured(tmp_path, selected)
    archive = source / "nested" if defect == "overlap" else tmp_path / "archive"
    if defect == "occupied":
        archive.mkdir()
        (archive / "keep").write_text("existing")
    with pytest.raises((ValueError, FileExistsError)):
        staging.stage_public_capture(source, archive, expected_index_sha256=digest)
    assert (source / "index.json").exists()


async def test_staged_verifier_refuses_consumed_or_extra_cells(tmp_path, selected):
    source, _, digest = await captured(tmp_path, selected)
    archive = tmp_path / "archive"
    staging.stage_public_capture(source, archive, expected_index_sha256=digest)
    (source / "cell-01/admitted.json").write_text("consumed")
    with pytest.raises(ValueError):
        staging.verify_staged_public(source, archive, expected_index_sha256=digest)


async def test_staged_cell_reproduces_exact_production_first_input(tmp_path, selected):
    import httpx

    from evals.heldout_inputs import CaptureComplete, CaptureProvider
    from evals.heldout_settings import ProtocolSettings, SettingsPin
    from evals.public_admission import PublicCellWorkspace
    from evals.real_run import load_tasks, run_task
    from providers.ollama import OllamaProvider

    source, index, digest = await captured(tmp_path, selected)
    archive = tmp_path / "archive"
    staging.stage_public_capture(source, archive, expected_index_sha256=digest)
    preserved = archive / "captured"
    settings = ProtocolSettings(
        _env_file=None,
        **SettingsPin.model_validate_json((preserved / "settings.json").read_bytes()).values,
    )
    wire = []
    normalized = []

    def intercepted(request):
        wire.append(request.content)
        raise CaptureComplete("mock replay only")

    adapter = OllamaProvider(
        host="http://capture.invalid",
        model=settings.ollama_model,
        embed_model="unused",
        timeout_seconds=settings.request_timeout_seconds,
        num_ctx=settings.ollama_num_ctx,
        think=settings.ollama_think,
        transport=httpx.MockTransport(intercepted),
    )
    try:
        result = await run_task(
            next(task for task in load_tasks() if task.id == index.cells[0].task_id),
            provider=CaptureProvider(adapter, normalized),
            settings=settings,
            mode="harness",
            attempt=1,
            public_cell=PublicCellWorkspace(ordinal=1, directory=source / "cell-01"),
            command_env=selected["command_env"],
        )
    finally:
        await adapter.aclose()
    assert result.termination == "error:CaptureComplete"
    assert wire == [(preserved / "01-wire.json").read_bytes()]
    assert normalized == [(preserved / "01-normalized.json").read_bytes()]
    assert not (source / "cell-02/admitted.json").exists()


@pytest.mark.parametrize("drift", ["archive", "earlier-fixture"])
async def test_final_checks_refuse_drift_during_later_preparation(
    tmp_path, selected, monkeypatch, drift
):
    source, _, digest = await captured(tmp_path, selected)
    archive = tmp_path / "archive"
    original = staging.prepare_public_cell

    def prepare(ordinal, directory):
        result = original(ordinal, directory)
        if ordinal == 6:
            if drift == "archive":
                (archive / "captured/01-wire.json").write_text("changed")
            else:
                fixture = source / "cell-01/workspace"
                next(p for p in fixture.rglob("*") if p.is_file()).write_text("changed")
        return result

    monkeypatch.setattr(staging, "prepare_public_cell", prepare)
    with pytest.raises(ValueError):
        staging.stage_public_capture(source, archive, expected_index_sha256=digest)
    assert not (source / "staged.json").exists() and (archive / "captured/index.json").exists()
