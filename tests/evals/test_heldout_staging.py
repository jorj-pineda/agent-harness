import hashlib
import json

import httpx
import pytest

from evals import heldout_inputs as inputs
from evals import heldout_staging as staging
from evals.heldout_comparison import validate_plan
from evals.heldout_workspaces import verify_prepared
from harness.config import Settings
from providers.ollama import OllamaProvider


@pytest.fixture
async def captured(tmp_path, monkeypatch):
    monkeypatch.setattr(inputs, "_harness_revision", lambda: {"commit": "a" * 40, "dirty": False})
    source = tmp_path / "inputs"
    index = await inputs.capture_inputs(source, prepared=True)
    digest = hashlib.sha256((source / "index.json").read_bytes()).hexdigest()
    return source, index, digest


async def test_stage_preserves_capture_and_reproduces_exact_first_input_at_original_root(
    captured, tmp_path
):
    source, index, digest = captured
    archive = tmp_path / "archive"
    original = {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    staging.stage_capture(source, archive, expected_index_sha256=digest)
    assert original == {
        str(p.relative_to(archive / "captured")): p.read_bytes()
        for p in (archive / "captured").rglob("*")
        if p.is_file()
    }
    for cell in index["cells"]:
        container = source / f"cell-{cell['ordinal']:02d}"
        assert not (container / "admitted.json").exists()
        verify_prepared(container, task_id=cell["task_id"])
    plan = validate_plan()
    task = staging.load_tasks(staging.REGISTRATIONS[0].root / "tasks.yaml")[0]
    wire = []

    def intercepted(request):
        wire.append(request.content)
        raise inputs.CaptureComplete("mock")

    adapter = OllamaProvider(
        host="http://capture.invalid",
        model=plan.model,
        embed_model="unused",
        num_ctx=plan.runtime.ollama_num_ctx,
        think=plan.runtime.ollama_think,
        timeout_seconds=plan.runtime.request_timeout_seconds,
        transport=httpx.MockTransport(intercepted),
    )
    normalized = []
    try:
        result = await inputs.run_task(
            task,
            provider=inputs.CaptureProvider(adapter, normalized),
            settings=Settings(
                _env_file=None,
                _env_prefix="STAGING_TEST_UNUSED_",
                ollama_model=plan.model,
                **plan.runtime.model_dump(),
            ),
            mode="harness",
            prepared_directory=source / "cell-01",
        )
    finally:
        await adapter.aclose()
    assert result.termination == "error:CaptureComplete"
    assert normalized == [original["01-normalized.json"]]
    assert wire == [original["01-wire.json"]]
    assert (source / "staged.json").is_file()


@pytest.mark.parametrize(
    "defect", ["hash", "wire", "workspace", "receipt", "extra", "dirty", "path", "occupied"]
)
async def test_changed_capture_refuses_before_moving_or_creating_archive(
    captured, tmp_path, defect
):
    source, index, digest = captured
    archive = tmp_path / "archive"
    if defect == "hash":
        digest = "0" * 64
    elif defect == "wire":
        (source / "01-wire.json").write_text("{}")
    elif defect == "workspace":
        (source / "cell-01/workspace/week_key.py").write_text("# changed")
    elif defect == "receipt":
        (source / "cell-01/admitted.json").chmod(0o644)
    elif defect == "extra":
        (source / "unknown").write_text("preserve")
    elif defect in ("dirty", "path"):
        if defect == "dirty":
            index["runtime"]["dirty"] = True
        else:
            index["cells"][0]["runtime_config"]["prepared_workspace_root"] = "/other"
        (source / "index.json").write_text(json.dumps(index))
        digest = hashlib.sha256((source / "index.json").read_bytes()).hexdigest()
    else:
        archive.mkdir()
        (archive / "user-work").write_text("preserve")
    with pytest.raises((ValueError, OSError)):
        staging.stage_capture(source, archive, expected_index_sha256=digest)
    assert (source / "index.json").exists()
    if defect == "occupied":
        assert (archive / "user-work").read_text() == "preserve"
    else:
        assert not archive.exists()


async def test_failure_after_archive_keeps_all_evidence_without_rollback(
    captured, tmp_path, monkeypatch
):
    source, _, digest = captured
    archive = tmp_path / "archive"

    def failed(*args, **kwargs):
        raise OSError("injected preparation failure")

    monkeypatch.setattr(staging, "prepare_fixture", failed)
    with pytest.raises(OSError):
        staging.stage_capture(source, archive, expected_index_sha256=digest)
    assert hashlib.sha256((archive / "captured/index.json").read_bytes()).hexdigest() == digest
    assert source.is_dir() and not (source / "staged.json").exists()
    with pytest.raises((ValueError, OSError)):
        staging.stage_capture(source, archive, expected_index_sha256=digest)
