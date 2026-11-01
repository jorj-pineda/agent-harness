import hashlib
import json

import pytest

from evals import heldout_inputs as inputs
from evals import heldout_operator_rehearsal as operator
from evals.heldout_staging import stage_capture
from harness import providers


@pytest.fixture
async def staged(tmp_path, monkeypatch):
    runtime = {"commit": "a" * 40, "dirty": False}
    monkeypatch.setattr(inputs, "_harness_revision", lambda: runtime)
    monkeypatch.setattr(operator, "_harness_revision", lambda: runtime)
    source = tmp_path / "inputs"
    await inputs.capture_inputs(source, prepared=True)
    digest = hashlib.sha256((source / "index.json").read_bytes()).hexdigest()
    archive = tmp_path / "archive"
    stage_capture(source, archive, expected_index_sha256=digest)
    return source, archive / "captured", digest, tmp_path / "operator"


async def run(staged, **kwargs):
    source, capture, digest, output = staged
    return await operator.rehearse(source, capture, output, expected_index_sha256=digest, **kwargs)


async def test_all_sixteen_cells_reproduce_pins_and_complete_via_native_mock(staged, monkeypatch):
    monkeypatch.setattr(
        providers, "build_configured_provider", lambda *a: pytest.fail("live factory")
    )
    monkeypatch.setattr(
        operator.httpx, "AsyncHTTPTransport", lambda *a, **k: pytest.fail("network transport")
    )
    source, capture, _, output = staged
    original = {
        str(p.relative_to(capture)): p.read_bytes() for p in capture.rglob("*") if p.is_file()
    }
    ledger = await run(staged)
    assert ledger.status == "finished" and all(cell.status == "recorded" for cell in ledger.cells)
    report = json.loads((output / "report.json").read_bytes())
    assert report["mode"] == "scripted-real-tools-rehearsal"
    assert len(report["results"]) == 16
    for ordinal, row in enumerate(report["results"], 1):
        assert row["passed"] and row["completion_status"] == "completed"
        assert row["heldout_preservation"]["status"] == "checks_passed"
        assert row["acceptance_identity"]["status"] == "matching"
        assert (output / f"cell-{ordinal:02d}/request-01.json").read_bytes() == (
            capture / f"{ordinal:02d}-wire.json"
        ).read_bytes()
        assert (source / f"cell-{ordinal:02d}/admitted.json").exists()
    assert original == {
        str(p.relative_to(capture)): p.read_bytes() for p in capture.rglob("*") if p.is_file()
    }
    evidence = json.loads((output / "operator.json").read_bytes())
    assert evidence["forwarded_requests"] == 0 and not evidence["inference_ran"]
    assert evidence["usage_kind"] == "scripted" and evidence["status"] == "finished"
    assert not evidence["inference_authorized"]
    with pytest.raises(ValueError):
        await run(staged)


@pytest.mark.parametrize("fault", ["missing_usage", "wrong_model", "transport"])
async def test_native_boundary_fault_stops_before_tools_and_second_admission(
    staged, monkeypatch, fault
):
    made = []
    original = operator.ScriptedTransport

    def transport(*args):
        instance = original(*args)
        made.append(instance)
        return instance

    monkeypatch.setattr(operator, "ScriptedTransport", transport)
    with pytest.raises(operator.ProtocolStop, match=fault):
        await run(staged, fault=fault)
    source, _, _, output = staged
    data = json.loads((output / "report.json").read_bytes())
    assert len(data["results"]) == 1
    assert data["results"][0]["tool_trace"] == []
    assert not data["results"][0]["modified"]
    assert [c["status"] for c in data["cohort"]["cells"]] == ["recorded"] + ["unrun"] * 15
    assert (source / "cell-01/admitted.json").exists()
    assert not (source / "cell-02/admitted.json").exists()
    assert len(made) == 1 and made[0].requests == 1 and made[0].closed and made[0].oracle.closed
    assert json.loads((output / "operator.json").read_bytes())["status"] == "stopped"


@pytest.mark.parametrize("fault", ["receipt", "runtime", "occupied", "unknown"])
async def test_preflight_refuses_before_provider_or_workspace_admission(staged, monkeypatch, fault):
    source, _, _, output = staged
    if fault == "receipt":
        (source / "staged.json").write_text("{}")
    elif fault == "runtime":
        monkeypatch.setattr(
            operator, "_harness_revision", lambda: {"commit": "b" * 40, "dirty": False}
        )
    elif fault == "occupied":
        output.mkdir()
        (output / "user-work").write_text("preserve")
    else:
        (source / "unexpected").write_text("preserve")
    monkeypatch.setattr(operator, "ScriptedTransport", lambda *a: pytest.fail("provider allocated"))
    with pytest.raises((ValueError, FileExistsError)):
        await run(staged)
    assert not (source / "cell-01/admitted.json").exists()
    if fault == "occupied":
        assert (output / "user-work").read_text() == "preserve"
    else:
        assert not output.exists()


async def test_cleanup_error_retains_row_and_stops_rest(staged, monkeypatch):
    close = operator.ScriptedTransport.aclose

    async def failed_close(self):
        await close(self)
        raise OSError("scripted close failure")

    monkeypatch.setattr(operator.ScriptedTransport, "aclose", failed_close)
    with pytest.raises(OSError, match="close failure"):
        await run(staged)
    source, _, _, output = staged
    report = json.loads((output / "report.json").read_bytes())
    assert len(report["results"]) == 1 and report["cohort"]["cleanup_error"] == "OSError"
    assert not (source / "cell-02/admitted.json").exists()
    assert report["cohort"]["cells"][0]["status"] == "recorded"
    assert all(c["status"] == "unrun" for c in report["cohort"]["cells"][1:])


@pytest.mark.parametrize("part", ["normalized", "wire"])
async def test_changed_first_input_never_reaches_mock_oracle(staged, monkeypatch, part):
    from dataclasses import replace

    load = operator.load_pins
    made = []
    original = operator.ScriptedTransport

    def changed(*args, **kwargs):
        return replace(load(*args, **kwargs), **{part: b"changed-input"})

    def transport(*args):
        instance = original(*args)
        made.append(instance)
        return instance

    monkeypatch.setattr(operator, "load_pins", changed)
    monkeypatch.setattr(operator, "ScriptedTransport", transport)
    with pytest.raises(operator.ProtocolStop, match="transport"):
        await run(staged)
    source, _, _, output = staged
    assert len(made) == 1 and made[0].requests == 0 and made[0].closed
    assert made[0].oracle.calls == 0
    assert not (source / "cell-02/admitted.json").exists()
    assert not (output / "cell-01/request-01.json").exists()


async def test_primary_failure_survives_close_failure_with_no_synthesized_row(staged, monkeypatch):
    close = operator.ScriptedTransport.aclose

    async def failed_run(*args, **kwargs):
        raise ValueError("primary runtime failure")

    async def failed_close(self):
        await close(self)
        raise OSError("secondary close failure")

    monkeypatch.setattr(operator, "run_task", failed_run)
    monkeypatch.setattr(operator.ScriptedTransport, "aclose", failed_close)
    with pytest.raises(ValueError, match="primary runtime failure"):
        await run(staged)
    source, _, _, output = staged
    data = json.loads((output / "report.json").read_bytes())
    assert data["results"] == []
    assert data["cohort"]["stop_reason"] == "ValueError"
    assert data["cohort"]["cleanup_error"] == "OSError"
    assert [c["status"] for c in data["cohort"]["cells"]] == ["no_result"] + ["unrun"] * 15
    assert not (source / "cell-01/admitted.json").exists()


async def test_unserialized_setting_drift_refuses_before_allocation_and_admission(
    staged, monkeypatch
):
    source, _, _, output = staged
    build = operator.protocol_settings

    def changed(plan):
        settings = build(plan)
        settings.max_tracked_bytes = 123
        return settings

    monkeypatch.setattr(operator, "protocol_settings", changed)
    monkeypatch.setattr(operator, "ScriptedTransport", lambda *a: pytest.fail("allocated"))
    with pytest.raises(ValueError, match="settings differ from capture"):
        await run(staged)
    assert not output.exists() and not (source / "cell-01/admitted.json").exists()


async def test_between_cell_settings_drift_stops_before_second_admission(staged, monkeypatch):
    from evals.heldout_comparison import validate_plan

    settings = operator.protocol_settings(validate_plan())
    monkeypatch.setattr(operator, "protocol_settings", lambda plan: settings)
    close = operator.ScriptedTransport.aclose

    async def drift(self):
        await close(self)
        settings.max_tracked_bytes = 123

    monkeypatch.setattr(operator.ScriptedTransport, "aclose", drift)
    with pytest.raises(ValueError, match="settings differ from capture"):
        await run(staged)
    source, _, _, output = staged
    data = json.loads((output / "report.json").read_bytes())
    assert len(data["results"]) == 1
    assert data["cohort"]["cells"][0]["status"] == "recorded"
    assert all(c["status"] == "unrun" for c in data["cohort"]["cells"][1:])
    assert not (source / "cell-02/admitted.json").exists()


async def test_synthetic_legacy_capture_stays_verifiable_but_not_admissible(staged, monkeypatch):
    from evals.heldout_staging import verify_capture

    source, capture, _, output = staged
    index_path = capture / "index.json"
    index = json.loads(index_path.read_bytes())
    del index["settings_sha256"]
    (capture / "settings.json").unlink()
    index_path.write_text(json.dumps(index))
    digest = hashlib.sha256(index_path.read_bytes()).hexdigest()
    legacy = verify_capture(capture, expected_index_sha256=digest, original_directory=source)
    assert legacy.settings_sha256 is None
    monkeypatch.setattr(operator, "ScriptedTransport", lambda *a: pytest.fail("allocated"))
    with pytest.raises(ValueError, match="legacy capture is not admissible"):
        await operator.rehearse(source, capture, output, expected_index_sha256=digest)
    assert not output.exists() and not (source / "cell-01/admitted.json").exists()


async def test_final_settings_archive_drift_cannot_finish_rehearsal(staged, monkeypatch):
    _, capture, _, output = staged
    close = operator.ScriptedTransport.aclose
    closed = []

    async def change_last(self):
        await close(self)
        closed.append(True)
        if len(closed) == 16:
            (capture / "settings.json").chmod(0o644)

    monkeypatch.setattr(operator.ScriptedTransport, "aclose", change_last)
    with pytest.raises(ValueError, match="Settings pin bytes/mode changed"):
        await run(staged)
    report = json.loads((output / "report.json").read_bytes())
    assert len(report["results"]) == 16
    assert report["cohort"]["status"] == "stopped"
    assert all(cell["status"] == "recorded" for cell in report["cohort"]["cells"])
    assert json.loads((output / "operator.json").read_bytes())["status"] == "stopped"


@pytest.mark.parametrize("fault", ["missing_model", "not_done"])
async def test_invalid_native_metadata_stops_before_tools_or_next_cell(staged, fault):
    with pytest.raises(operator.ProtocolStop, match="transport"):
        await run(staged, fault=fault)
    source, _, _, output = staged
    data = json.loads((output / "report.json").read_bytes())
    assert len(data["results"]) == 1
    assert not data["results"][0]["modified"] and data["results"][0]["tool_trace"] == []
    assert [cell["status"] for cell in data["cohort"]["cells"]] == ["recorded"] + ["unrun"] * 15
    assert not (source / "cell-02/admitted.json").exists()
    raw = json.loads((output / "cell-01/response-01.json").read_bytes())
    assert ("model" not in raw) if fault == "missing_model" else raw["done"] is False
