import hashlib
import json
import stat

import pytest

from evals import heldout_inputs as capture


async def test_production_adapter_inputs_are_retained_without_forwarding(monkeypatch, tmp_path):
    import httpx

    monkeypatch.setattr(
        httpx.AsyncHTTPTransport,
        "handle_async_request",
        lambda *a: pytest.fail("network transport reached"),
    )
    directory = tmp_path / "inputs"
    result = await capture.capture_inputs(directory)
    assert result["forwarded_requests"] == 0 and result["inference_ran"] is False
    assert result["planned_cells"] == len(result["cells"]) == 16
    assert len(list(directory.iterdir())) == 34
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for row in result["cells"]:
        for name, digest in row["files"].items():
            path = directory / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        normalized = json.loads((directory / f"{row['ordinal']:02d}-normalized.json").read_text())
        wire = json.loads((directory / f"{row['ordinal']:02d}-wire.json").read_text())
        assert normalized["max_tokens"] == wire["options"]["num_predict"] == 2048
        assert wire["options"]["num_ctx"] == 16384 and wire["think"] is False
        assert wire["model"] == "gemma4:12b" and wire["stream"] is False
        assert normalized["messages"][-1]["role"] == "user"
        if row["attempt"] == 2:
            assert (
                row["repeat_bytes_equal"] is False
            )  # Raw workspace-root differences remain visible.
    with pytest.raises(FileExistsError):
        await capture.capture_inputs(directory)
    assert json.loads((directory / "index.json").read_text()) == result


async def test_failed_boundary_capture_preserves_partial_directory_without_completion_index(
    monkeypatch, tmp_path
):
    async def failed(*args, **kwargs):
        raise RuntimeError("before capture")

    monkeypatch.setattr(capture, "run_task", failed)
    directory = tmp_path / "failed"
    with pytest.raises(RuntimeError):
        await capture.capture_inputs(directory)
    assert directory.is_dir() and not (directory / "index.json").exists()


async def test_settings_drift_during_capture_refuses_without_native_boundary_or_final_index(
    monkeypatch, tmp_path
):
    run = capture.run_task

    async def drift(*args, **kwargs):
        kwargs["settings"].max_tracked_bytes = 123
        return await run(*args, **kwargs)

    monkeypatch.setattr(capture, "run_task", drift)
    monkeypatch.setattr(
        capture.CaptureProvider, "chat", lambda *a, **k: pytest.fail("native capture boundary")
    )
    directory = tmp_path / "drift"
    with pytest.raises(ValueError, match="settings differ from capture"):
        await capture.capture_inputs(directory)
    assert (directory / "settings.json").exists()
    assert not (directory / "index.json").exists()
    assert not (directory / "01-wire.json").exists()
