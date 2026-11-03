import hashlib

import httpx
import pytest

from evals.heldout_inputs import capture_inputs
from evals.heldout_staging import verify_capture


async def test_residency_capture_uses_fixed_order_and_cannot_masquerade_as_legacy(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport,
        "handle_async_request",
        lambda *a: pytest.fail("network reached"),
    )
    directory = tmp_path / "capture"
    result = await capture_inputs(directory, prepared=True, residency_schedule=True)
    assert result["format"] == "residency-first-inputs-v1"
    assert result["evidence_label"] == "development-on-retired-v1"
    assert result["forwarded_requests"] == 0
    assert [row["keep_alive_seconds"] for row in result["cells"]] == ([0, 300] * 4 + [300, 0] * 4)
    assert all(row["mode"] == "harness" for row in result["cells"])
    assert len(result["cells"]) == 16
    digest = hashlib.sha256((directory / "index.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        verify_capture(directory, expected_index_sha256=digest)
    with pytest.raises(FileExistsError):
        await capture_inputs(directory, prepared=True, residency_schedule=True)


async def test_residency_requires_named_workspaces_before_output(tmp_path):
    directory = tmp_path / "capture"
    with pytest.raises(ValueError, match="named prepared"):
        await capture_inputs(directory, residency_schedule=True)
    assert not directory.exists()
