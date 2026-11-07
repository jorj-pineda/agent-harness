import json

import httpx
import pytest

import evals.public_input_guard as guard
from evals.heldout_input_guard import InputPins, PinnedOllamaProvider
from evals.public_staging import NormalizedInput, stage_public_capture
from providers.base import ToolSpec
from tests.evals import test_public_inputs as input_tests
from tests.evals.test_public_staging import captured

selected = input_tests.selected


@pytest.mark.parametrize("ordinal", [True, 0, 7, 1.0, "1"])
def test_invalid_selection_refuses_before_archive_inspection(tmp_path, monkeypatch, ordinal):
    monkeypatch.setattr(guard, "verify_public_capture", lambda *a, **k: pytest.fail("read archive"))
    with pytest.raises(ValueError, match="six public"):
        guard.load_public_pins(
            tmp_path / "absent",
            original_directory=tmp_path / "original",
            expected_index_sha256="a" * 64,
            ordinal=ordinal,
        )


async def test_all_public_cells_select_exact_archived_bytes(tmp_path, selected):
    source, index, digest = await captured(tmp_path, selected)
    archive = tmp_path / "archive"
    stage_public_capture(source, archive, expected_index_sha256=digest)
    for cell in index.cells:
        pins = guard.load_public_pins(
            archive / "captured",
            original_directory=source,
            expected_index_sha256=digest,
            ordinal=cell.ordinal,
        )
        assert (
            pins.normalized
            == (archive / f"captured/{cell.ordinal:02d}-normalized.json").read_bytes()
        )
        assert pins.wire == (archive / f"captured/{cell.ordinal:02d}-wire.json").read_bytes()
    assert not any(source.glob("cell-*/admitted.json"))


@pytest.mark.parametrize("drift", ["wire", "index"])
async def test_midload_drift_refuses_without_rebinding(tmp_path, selected, monkeypatch, drift):
    source, _, digest = await captured(tmp_path, selected)
    original = guard.verify_public_capture
    calls = []

    def verify(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(True)
        if len(calls) == 1:
            path = source / ("01-wire.json" if drift == "wire" else "index.json")
            path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(guard, "verify_public_capture", verify)
    with pytest.raises(ValueError):
        guard.load_public_pins(
            source, original_directory=source, expected_index_sha256=digest, ordinal=1
        )
    assert (source / "cell-01/admitted.json").exists()


@pytest.mark.parametrize("defect", ["none", "normalized", "wire"])
async def test_shared_guard_enforces_public_first_input_before_forwarding(
    tmp_path, selected, defect
):
    source, _, digest = await captured(tmp_path, selected)
    pins = guard.load_public_pins(
        source, original_directory=source, expected_index_sha256=digest, ordinal=1
    )
    normalized = NormalizedInput.model_validate_json(pins.normalized)
    # Normalized JSON sorts schema keys; native bytes preserve registry insertion order.
    # Recreate that order from the native capture, not its canonicalized counterpart.
    wire = json.loads(pins.wire)
    tools = [
        ToolSpec(
            name=t["function"]["name"],
            description=t["function"]["description"],
            parameters_schema=t["function"]["parameters"],
        )
        for t in wire["tools"]
    ]
    if defect == "wire":
        pins = InputPins(pins.normalized, pins.wire + b" ")
    if defect == "normalized":
        normalized.messages[1].content += "changed request"
    delegated = []

    def response(request):
        delegated.append(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gemma4:12b",
                "done": True,
                "message": {"role": "assistant", "content": "scripted only"},
                "prompt_eval_count": 1,
                "eval_count": 1,
            },
        )

    provider = PinnedOllamaProvider(
        pins=pins,
        host="http://127.0.0.1:11447",
        model="gemma4:12b",
        num_ctx=16384,
        think=False,
        timeout_seconds=30,
        transport=httpx.MockTransport(response),
    )
    try:
        if defect == "none":
            result = await provider.chat(
                normalized.messages, tools=tools, max_tokens=normalized.max_tokens
            )
            assert result.content == "scripted only" and delegated == [pins.wire]
        else:
            with pytest.raises(ValueError, match=r"first .*input differs"):
                await provider.chat(
                    normalized.messages, tools=tools, max_tokens=normalized.max_tokens
                )
            with pytest.raises(ValueError, match="closed"):
                await provider.chat(normalized.messages, tools=tools, max_tokens=2048)
            assert not delegated
    finally:
        await provider.aclose()
    assert provider.closed
