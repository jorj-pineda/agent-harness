import json
import shutil

import httpx
import pytest

import evals.public_feedback_input_guard as guard
from evals.heldout_input_guard import InputPins, PinnedOllamaProvider
from evals.public_staging import NormalizedInput
from providers.base import ToolSpec
from tests.evals import test_public_feedback_archive as archive_tests

captured = archive_tests.captured
sha = archive_tests.sha


@pytest.mark.parametrize("ordinal", [True, 0, 9, 1.0, "1"])
def test_invalid_ordinal_refuses_before_archive_reads(tmp_path, monkeypatch, ordinal):
    monkeypatch.setattr(
        guard, "verify_feedback_capture", lambda *a, **k: pytest.fail("read archive")
    )
    with pytest.raises(ValueError, match="eight feedback"):
        guard.load_feedback_pins(
            tmp_path / "absent",
            original_directory=tmp_path / "original",
            expected_index_sha256="a" * 64,
            ordinal=ordinal,
        )


def test_all_eight_cells_load_exact_first_input_bytes(captured):
    for ordinal in range(1, 9):
        pins = guard.load_feedback_pins(
            captured,
            original_directory=captured,
            expected_index_sha256=sha(captured / "index.json"),
            ordinal=ordinal,
        )
        assert pins.normalized == (captured / f"{ordinal:02d}-normalized.json").read_bytes()
        assert pins.wire == (captured / f"native-{ordinal:02d}/request-01.request.bin").read_bytes()


@pytest.mark.parametrize("drift", ["wire", "index"])
def test_changed_selection_during_extraction_refuses(captured, tmp_path, monkeypatch, drift):
    source = tmp_path / "capture"
    shutil.copytree(captured, source)
    digest = sha(source / "index.json")
    verify = guard.verify_feedback_capture
    checks = []

    def change(*args, **kwargs):
        result = verify(*args, **kwargs)
        checks.append(True)
        if len(checks) == 1:
            path = source / (
                "native-01/request-01.request.bin" if drift == "wire" else "index.json"
            )
            path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(guard, "verify_feedback_capture", change)
    with pytest.raises(ValueError):
        guard.load_feedback_pins(
            source, original_directory=captured, expected_index_sha256=digest, ordinal=1
        )


@pytest.mark.parametrize("defect", ["none", "normalized", "wire"])
async def test_shared_pin_guard_enforces_feedback_ordinal_eight(captured, defect):
    pins = guard.load_feedback_pins(
        captured,
        original_directory=captured,
        expected_index_sha256=sha(captured / "index.json"),
        ordinal=8,
    )
    normalized = NormalizedInput.model_validate_json(pins.normalized)
    tools = [
        ToolSpec(
            name=t["function"]["name"],
            description=t["function"]["description"],
            parameters_schema=t["function"]["parameters"],
        )
        for t in json.loads(pins.wire)["tools"]
    ]
    if defect == "wire":
        pins = InputPins(pins.normalized, pins.wire + b" ")
    elif defect == "normalized":
        normalized.messages[1].content += "changed"
    delegated = []

    def respond(request):
        delegated.append(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gemma4:12b",
                "done": True,
                "message": {"role": "assistant", "content": "scripted"},
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
        timeout_seconds=180,
        transport=httpx.MockTransport(respond),
    )
    try:
        kwargs = {
            "tools": tools,
            "temperature": normalized.temperature,
            "max_tokens": normalized.max_tokens,
        }
        if defect == "none":
            await provider.chat(normalized.messages, **kwargs)
            assert delegated == [pins.wire]
        else:
            for _ in range(2):
                with pytest.raises(ValueError):
                    await provider.chat(normalized.messages, **kwargs)
            assert not delegated
    finally:
        await provider.aclose()
