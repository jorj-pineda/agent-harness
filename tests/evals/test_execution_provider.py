import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from evals import heldout_execution_provider as factory
from evals.heldout_comparison import validate_plan
from evals.heldout_inputs import _save
from evals.heldout_settings import protocol_settings
from evals.heldout_staging import verify_capture
from providers.base import ChatMessage, ToolSpec
from tests.evals.test_execution_preflight import prepared as prepared


@pytest.fixture
def setup(prepared):
    manifest = prepared
    directory = Path(manifest.output_directory)
    directory.mkdir()
    raw = manifest.model_dump_json().encode()
    _save(directory / "manifest.json", raw)
    capture = Path(manifest.capture_directory)
    index = verify_capture(
        capture,
        expected_index_sha256=manifest.capture_index_sha256,
        original_directory=Path(manifest.staged_directory),
    )
    # Explicit synthetic admission, not a provenance or live-launch test.
    admission = SimpleNamespace(
        manifest=manifest,
        directory=directory,
        approval=SimpleNamespace(manifest_sha256=hashlib.sha256(raw).hexdigest()),
        capture=index,
    )
    body = json.loads((capture / "01-normalized.json").read_bytes())
    messages = [ChatMessage.model_validate(row) for row in body["messages"]]
    wire = json.loads((capture / "01-wire.json").read_bytes())
    kwargs = {
        "tools": [
            ToolSpec(
                name=row["function"]["name"],
                description=row["function"]["description"],
                parameters_schema=row["function"]["parameters"],
            )
            for row in wire["tools"]
        ],
        "temperature": body["temperature"],
        "max_tokens": body["max_tokens"],
    }
    return admission, protocol_settings(validate_plan()), messages, kwargs


@pytest.mark.parametrize(
    "fault", ["none", "ownership", "runtime", "dependency", "settings", "manifest"]
)
async def test_guarded_native_request_and_predelegation_drift(setup, monkeypatch, fault):
    admission, settings, messages, kwargs = setup
    calls = []
    owned = [True]

    def check_owned():
        if not owned[0]:
            raise ValueError("ownership lost")

    def respond(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "model": "gemma4:12b",
                "done": True,
                "message": {"role": "assistant", "content": "scripted"},
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )

    provider = factory.build_cell_provider(
        admission,
        1,
        settings=settings,
        transport=httpx.MockTransport(respond),
        check_owned=check_owned,
    )
    if fault == "ownership":
        owned[0] = False
    elif fault == "runtime":
        monkeypatch.setattr(
            factory, "_runtime_matches", lambda *a: (_ for _ in ()).throw(ValueError("drift"))
        )
    elif fault == "dependency":
        monkeypatch.setattr(
            factory,
            "check_dependency_selection",
            lambda *a: (_ for _ in ()).throw(ValueError("drift")),
        )
    elif fault == "settings":
        settings.max_tool_iterations += 1
    elif fault == "manifest":
        (admission.directory / "manifest.json").write_bytes(b"changed")
    try:
        if fault == "none":
            response = await provider.chat(messages, **kwargs)
            assert response.content == "scripted" and len(calls) == 1
            assert (
                calls[0].content
                == (Path(admission.manifest.capture_directory) / "01-wire.json").read_bytes()
            )
        else:
            with pytest.raises(ValueError):
                await provider.chat(messages, **kwargs)
            with pytest.raises(ValueError):
                await provider.chat(messages, **kwargs)
            assert not calls
    finally:
        await provider.aclose()
