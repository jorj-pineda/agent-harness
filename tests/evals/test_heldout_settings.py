import hashlib
import json

import pytest
from pydantic import ConfigDict, Field, computed_field

from evals import heldout_settings as pins
from evals.heldout_comparison import validate_plan
from evals.heldout_inputs import _save
from providers.base import ProviderResponse, TokenUsage


def configured():
    return pins.protocol_settings(validate_plan())


def test_full_classified_settings_are_stable_without_credential_values(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-load")
    monkeypatch.setenv("MODEL_PROFILE", "must-not-load")
    monkeypatch.setenv("MAX_TRACKED_BYTES", "1")
    monkeypatch.setenv("HELDOUT_INPUT_CAPTURE_UNUSED_MAX_TRACKED_BYTES", "2")
    settings = configured()
    raw = pins.settings_bytes(settings)
    data = json.loads(raw)
    assert set(data["values"]) == pins.PUBLIC_FIELDS
    assert not (set(data["values"]) & pins.CREDENTIAL_FIELDS)
    assert data["credential_policy"] == "all-unset-excluded"
    assert data["values"]["max_tracked_bytes"] == 256_000_000
    assert data["values"]["ollama_host"] == "http://127.0.0.1:11447"
    assert b"must-not-load" not in raw
    assert pins.settings_bytes(configured()) == raw


@pytest.mark.parametrize("field", sorted(pins.CREDENTIAL_FIELDS))
def test_credentials_refuse_before_serialization_without_echo(monkeypatch, field):
    settings = configured()
    setattr(settings, field, "private-sentinel")
    monkeypatch.setattr(
        pins.ProtocolSettings, "model_dump", lambda *a, **k: pytest.fail("serialized")
    )
    with pytest.raises(ValueError, match="must not contain") as caught:
        pins.settings_bytes(settings)
    assert "private-sentinel" not in str(caught.value)


def test_new_unclassified_field_refuses_before_any_serialization(monkeypatch):
    class ExpandedSettings(pins.ProtocolSettings):
        future_secret: str = Field(default="private-sentinel")

    settings = ExpandedSettings(_env_file=None)
    monkeypatch.setattr(ExpandedSettings, "model_dump", lambda *a, **k: pytest.fail("serialized"))
    with pytest.raises(ValueError, match="classification") as caught:
        pins.settings_bytes(settings)
    assert "private-sentinel" not in str(caught.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ollama_host", "http://user:private-sentinel@localhost"),
        ("model_profile", "unreviewed"),
        ("openai_compatible_base_url", "https://example.invalid"),
    ],
)
def test_endpoint_and_profile_mutation_refuse_without_echo(field, value):
    settings = configured()
    setattr(settings, field, value)
    with pytest.raises(ValueError) as caught:
        pins.settings_bytes(settings)
    assert "private-sentinel" not in str(caught.value)


@pytest.mark.parametrize("fault", ["mode", "hash", "oversize", "link", "schema"])
def test_changed_pin_files_refuse(tmp_path, fault):
    path = tmp_path / "settings.json"
    raw = pins.settings_bytes(configured())
    digest = _save(path, raw)
    assert pins.load_settings_pin(path, expected_sha256=digest) == raw
    if fault == "mode":
        path.chmod(0o644)
    elif fault == "hash":
        path.write_bytes(b"changed")
    elif fault == "oversize":
        path.write_bytes(b"x" * 64_001)
    elif fault == "link":
        other = tmp_path / "other"
        path.rename(other)
        path.symlink_to(other)
    else:
        path.write_bytes(b'{"format":"unknown","values":{}}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises((ValueError, OSError)):
        pins.load_settings_pin(path, expected_sha256=digest)


async def test_nonserialized_setting_drift_latches_before_any_delegate_call():
    class Provider:
        name = "ollama"
        calls = 0
        closed = False

        async def chat(self, *args, **kwargs):
            self.calls += 1
            return ProviderResponse(
                content="scripted",
                model="gemma4:12b",
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
                latency_ms=0,
            )

        async def aclose(self):
            self.closed = True

    settings = configured()
    expected = pins.settings_bytes(settings)
    raw = Provider()
    provider = pins.SettingsPinnedProvider(raw, settings, expected)
    await provider.chat([])
    assert raw.calls == 1
    previous = settings.max_tracked_bytes
    settings.max_tracked_bytes = 123
    with pytest.raises(ValueError, match="differ from capture"):
        await provider.chat([])
    settings.max_tracked_bytes = previous
    with pytest.raises(ValueError, match="latched closed"):
        await provider.chat([])
    assert raw.calls == 1
    await provider.aclose()
    assert raw.closed


@pytest.mark.parametrize("kind", ["computed", "extra"])
def test_unclassified_computed_or_extra_fields_refuse_before_serialization(monkeypatch, kind):
    if kind == "computed":

        class Expanded(pins.ProtocolSettings):
            @computed_field
            @property
            def future_secret(self) -> str:
                return "private-sentinel"

        settings = Expanded(_env_file=None)
    else:

        class Expanded(pins.ProtocolSettings):
            model_config = ConfigDict(extra="allow")

        settings = Expanded(_env_file=None, future_secret="private-sentinel")
    monkeypatch.setattr(Expanded, "model_dump", lambda *a, **k: pytest.fail("serialized"))
    with pytest.raises(ValueError, match="classification") as caught:
        pins.settings_bytes(settings)
    assert "private-sentinel" not in str(caught.value)
