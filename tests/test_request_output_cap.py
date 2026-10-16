import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from harness.config import Settings
from harness.configuration_review import compare_profiles, review_configuration
from harness.context import TokenBudget
from harness.execution_profile import ExecutionProfile, capture_execution_profile
from harness.runtime import build_registry, run_configured_turn
from harness.session_store import SessionArchive
from harness.state import Session
from harness.trace import BudgetStopTrace, RequestTrace
from memory import FactStore
from providers.base import ChatMessage, TokenUsage, ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tests.api.test_session_persistence import make_app
from tools import ToolRegistry


def test_request_cap_resets_but_turn_budget_decreases():
    messages = [ChatMessage(role="user", content="inspect")]
    budget = TokenBudget(max_output=15, max_request_output=6)
    assert budget.request_allowance(messages, []) == 6
    for _ in range(2):
        budget.record(TokenUsage(prompt_tokens=2, completion_tokens=6), 1)
        assert budget.after_response(wants_tools=True) is None
    assert budget.request_allowance(messages, []) == 3
    assert budget.completion_used == 12
    assert TokenBudget().request_allowance(messages, []) is None


@pytest.mark.parametrize(
    "limits,expected",
    [({}, 10), ({"max_output": 8}, 8), ({"max_context": 20}, 6), ({"max_total": 18}, 4)],
)
def test_request_cap_cannot_expand_other_allowances(limits, expected):
    messages = [ChatMessage(role="user", content="x" * 30)]
    budget = TokenBudget(max_request_output=10, **limits)
    assert budget.request_allowance(messages, []) == expected


@pytest.mark.parametrize("output,status", [(None, "incomplete"), (11, "budget_exhausted")])
async def test_unchecked_or_exceeded_request_cap_blocks_file_tools(tmp_path, output, status):
    provider = ScriptedProvider()
    provider.script(
        make_response(
            prompt_tokens=100,
            completion_tokens=output,
            tool_calls=[
                ToolCall(
                    id="write", name="write_file", arguments={"path": "new.py", "content": "unsafe"}
                )
            ],
        )
    )
    trace = []
    settings = Settings(_env_file=None, max_request_output_tokens=10)
    with FactStore(tmp_path / "facts.db") as store:
        result = await run_configured_turn(
            settings=settings,
            session=Session(user_id="dev", workspace_root=str(tmp_path)),
            user_id="dev",
            message="Add a module",
            provider=provider,
            fact_store=store,
            registry=build_registry(fact_store=store, user_id="dev", workspace_root=str(tmp_path)),
            trace=trace,
        )
    assert result.completion_status == status
    assert not (tmp_path / "new.py").exists()
    assert len(provider.calls) == 1
    request = next(item for item in trace if isinstance(item, RequestTrace))
    assert request.max_tokens == 10
    stop = next(item for item in trace if isinstance(item, BudgetStopTrace))
    assert stop.stage == "post_response" and stop.max_request_output_tokens == 10
    assert result.token_usage.completion_tokens == output


async def test_review_runtime_and_recovery_share_request_cap(tmp_path):
    class ObservedProvider(ScriptedProvider):
        def __init__(self):
            super().__init__()
            self.allowances = []

        async def chat(self, messages, **kwargs):
            self.allowances.append(kwargs.get("max_tokens"))
            return await super().chat(messages, **kwargs)

    settings = Settings(_env_file=None, max_request_output_tokens=10, max_completion_retries=1)
    provider = ObservedProvider()
    provider.script(
        make_response(content="", prompt_tokens=10, completion_tokens=10),
        make_response(content="Done", prompt_tokens=12, completion_tokens=10),
    )
    session = Session(user_id="dev")
    with FactStore(tmp_path / "facts.db") as store:
        review = review_configuration(
            SessionArchive(session=session),
            settings=settings,
            provider="scripted",
            registry=ToolRegistry(),
            fact_store=store,
            user_id="dev",
            message="Inspect",
        )
        result = await run_configured_turn(
            settings=settings,
            session=session,
            user_id="dev",
            message="Inspect",
            provider=provider,
            fact_store=store,
            registry=ToolRegistry(),
        )
    assert review.context.configured_request_output_cap == 10
    assert review.context.request_output_allowance == 10
    assert provider.allowances == [10, 10]
    assert result.completion_status == "completed"
    assert result.token_usage.completion_tokens == 20


def test_request_cap_settings_env_and_legacy_configuration(monkeypatch):
    monkeypatch.setenv("MAX_REQUEST_OUTPUT_TOKENS", "512")
    assert Settings(_env_file=None).max_request_output_tokens == 512
    for cap in (-1, 1_000_001):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, max_request_output_tokens=cap)
    monkeypatch.delenv("MAX_REQUEST_OUTPUT_TOKENS")
    current = capture_execution_profile(Settings(_env_file=None), "ollama", ToolRegistry())
    payload = current.model_dump()
    payload["settings"].pop("max_request_output_tokens")
    old = ExecutionProfile.model_validate(payload)
    assert compare_profiles(old, current).status == "matching"
    changed = capture_execution_profile(
        Settings(_env_file=None, max_request_output_tokens=512), "ollama", ToolRegistry()
    )
    assert "settings.max_request_output_tokens" in compare_profiles(old, changed).changed_fields


def test_request_cap_evidence_survives_api_restart(tmp_path):
    provider = ScriptedProvider()
    provider.script(make_response(content="Reviewed", prompt_tokens=10, completion_tokens=5))
    with TestClient(make_app(tmp_path, provider, max_request_output_tokens=10)) as api:
        sid = api.post("/sessions", json={"user_id": "dev"}).json()["session_id"]
        result = api.post("/chat", json={"user_id": "dev", "session_id": sid, "message": "Inspect"})
        assert result.json()["completion_status"] == "completed"
        saved = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        assert (
            saved["responses"][0]["execution_profile"]["settings"]["max_request_output_tokens"]
            == 10
        )
    with TestClient(make_app(tmp_path, provider, max_request_output_tokens=10)) as api:
        assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
        review = api.post(f"/sessions/{sid}/configuration-review", json={"user_id": "dev"}).json()
        assert review["context"]["request_output_allowance"] == 10
    assert len(provider.calls) == 1
