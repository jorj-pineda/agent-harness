import json

import pytest
from fastapi.testclient import TestClient

from harness.context import TokenBudget
from harness.loop import run_turn
from harness.state import Session
from harness.trace import RequestTrace
from providers.base import ChatMessage, TokenUsage, ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tests.api.test_session_persistence import make_app
from tests.api.test_stream import _parse_sse
from tools import ToolRegistry
from tools.code import build_code_tools
from workspace import Workspace


@pytest.mark.parametrize(
    "output,expected", [(0, None), (5, None), (6, "budget_exhausted"), (None, "incomplete")]
)
def test_effective_cap_preserves_boundary_and_unknown_counts(output, expected):
    budget = TokenBudget(max_request_output=20)
    budget.record(TokenUsage(prompt_tokens=1, completion_tokens=output), 1)
    stop = budget.after_response(wants_tools=True, request_output=5)
    assert (stop.status if stop else None) == expected
    assert budget.completion_used == output


def test_uncapped_and_omitted_allowances_preserve_existing_behavior():
    budget = TokenBudget()
    budget.record(TokenUsage(), 1)
    assert budget.after_response(wants_tools=True, request_output=None) is None
    configured = TokenBudget(max_request_output=5)
    configured.record(TokenUsage(prompt_tokens=1, completion_tokens=6), 1)
    assert configured.after_response(wants_tools=False).status == "budget_exhausted"


def test_context_only_finite_allowance_requires_reported_output():
    budget = TokenBudget(max_context=100)
    budget.record(TokenUsage(prompt_tokens=1), 1)
    stop = budget.after_response(wants_tools=False, request_output=5)
    assert stop.status == "incomplete"
    assert budget.completion_used is None


@pytest.mark.parametrize("constraint", ["context", "total"])
@pytest.mark.parametrize("configured_cap", [0, 20])
@pytest.mark.parametrize("wants_tools", [False, True])
async def test_reduced_allowance_stops_before_mutation_or_completion(
    tmp_path, constraint, configured_cap, wants_tools
):
    path = tmp_path / "user.txt"
    path.write_text("preserved")
    registry = ToolRegistry()
    for tool in build_code_tools(Workspace(root=tmp_path)):
        registry.register(tool)
    message = "Edit"
    estimate = TokenBudget().estimate_prompt(
        [ChatMessage(role="user", content=message)], registry.as_tool_specs()
    )
    provider = ScriptedProvider()
    provider.script(
        make_response(
            content="Claimed complete",
            prompt_tokens=1,
            completion_tokens=6,
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "user.txt", "content": "overwrite"},
                )
            ]
            if wants_tools
            else [],
        )
    )
    trace = []
    result = await run_turn(
        session=Session(),
        user_input=message,
        provider=provider,
        registry=registry,
        max_iterations=1,
        max_context_tokens=estimate + 5 if constraint == "context" else 0,
        max_total_tokens_per_turn=estimate + 5 if constraint == "total" else 0,
        max_request_output_tokens=configured_cap,
        trace=trace,
    )
    assert next(item for item in trace if isinstance(item, RequestTrace)).max_tokens == 5
    assert result.completion_status == "budget_exhausted"
    assert "per-request output-token cap" in result.completion_reason
    assert path.read_text() == "preserved"
    assert result.token_usage.completion_tokens == 6
    assert len(provider.calls) == 1
    assert all(call.result is None and "not executed" in call.error for call in result.tool_calls)


@pytest.mark.parametrize("transport", ["http", "stream"])
def test_effective_cap_refusal_survives_restart_without_resubmission(tmp_path, transport):
    class OverrunProvider(ScriptedProvider):
        async def chat(self, messages, **kwargs):
            allowance = kwargs["max_tokens"]
            assert 0 < allowance < 20_000
            self.script(
                make_response(
                    prompt_tokens=1,
                    completion_tokens=allowance + 1,
                    tool_calls=[
                        ToolCall(
                            id="edit",
                            name="write_file",
                            arguments={"path": "user.txt", "content": "overwrite"},
                        )
                    ],
                )
            )
            return await super().chat(messages, **kwargs)

    repo = tmp_path / "repo"
    repo.mkdir()
    path = repo / "user.txt"
    path.write_text("preserved")
    provider = OverrunProvider()
    with TestClient(
        make_app(tmp_path, provider, max_context_tokens=8000, max_request_output_tokens=20_000)
    ) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        payload = {"user_id": "dev", "session_id": sid, "message": "Edit", "request_id": "one"}
        if transport == "http":
            result = client.post("/chat", json=payload).json()
        else:
            events = _parse_sse(client.get("/chat/stream", params=payload).text)
            result = json.loads(events[-1]["data"])["response"]
        assert result["completion_status"] == "budget_exhausted"
        assert "per-request output-token cap" in result["completion_reason"]
        endpoint = f"/runs/{result['run_id']}"
        saved = client.get(endpoint, params={"user_id": "dev"}).json()
        request, response, stop = saved["budget_evidence"]["events"]
        assert response["usage"]["completion_tokens"] == request["max_tokens"] + 1
        assert stop["kind"] == "post_response_stop"
        assert "request_block" not in stop
        assert path.read_text() == "preserved"
        assert len(provider.calls) == 1
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted)) as client:
        assert client.get(endpoint, params={"user_id": "dev"}).json() == saved
        assert client.post("/chat", json=payload).json() == result
        assert restarted.calls == []
        assert path.read_text() == "preserved"
