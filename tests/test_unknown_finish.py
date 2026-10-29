import pytest
from anthropic.types import Message
from openai.types.chat import ChatCompletion

from harness.loop import run_turn
from harness.state import Session
from providers.anthropic import _response_from_anthropic
from providers.openai import _response_from_openai
from tests.api.conftest import ScriptedProvider
from tools import ToolRegistry


@pytest.mark.parametrize("backend", ["anthropic", "openai"])
@pytest.mark.parametrize("reason", [None, "unexpected_terminal"])
async def test_unsupported_finish_cannot_be_successful_completion(backend, reason):
    if backend == "anthropic":
        native = Message.model_validate(
            {
                "id": "fixture",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Partial answer"}],
                "model": "fixture",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 2},
            }
        ).model_copy(update={"stop_reason": reason})
        response = _response_from_anthropic(native, fallback_model="fixture", latency_ms=0)
    else:
        # SDK wire construction is non-strict and can retain unknown native reasons.
        native = ChatCompletion.model_construct(
            id="fixture",
            object="chat.completion",
            created=0,
            model="fixture",
            choices=[],
            usage=None,
        )
        from openai.types.chat import ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice

        native.choices = [
            Choice.model_construct(
                index=0,
                finish_reason=reason,
                message=ChatCompletionMessage(role="assistant", content="Partial answer"),
            )
        ]
        response = _response_from_openai(native, fallback_model="fixture", latency_ms=0)
    provider = ScriptedProvider()
    provider.script(response)
    result = await run_turn(
        session=Session(),
        user_input="Inspect",
        provider=provider,
        registry=ToolRegistry(),
        max_completion_retries=1,
    )
    assert result.completion_status == "incomplete"
    assert response.finish_reason == "error"
    assert result.answer == "Partial answer"
    assert len(provider.calls) == 1
