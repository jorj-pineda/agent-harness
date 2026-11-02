from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, field_validator

from tools import Tool, ToolError, ToolRegistry


class _Args(BaseModel):
    value: int


def _registry_with(fn) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(name="t", description="d", input_model=_Args, fn=fn))
    return reg


async def test_invoke_runs_sync_tool() -> None:
    def run(args: _Args) -> int:
        return args.value * 2

    reg = _registry_with(run)
    assert await reg.invoke("t", {"value": 3}) == 6


async def test_invoke_runs_async_tool() -> None:
    async def run(args: _Args) -> int:
        return args.value + 1

    reg = _registry_with(run)
    assert await reg.invoke("t", {"value": 4}) == 5


async def test_invoke_wraps_validation_error_as_tool_error() -> None:
    def run(args: _Args) -> int:
        return args.value

    reg = _registry_with(run)
    with pytest.raises(ToolError, match="Invalid arguments"):
        await reg.invoke("t", {"value": "not-an-int"})


async def test_invoke_wraps_missing_field_as_tool_error() -> None:
    def run(args: _Args) -> int:
        return args.value

    reg = _registry_with(run)
    with pytest.raises(ToolError, match="Invalid arguments"):
        await reg.invoke("t", {})


async def test_validation_feedback_keeps_nested_locations_without_input_or_pydantic_noise() -> None:
    class Args(BaseModel):
        counts: list[int]
        enabled: bool

    calls = []
    reg = ToolRegistry()
    reg.register(Tool(name="counts", description="d", input_model=Args, fn=calls.append))
    with pytest.raises(ToolError) as error:
        await reg.invoke("counts", {"counts": [1, "invalid-value-marker"]})
    message = str(error.value)
    assert "counts.1: Input should be a valid integer" in message
    assert "enabled: Field required" in message
    assert "invalid-value-marker" not in message
    assert "input_value" not in message
    assert "input_type" not in message
    assert "https://errors.pydantic.dev" not in message
    assert calls == []


async def test_validation_feedback_preserves_custom_correction_without_rewriting() -> None:
    class Args(BaseModel):
        argv: list[str]

        @field_validator("argv", mode="before")
        @classmethod
        def array_only(cls, value: object) -> object:
            if isinstance(value, str):
                raise ValueError('Use separate elements: {"argv": ["python", "-m", "pytest"]}.')
            return value

    calls = []
    reg = ToolRegistry()
    reg.register(Tool(name="check", description="d", input_model=Args, fn=calls.append))
    arguments = {"argv": "python -m pytest"}
    with pytest.raises(ToolError) as error:
        await reg.invoke("check", arguments)
    assert 'argv: Value error, Use separate elements: {"argv": ["python", "-m", "pytest"]}.' in str(
        error.value
    )
    assert arguments == {"argv": "python -m pytest"}
    assert calls == []
    await reg.invoke("check", {"argv": ["python", "-m", "pytest"]})
    assert len(calls) == 1
    assert calls[0].argv == ["python", "-m", "pytest"]


async def test_invoke_unknown_tool_raises() -> None:
    reg = ToolRegistry()
    with pytest.raises(ToolError, match="Unknown tool"):
        await reg.invoke("ghost", {})


async def test_invoke_wraps_arbitrary_exception_in_tool_error() -> None:
    def boom(args: _Args) -> int:
        raise RuntimeError("kaboom")

    reg = _registry_with(boom)
    with pytest.raises(ToolError, match="kaboom"):
        await reg.invoke("t", {"value": 1})


async def test_invoke_passes_through_tool_error_from_tool() -> None:
    def raises_tool_error(args: _Args) -> int:
        raise ToolError("specific failure")

    reg = _registry_with(raises_tool_error)
    with pytest.raises(ToolError, match="specific failure"):
        await reg.invoke("t", {"value": 1})


async def test_invoke_times_out_slow_async_tool() -> None:
    async def slow(args: _Args) -> int:
        await asyncio.sleep(1.0)
        return args.value

    reg = _registry_with(slow)
    with pytest.raises(ToolError, match="timed out"):
        await reg.invoke("t", {"value": 1}, timeout=0.05)
