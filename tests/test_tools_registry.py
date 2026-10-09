from __future__ import annotations

import pytest
from pydantic import BaseModel

from tools import Tool, ToolError, ToolRegistry


class _Args(BaseModel):
    x: int


def _noop(args: _Args) -> int:
    return args.x


def _make_tool(name: str = "noop") -> Tool:
    return Tool(name=name, description="does nothing", input_model=_Args, fn=_noop)


def test_register_and_lookup() -> None:
    reg = ToolRegistry()
    tool = _make_tool()
    reg.register(tool)

    assert "noop" in reg
    assert reg.get("noop") is tool
    assert reg.names() == ["noop"]
    assert len(reg) == 1


def test_duplicate_registration_raises() -> None:
    reg = ToolRegistry()
    reg.register(_make_tool())
    with pytest.raises(ToolError, match="already registered"):
        reg.register(_make_tool())


def test_unknown_tool_raises() -> None:
    reg = ToolRegistry()
    with pytest.raises(ToolError, match="Unknown tool"):
        reg.get("ghost")


def test_as_tool_specs_produces_provider_specs() -> None:
    reg = ToolRegistry()
    reg.register(_make_tool())
    specs = reg.as_tool_specs()

    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "noop"
    assert spec.description == "does nothing"
    schema = spec.parameters_schema
    assert schema["type"] == "object"
    assert "x" in schema["properties"]
    assert schema["properties"]["x"]["type"] == "integer"


def test_iter_yields_registered_tools_in_order() -> None:
    reg = ToolRegistry()
    a = _make_tool("a")
    b = _make_tool("b")
    reg.register(a)
    reg.register(b)

    assert list(reg) == [a, b]


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
async def test_async_cleanup_settles_despite_repeated_caller_cancellation(stop: str) -> None:
    import asyncio

    started = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    cleaned = False

    async def run(_args: _Args) -> None:
        nonlocal cleaned
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            cleaned = True

    reg = ToolRegistry()
    reg.register(Tool(name="cleanup", description="test cleanup", input_model=_Args, fn=run))
    call = asyncio.create_task(
        reg.invoke("cleanup", {"x": 1}, timeout=0.02 if stop == "timeout" else 10)
    )
    try:
        await asyncio.wait_for(started.wait(), 2)
        if stop == "cancel":
            call.cancel()
        await asyncio.wait_for(cleaning.wait(), 2)
        assert not call.done()
        if stop == "cancel":
            call.cancel()
            await asyncio.sleep(0)
            assert not call.done()
    finally:
        release.set()
        if stop == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await call
        else:
            with pytest.raises(ToolError, match="timed out"):
                await call
    assert cleaned
