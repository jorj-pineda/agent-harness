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
