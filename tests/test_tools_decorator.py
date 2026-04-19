from __future__ import annotations

import pytest
from pydantic import BaseModel

from tools import ToolError, ToolRegistry, get_default_registry, tool


class _Args(BaseModel):
    value: int


def test_decorator_infers_name_description_and_input_model() -> None:
    reg = ToolRegistry()

    @tool(registry=reg)
    def doubler(args: _Args) -> int:
        """Doubles the value."""
        return args.value * 2

    got = reg.get("doubler")
    assert got.name == "doubler"
    assert got.description == "Doubles the value."
    assert got.input_model is _Args
    # decorator returns the original function — still directly callable
    assert doubler(_Args(value=3)) == 6


def test_decorator_uses_first_docstring_line_only() -> None:
    reg = ToolRegistry()

    @tool(registry=reg)
    def summary_only(args: _Args) -> int:
        """Short summary.

        Longer explanation we don't ship to the model.
        """
        return args.value

    assert reg.get("summary_only").description == "Short summary."


def test_decorator_respects_explicit_name_override() -> None:
    reg = ToolRegistry()

    @tool(name="custom", registry=reg)
    def some_fn(args: _Args) -> int:
        """Doc."""
        return args.value

    assert "custom" in reg
    assert "some_fn" not in reg


def test_decorator_produces_provider_ready_spec() -> None:
    reg = ToolRegistry()

    @tool(registry=reg)
    def echo(args: _Args) -> int:
        """Echoes the value back."""
        return args.value

    (spec,) = reg.as_tool_specs()
    assert spec.name == "echo"
    assert spec.description == "Echoes the value back."
    assert spec.parameters_schema["properties"]["value"]["type"] == "integer"


def test_bare_decorator_form_registers_in_default_registry() -> None:
    @tool
    def unique_bare_form_tool_xyz(args: _Args) -> int:
        """Doc."""
        return args.value

    assert "unique_bare_form_tool_xyz" in get_default_registry()


def test_paren_decorator_form_registers_in_default_registry() -> None:
    @tool()
    def unique_paren_form_tool_xyz(args: _Args) -> int:
        """Doc."""
        return args.value

    assert "unique_paren_form_tool_xyz" in get_default_registry()


def test_missing_docstring_raises() -> None:
    reg = ToolRegistry()
    with pytest.raises(ToolError, match="docstring"):

        @tool(registry=reg)
        def no_doc(args: _Args) -> int:
            return args.value


def test_missing_input_annotation_raises() -> None:
    reg = ToolRegistry()
    with pytest.raises(ToolError, match="Pydantic BaseModel"):

        @tool(registry=reg)
        def bad(args) -> int:  # type: ignore[no-untyped-def]
            """Doc."""
            return 0


def test_non_basemodel_annotation_raises() -> None:
    reg = ToolRegistry()
    with pytest.raises(ToolError, match="Pydantic BaseModel"):

        @tool(registry=reg)
        def bad(args: int) -> int:
            """Doc."""
            return 0


def test_wrong_arity_raises() -> None:
    reg = ToolRegistry()
    with pytest.raises(ToolError, match="exactly one argument"):

        @tool(registry=reg)
        def too_many(a: _Args, b: _Args) -> int:
            """Doc."""
            return 0
