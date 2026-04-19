"""@tool decorator + module-level default registry.

The decorator introspects the function signature to build a Tool:
  - name: function name (overridable)
  - description: first line of the docstring
  - input_model: the sole parameter's type annotation, which must be a
    Pydantic BaseModel subclass

The decorator returns the original function so it stays directly callable
in tests and internal code paths.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from typing import Any, TypeVar, overload

from pydantic import BaseModel

from .base import Tool, ToolError
from .registry import ToolRegistry

F = TypeVar("F", bound=Callable[..., Any])

_default_registry = ToolRegistry()


def get_default_registry() -> ToolRegistry:
    return _default_registry


@overload
def tool(fn: F, /) -> F: ...
@overload
def tool(*, name: str | None = ..., registry: ToolRegistry | None = ...) -> Callable[[F], F]: ...


def tool(
    fn: F | None = None,
    /,
    *,
    name: str | None = None,
    registry: ToolRegistry | None = None,
) -> F | Callable[[F], F]:
    def decorator(func: F) -> F:
        # `registry or _default_registry` is wrong: ToolRegistry defines __len__,
        # so an empty registry is falsy and silently routes to the default.
        target = _default_registry if registry is None else registry
        target.register(_build_tool(func, name=name))
        return func

    if fn is not None:
        return decorator(fn)
    return decorator


def _build_tool(func: Callable[..., Any], *, name: str | None) -> Tool:
    params = list(inspect.signature(func).parameters.values())
    if len(params) != 1:
        raise ToolError(f"Tool {func.__name__!r} must take exactly one argument (a Pydantic model)")
    (param,) = params

    try:
        hints = typing.get_type_hints(func)
    except (NameError, TypeError) as exc:
        raise ToolError(f"Tool {func.__name__!r} has unresolved type hints: {exc}") from exc

    annotation = hints.get(param.name)
    if (
        annotation is None
        or not isinstance(annotation, type)
        or not issubclass(annotation, BaseModel)
    ):
        raise ToolError(
            f"Tool {func.__name__!r} parameter {param.name!r} must be annotated "
            "with a Pydantic BaseModel subclass"
        )

    doc = inspect.getdoc(func)
    if not doc:
        raise ToolError(f"Tool {func.__name__!r} must have a docstring")
    description = doc.split("\n", 1)[0].strip()

    return Tool(
        name=name or func.__name__,
        description=description,
        input_model=annotation,
        fn=func,
    )
