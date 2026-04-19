"""Core tool types.

A Tool is a Python function that takes a single Pydantic input model and
returns any JSON-serializable value. Using a Pydantic model for input lets
`model_json_schema()` do the schema work — no docstring parsing, no hand-
rolled walkers over type hints.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from providers.base import ToolSpec


class ToolError(Exception):
    """Raised when a tool fails input validation or execution."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[..., Any]

    def to_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters_schema=self.input_model.model_json_schema(),
        )
