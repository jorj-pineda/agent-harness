"""Tool registry: name→Tool lookup with duplicate/missing protection.

Registries are plain objects — the harness creates one per session (or uses
a shared default). Nothing here is a module-level singleton; the @tool
decorator (3b) will accept a registry argument.
"""

from __future__ import annotations

from collections.abc import Iterator

from providers.base import ToolSpec

from .base import Tool, ToolError


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolError(f"Tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"Unknown tool: {name!r}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def as_tool_specs(self) -> list[ToolSpec]:
        return [t.to_spec() for t in self._tools.values()]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
