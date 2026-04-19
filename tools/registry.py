"""Tool registry: name→Tool lookup + validated invocation.

Registries are plain objects — the harness creates one per session (or uses
a shared default). Nothing here is a module-level singleton; the @tool
decorator accepts a registry argument.

`invoke` is the single entry point the ReAct loop uses: it validates the
provider's raw argument dict against the tool's Pydantic input model,
dispatches sync or async tools uniformly, and wraps every failure mode in
`ToolError` so the loop has one exception type to handle.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Iterator
from typing import Any

from pydantic import ValidationError

from providers.base import ToolSpec

from .base import Tool, ToolError

DEFAULT_INVOKE_TIMEOUT_S = 30.0


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

    async def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float = DEFAULT_INVOKE_TIMEOUT_S,  # noqa: ASYNC109  (library-level default; callers may still wrap with asyncio.timeout)
    ) -> Any:
        tool = self.get(name)

        try:
            validated = tool.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolError(f"Invalid arguments for tool {name!r}: {exc}") from exc

        try:
            async with asyncio.timeout(timeout):
                if inspect.iscoroutinefunction(tool.fn):
                    return await tool.fn(validated)
                # Run sync tools in a worker thread so the event loop stays
                # responsive; the thread itself can't be interrupted, but control
                # returns to the harness on schedule.
                return await asyncio.to_thread(tool.fn, validated)
        except TimeoutError as exc:
            raise ToolError(f"Tool {name!r} timed out after {timeout}s") from exc
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"Tool {name!r} raised: {exc!r}") from exc

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
