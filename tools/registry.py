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

from pydantic import BaseModel, ValidationError

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
        timeout: float | None = None,  # noqa: ASYNC109 — public override for tool deadlines
    ) -> Any:
        tool = self.get(name)

        try:
            validated = tool.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolError(f"Invalid arguments for tool {name!r}: {exc}") from exc

        effective_timeout = timeout if timeout is not None else tool.timeout_seconds
        if effective_timeout is None:
            effective_timeout = DEFAULT_INVOKE_TIMEOUT_S
        try:
            async with asyncio.timeout(effective_timeout):
                return await _execute_and_settle(tool, validated)
        except TimeoutError as exc:
            raise ToolError(f"Tool {name!r} timed out after {effective_timeout}s") from exc
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


async def _execute_and_settle(tool: Tool, validated: BaseModel) -> Any:
    """Keep invocation alive until owned work has stopped, even on cancellation.

    Sync worker threads cannot be interrupted. Async tools receive one cancel
    request, then get time to finish cleanup (including managed subprocesses).
    Further cancellation of the caller cannot abandon that cleanup. Deadlines
    may therefore be exceeded; they do not imply hard execution-time bounds.
    """
    is_async = inspect.iscoroutinefunction(tool.fn)
    task = asyncio.create_task(
        tool.fn(validated) if is_async else asyncio.to_thread(tool.fn, validated)
    )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if is_async:
            task.cancel()
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()  # Retrieve errors without replacing the original cancellation.
        raise
