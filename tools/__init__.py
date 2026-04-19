"""Tool registry package: @tool decorator, Tool type, registry.

Higher layers import from here only.
"""

from __future__ import annotations

from .base import Tool, ToolError
from .decorator import get_default_registry, tool
from .registry import ToolRegistry

__all__ = ["Tool", "ToolError", "ToolRegistry", "get_default_registry", "tool"]
