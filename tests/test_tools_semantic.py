from __future__ import annotations

import pytest

from tools import ToolError, ToolRegistry
from tools.semantic import register_semantic_search_stub


async def test_semantic_search_stub_raises_with_guidance() -> None:
    reg = ToolRegistry()
    register_semantic_search_stub(reg)
    with pytest.raises(ToolError, match="not enabled"):
        await reg.invoke("semantic_search", {"query": "auth middleware"})
