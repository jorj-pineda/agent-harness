"""Explicit inspection-tool policy; this is not host process isolation."""

from tools import ToolRegistry

INSPECTION_TOOL_NAMES = frozenset({"read_file", "grep_repo", "list_dir", "tree", "recall_facts"})
INSPECTION_BLOCK_REASON = (
    "Inspection-only session blocked a tool outside its read/recall allowlist. "
    "No tools in this batch were run. Create a separate standard session for other work."
)


def execution_registry(registry: ToolRegistry, *, inspection_only: bool) -> ToolRegistry:
    if not inspection_only:
        return registry
    filtered = ToolRegistry()
    for tool in registry:
        if tool.name in INSPECTION_TOOL_NAMES:
            filtered.register(tool)
    return filtered
