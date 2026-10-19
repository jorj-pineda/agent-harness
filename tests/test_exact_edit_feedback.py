"""Offline wire/tool regressions for the frozen trial's escaping failure shapes.

Synthetic source and mocked inference validate diagnostics, not model recovery.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from providers.base import ChatMessage
from providers.ollama import OllamaProvider
from tools import ToolError, ToolRegistry
from tools.code import register_code_tools
from workspace import Workspace

SOURCE = 'function label() {\n  return "old";\n}\n'


@pytest.mark.parametrize(
    ("old_text", "expected_hint"),
    [
        # T01: literal escapes inside the span, but a real final newline.
        ('function label() {\\n  return "old";\\n}\n', "file has line breaks"),
        # T01's other call, T04 and T09: quotes have an extra escaping layer too.
        (
            'function label() {\\n  return \\"old\\";\\n}\n',
            "file has line breaks and double quotes",
        ),
        (
            'function label() {\\n  return \\"old\\";\\n}\\n',
            "file has line breaks and double quotes",
        ),
        ('function label() {\\n  return \\"old\\";\\n}', "file has line breaks and double quotes"),
        ('return \\"old\\";', "file has double quotes"),
    ],
)
async def test_ollama_exact_edit_feedback_preserves_arguments_and_requires_explicit_correction(
    tmp_path: Path, old_text: str, expected_hint: str
) -> None:
    target = tmp_path / "label.ts"
    target.write_text(SOURCE, encoding="utf-8")
    target.chmod(0o640)
    registry = ToolRegistry()
    register_code_tools(registry, workspace=Workspace(root=tmp_path))
    read = await registry.invoke("read_file", {"path": "label.ts"})
    arguments = {
        "path": "label.ts",
        "old_text": old_text,
        "new_text": 'function label() {\\n  return \\"new\\";\\n}',
        "expected_sha256": read["sha256"],
    }
    original_arguments = json.dumps(arguments)

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "tool_calls": [{"function": {"name": "replace_text", "arguments": arguments}}]
                }
            },
            request=request,
        )

    provider = OllamaProvider(
        host="http://fake",
        model="gemma4:12b",
        embed_model="unused",
        transport=httpx.MockTransport(respond),
    )
    try:
        response = await provider.chat([ChatMessage(role="user", content="Change the label")])
    finally:
        await provider.aclose()
    call = response.tool_calls[0]
    assert call.arguments == arguments
    with pytest.raises(ToolError, match=expected_hint) as rejected:
        await registry.invoke(call.name, call.arguments)
    assert "No edit was applied; arguments were not rewritten" in str(rejected.value)
    assert target.read_bytes() == SOURCE.encode()
    assert target.stat().st_mode & 0o777 == 0o640
    assert json.dumps(call.arguments) == original_arguments

    # A caller must supply an exact span and its own replacement, not replay the
    # malformed arguments. This is scripted correction, not a live-model result.
    corrected = {
        **call.arguments,
        "old_text": read["content"],
        "new_text": read["content"].replace('"old"', '"new"'),
    }
    result = await registry.invoke("replace_text", corrected)
    assert result["replacement_count"] == 1
    assert target.read_bytes() == SOURCE.replace('"old"', '"new"').encode()
    assert target.stat().st_mode & 0o777 == 0o640


@pytest.mark.parametrize(
    ("source", "old_text", "hint"),
    [
        ("x\ny\nx\ny\n", "x\\ny", "Copy old_text exactly"),
        ("x\ny\n", "missing\\ny", "Copy old_text exactly"),
        ("x\ny\n", "x\\ty", "Copy old_text exactly"),
        ("x\r\ny\n", "x\\ny", "Copy old_text exactly"),
        ("x\ny\nx\\ny x\\ny", "x\\ny", "found 2"),
        ('"a" "a"', '\\"a\\"', "Copy old_text exactly"),
    ],
)
async def test_escape_hint_requires_a_unique_candidate_and_keeps_original_match_priority(
    tmp_path: Path, source: str, old_text: str, hint: str
) -> None:
    target = tmp_path / "source.txt"
    target.write_bytes(source.encode())
    registry = ToolRegistry()
    register_code_tools(registry, workspace=Workspace(root=tmp_path))
    read = await registry.invoke("read_file", {"path": "source.txt"})
    with pytest.raises(ToolError, match=hint):
        await registry.invoke(
            "replace_text",
            {
                "path": "source.txt",
                "old_text": old_text,
                "new_text": "x",
                "expected_sha256": read["sha256"],
            },
        )
    assert target.read_bytes() == source.encode()


async def test_exact_literal_escapes_are_not_decoded_even_when_alternative_exists(
    tmp_path: Path,
) -> None:
    target = tmp_path / "source.txt"
    source = 'x\\ny \\"quoted\\"\nx\ny "quoted"\n'
    target.write_bytes(source.encode())
    registry = ToolRegistry()
    register_code_tools(registry, workspace=Workspace(root=tmp_path))
    read = await registry.invoke("read_file", {"path": "source.txt"})
    await registry.invoke(
        "replace_text",
        {
            "path": "source.txt",
            "old_text": 'x\\ny \\"quoted\\"',
            "new_text": "z\\nw",
            "expected_sha256": read["sha256"],
        },
    )
    assert target.read_bytes() == b'z\\nw\nx\ny "quoted"\n'


async def test_stale_hash_blocks_before_escape_diagnosis(tmp_path: Path) -> None:
    target = tmp_path / "source.txt"
    target.write_bytes(b"x\ny")
    registry = ToolRegistry()
    register_code_tools(registry, workspace=Workspace(root=tmp_path))
    read = await registry.invoke("read_file", {"path": "source.txt"})
    target.write_bytes(b"x\ny\nlater user work")
    with pytest.raises(ToolError, match="File changed since read_file"):
        await registry.invoke(
            "replace_text",
            {
                "path": "source.txt",
                "old_text": "x\\ny",
                "new_text": "z",
                "expected_sha256": read["sha256"],
            },
        )
    assert target.read_bytes() == b"x\ny\nlater user work"
