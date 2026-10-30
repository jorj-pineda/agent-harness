"""Capture production-adapter first inputs offline; never forwards inference."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import httpx

from evals.heldout_comparison import PLAN_PATH, validate_plan
from evals.heldout_guard import REGISTRATIONS
from evals.real_run import (
    MODES,
    _harness_revision,
    _plan_cohort,
    _validate_report_path,
    load_tasks,
    run_task,
)
from harness.config import Settings
from harness.trace import request_identity
from providers.base import ChatMessage, ProviderResponse, ToolSpec
from providers.ollama import OllamaProvider


class CaptureComplete(RuntimeError):
    """Deliberate mock boundary stop, never a product-model attempt."""


def _save(path: Path, content: bytes) -> str:
    if len(content) > 2_000_000:
        raise ValueError("Input capture exceeds byte limit")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
    return hashlib.sha256(content).hexdigest()


def normalized_input(
    messages: list[ChatMessage],
    tools: list[ToolSpec] | None,
    temperature: float,
    max_tokens: int | None,
) -> bytes:
    identity = request_identity(messages, tools or [], max_tokens)
    if identity is None:
        raise ValueError("Normalized input identity unavailable")
    content = json.dumps(
        {
            "identity": identity.model_dump(),
            "messages": [m.model_dump(mode="json") for m in messages],
            "tools": [t.model_dump(mode="json") for t in tools] if tools is not None else None,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        sort_keys=True,
    ).encode()
    if len(content) > 2_000_000:
        raise ValueError("Normalized input exceeds capture bound")
    return content


class CaptureProvider:
    name = "ollama"

    def __init__(self, adapter: OllamaProvider, normalized: list[bytes]) -> None:
        self.adapter, self.normalized = adapter, normalized

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        if self.normalized:
            raise ValueError("Unexpected second request during first-input capture")
        self.normalized.append(normalized_input(messages, tools, temperature, max_tokens))
        return await self.adapter.chat(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )


def _prepare_directory(directory: Path) -> None:
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)


def _plan_digest(path: Path) -> str:
    with path.open("rb") as stream:
        content = stream.read(64001)
    if len(content) > 64000:
        raise ValueError("Plan exceeds capture bound")
    return hashlib.sha256(content).hexdigest()


async def capture_inputs(
    directory: Path, plan_path: Path = PLAN_PATH, *, prepared: bool = False
) -> dict[str, Any]:
    before_plan = _plan_digest(plan_path)
    plan = validate_plan(plan_path)
    manifest = REGISTRATIONS[0].root / "tasks.yaml"
    tasks = load_tasks(manifest)
    _validate_report_path(directory / "index.json", manifest, tasks)
    # A fresh exclusive directory makes interrupted/failed captures non-retryable artifacts.
    _prepare_directory(directory)
    ledger, schedule = _plan_cohort(tasks, MODES, plan.repeats)
    settings = Settings(
        _env_file=None,
        _env_prefix="HELDOUT_INPUT_CAPTURE_UNUSED_",
        ollama_model=plan.model,
        **plan.runtime.model_dump(),
    )
    rows = []
    prior: dict[tuple[str, str], tuple[str, str]] = {}
    for index, (task, mode, attempt) in enumerate(schedule):
        prepared_directory = None
        if prepared:
            from evals.heldout_workspaces import prepare_fixture

            prepared_directory = directory / f"cell-{index + 1:02d}"
            prepare_fixture(task.id, prepared_directory)
        normalized: list[bytes] = []
        wire: list[bytes] = []

        def intercepted(request: httpx.Request, captured: list[bytes] = wire) -> httpx.Response:
            if request.method != "POST" or request.url.path != "/api/chat" or captured:
                raise ValueError("Unexpected production-adapter request")
            captured.append(request.content)
            raise CaptureComplete("Mock boundary: no request forwarded")

        adapter = OllamaProvider(
            host="http://capture.invalid",
            model=plan.model,
            embed_model="unused",
            timeout_seconds=plan.runtime.request_timeout_seconds,
            num_ctx=plan.runtime.ollama_num_ctx,
            think=plan.runtime.ollama_think,
            transport=httpx.MockTransport(intercepted),
        )
        try:
            result = await run_task(
                task,
                provider=CaptureProvider(adapter, normalized),
                settings=settings,
                mode=mode,
                attempt=attempt,
                prepared_directory=prepared_directory,
            )
        finally:
            await adapter.aclose()
        if len(normalized) != 1 or len(wire) != 1 or result.termination != "error:CaptureComplete":
            raise ValueError("First-input capture did not reach the unique mock boundary")
        _validate_report_path(directory / "index.json", manifest, tasks)
        names = (f"{index + 1:02d}-normalized.json", f"{index + 1:02d}-wire.json")
        digests = tuple(
            _save(directory / name, content)
            for name, content in zip(names, (normalized[0], wire[0]), strict=True)
        )
        key = (task.id, mode)
        repeat_equal = prior[key] == digests if key in prior else None
        prior[key] = digests
        rows.append(
            {
                "ordinal": index + 1,
                "task_id": task.id,
                "mode": mode,
                "attempt": attempt,
                "files": dict(zip(names, digests, strict=True)),
                "runtime_config": result.runtime_config,
                "repeat_bytes_equal": repeat_equal,
            }
        )
    if _plan_digest(plan_path) != before_plan:
        raise ValueError("Plan changed during capture")
    result_index = {
        "format": "heldout-first-inputs-v1",
        "evidence_kind": "mock-production-adapter",
        "inference_ran": False,
        "forwarded_requests": 0,
        "runtime": _harness_revision(),
        "plan_sha256": _plan_digest(plan_path),
        "seal_sha256": plan.seal_sha256,
        "cells": rows,
        "planned_cells": len(ledger.cells),
    }
    _save(directory / "index.json", json.dumps(result_index, indent=2).encode())
    return result_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--prepared-workspaces", action="store_true")
    args = parser.parse_args()
    asyncio.run(capture_inputs(args.directory, prepared=args.prepared_workspaces))
    print("Captured sixteen first-input pairs; zero requests forwarded. Not launch approval.")


if __name__ == "__main__":
    main()
