"""Evaluation-only launcher for the reviewed two-task provider diagnostic probe.

Load augmented rows from the execution manifest before allocating components.
This is not a general runner, a new execution policy, or authorization to infer.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from api.server import Components, ComponentsFactory, build_components, create_app
from harness.config import Settings
from harness.router import ProviderRouter
from harness.trace import request_identity
from providers.base import ChatMessage, ChatProvider, ProviderResponse, ToolSpec


class InitialIdentity(BaseModel):
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProbeTask(BaseModel):
    id: Literal["T05", "T09"]
    directory: Path
    workspace_root: Path
    settings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_initial_identity: InitialIdentity


class ProbeManifest(BaseModel):
    kind: Literal["unrun-two-task-provider-diagnostic-freeze"]
    tasks: list[ProbeTask] = Field(min_length=2, max_length=2)


def load_task(root: Path, task_id: str) -> tuple[ProbeTask, Settings]:
    """Ignore raw tasks.json; reject missing/invalid capture evidence before resources."""
    root = root.resolve(strict=True)
    manifest = ProbeManifest.model_validate_json(
        (root / "provider-diagnostic-freeze.json").read_bytes()
    )
    if [task.id for task in manifest.tasks] != ["T05", "T09"]:
        raise ValueError("Expected frozen T05 then T09")
    task = next((task for task in manifest.tasks if task.id == task_id), None)
    if task is None:
        raise ValueError("Unknown probe task")
    directory = root / "tasks" / task.id
    if task.directory != directory or task.workspace_root != directory / "workspace":
        raise ValueError("Frozen task paths do not match probe root")
    if task.workspace_root.resolve(strict=True) != task.workspace_root:
        raise ValueError("Frozen workspace identity path changed")
    data = (directory / "settings.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != task.settings_sha256:
        raise ValueError("Frozen settings changed")
    settings = Settings(_env_file=None, **json.loads(data))
    if (
        settings.default_provider != "ollama"
        or settings.ollama_model != "gemma4:12b"
        or settings.coding_toolset != "whole_file"
        or settings.enable_support_tools
    ):
        raise ValueError("Unexpected probe configuration")
    return task, settings


class FirstRequestGuard:
    def __init__(self, inner: ChatProvider, expected: str, *, review_only: bool = False):
        self.inner = inner
        self.name = inner.name
        self.expected = expected
        self.review_only = review_only
        self.forwarded_requests = 0

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        if self.review_only:
            raise ValueError("Review-only probe cannot dispatch inference")
        if self.forwarded_requests == 0:
            identity = request_identity(messages, tools or [], max_tokens)
            if identity is None or identity.request_sha256 != self.expected or temperature != 0.0:
                raise ValueError("Frozen first-input mismatch; no inference forwarded")
        self.forwarded_requests += 1
        return await self.inner.chat(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )

    async def aclose(self) -> None:
        close = getattr(self.inner, "aclose", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result


def create_probe_app(
    root: Path,
    task_id: str,
    phase: Literal["source", "review"],
    *,
    components_builder: ComponentsFactory = build_components,
) -> FastAPI:
    task, settings = load_task(root, task_id)
    if phase not in {"source", "review"}:
        raise ValueError("Unknown probe phase")

    def factory(current: Settings) -> Components:
        components = components_builder(current)
        components.providers["ollama"] = FirstRequestGuard(
            components.providers["ollama"],
            task.source_initial_identity.request_sha256,
            review_only=phase == "review",
        )
        components.router = ProviderRouter(components.providers, default="ollama")
        return components

    return create_app(settings=settings, components_factory=factory)


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=["T05", "T09"], required=True)
    parser.add_argument("--phase", choices=["source", "review"], required=True)
    args = parser.parse_args()
    app = create_probe_app(args.root, args.task, args.phase)
    uvicorn.run(app, host="127.0.0.1", port=18757)


if __name__ == "__main__":
    main()
