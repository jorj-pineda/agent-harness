"""Owned GET-only point observations, not proof of continuous model residency."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from evals.heldout_backend_rehearsal import ENDPOINT
from evals.heldout_comparison import validate_plan
from evals.heldout_environment import Tags, _get
from evals.heldout_inputs import _save


async def record_residency(directory: Path, ordinal: int, seconds: int, check_owned) -> None:
    """Check ownership around one bounded GET; no warmup, retry or unload request."""
    check_owned()
    async with httpx.AsyncClient(
        base_url=ENDPOINT, timeout=5, trust_env=False, follow_redirects=False
    ) as client:
        data = Tags.model_validate_json(await _get(client, "/api/ps"))
    check_owned()
    plan = validate_plan()
    if len(data.models) > 1 or any(
        model.name != plan.model or model.model != plan.model or model.digest != plan.model_digest
        for model in data.models
    ):
        raise ValueError("Residency observation contains an unexpected loaded model")
    _save(
        directory / f"residency-{ordinal:02d}.json",
        json.dumps(
            {
                "format": "owned-residency-observation-v1",
                "observed_at": datetime.now(UTC).isoformat(),
                "selected_keep_alive_seconds": seconds,
                "state": "loaded" if data.models else "not_loaded",
                "models": [model.model_dump() for model in data.models],
                "continuous_residency_verified": False,
            }
        ).encode(),
    )


class ResidencyObservedProvider:
    """Preserve a post-response observation before returning tool-bearing responses."""

    def __init__(self, inner, observe) -> None:
        self.inner, self.observe = inner, observe
        self.name = inner.name
        self.responses = 0
        self.failed = False
        self.observation_failed = False

    @property
    def stop_reason(self):
        return "residency_observation" if self.observation_failed else self.inner.stop_reason

    async def chat(self, messages, *, tools=None, temperature=0, max_tokens=None):
        if self.failed:
            raise ValueError("Residency observation provider is latched closed")
        try:
            response = await self.inner.chat(
                messages, tools=tools, temperature=temperature, max_tokens=max_tokens
            )
            self.responses += 1
            try:
                await self.observe(self.responses)
            except BaseException:
                self.observation_failed = True
                raise
            return response
        except BaseException:
            self.failed = True
            raise

    async def aclose(self):
        await self.inner.aclose()
