"""Read-only loopback Ollama metadata observation; no generation or lifecycle actions."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field

from evals.heldout_comparison import validate_plan
from evals.heldout_guard import REGISTRATIONS
from evals.heldout_inputs import _save
from evals.real_run import _validate_report_path, load_tasks


class Version(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    version: str = Field(min_length=1, max_length=128)


class Model(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    name: str = Field(min_length=1, max_length=256)
    model: str = Field(min_length=1, max_length=256)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class Tags(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    models: list[Model] = Field(max_length=1000)


def local_endpoint(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or parsed.hostname is None
        or not ipaddress.ip_address(parsed.hostname).is_loopback
    ):
        raise ValueError("Metadata endpoint must be a credential-free literal loopback HTTP origin")
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise ValueError("Invalid loopback port")
    return base_url.rstrip("/")


async def _get(client: httpx.AsyncClient, path: str) -> bytes:
    async with client.stream("GET", path) as response:
        response.raise_for_status()
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > 1_000_000:
                raise ValueError("Metadata response exceeds byte limit")
            body.extend(chunk)
        return bytes(body)


async def observe(
    base_url: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> dict[str, Any]:
    endpoint = local_endpoint(base_url)
    plan = validate_plan()
    async with httpx.AsyncClient(
        base_url=endpoint, transport=transport, timeout=5, follow_redirects=False, trust_env=False
    ) as client:
        version = Version.model_validate_json(await _get(client, "/api/version"))
        tags = Tags.model_validate_json(await _get(client, "/api/tags"))
    matches = [row for row in tags.models if row.name == plan.model or row.model == plan.model]
    if len(matches) > 1 or any(row.name != row.model for row in matches):
        raise ValueError("Ambiguous local model metadata")
    digest = matches[0].digest if matches else None
    return {
        "format": "heldout-local-metadata-v1",
        "observed_at": datetime.now(UTC).isoformat(),
        "endpoint": endpoint,
        "model": plan.model,
        "expected_version": plan.expected_ollama_version,
        "observed_version": version.version,
        "expected_digest": plan.model_digest,
        "observed_digest": digest,
        "matches_protocol": version.version == plan.expected_ollama_version
        and digest == plan.model_digest,
        "generation_requests": 0,
        "resource_ownership_verified": False,
        "inference_authorized": False,
    }


def _fresh_report(report: Path) -> None:
    if report.exists() or report.is_symlink():
        raise FileExistsError("Metadata report path already occupied")


async def record_observation(base_url: str, report: Path) -> dict[str, Any]:
    manifest = REGISTRATIONS[0].root / "tasks.yaml"
    tasks = load_tasks(manifest)
    _validate_report_path(report, manifest, tasks)
    _fresh_report(report)
    endpoint = local_endpoint(base_url)
    try:
        observation = await observe(endpoint)
    except (httpx.HTTPError, ValueError) as error:
        observation = {
            "format": "heldout-local-metadata-v1",
            "observed_at": datetime.now(UTC).isoformat(),
            "endpoint": endpoint,
            "status": "unavailable",
            "error_type": type(error).__name__,
            "matches_protocol": False,
            "generation_requests": 0,
            "resource_ownership_verified": False,
            "inference_authorized": False,
        }
    _validate_report_path(report, manifest, tasks)
    _save(report, json.dumps(observation, indent=2).encode())
    return observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(record_observation(args.base_url, args.report))
    print("Metadata recorded; ownership, backend compliance and launch approval remain unverified.")
    if result.get("status") == "unavailable":
        raise SystemExit("Local metadata unavailable; resolve the environment before freeze.")
    if not result["matches_protocol"]:
        raise SystemExit(
            "Metadata differs from protocol; no launch is permitted by this observation."
        )


if __name__ == "__main__":
    main()
