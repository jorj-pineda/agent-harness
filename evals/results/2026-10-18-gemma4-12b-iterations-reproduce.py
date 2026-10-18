"""Frozen six-attempt comparison of explicit 8/12-iteration local profiles."""

# Local synchronous metadata/cleanup run outside model requests.
# ruff: noqa: ASYNC212, ASYNC221
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import httpx

from evals.real_run import load_tasks, run_task, validate_task, write_report
from harness.config import Settings
from harness.providers import build_configured_provider

BASE = "f25c7e0eb7755effa017f09c15910dd7730cd31c"
MODEL = "gemma4:12b"
DIGEST = "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"
IDS = ("extract_label_normalizer", "positive_total_existing_failure", "retry_after_seconds")
LIMITS = (8, 12)
OPTIONS = {
    "ollama_host": "http://127.0.0.1:11434",
    "ollama_model": MODEL,
    "ollama_num_ctx": 16384,
    "ollama_think": False,
    "coding_prompt_profile": "compact",
    "coding_toolset": "full",
    "max_tool_calls_per_turn": 24,
    "max_turn_wall_seconds": 180,
    "request_timeout_seconds": 60,
    "max_completion_tokens_per_turn": 8192,
    "max_request_output_tokens": 2048,
    "max_total_tokens_per_turn": 64000,
    "max_context_tokens": 16384,
    "min_request_output_tokens": 256,
    "prompt_estimate_bytes_per_token": 3,
    "max_completion_retries": 1,
    "max_identical_tool_calls": 2,
    "max_files_touched_per_turn": 4,
    "require_plan_before_edit": False,
    "require_verification_before_finish": False,
    "project_check_argv": None,
    "load_root_agents": False,
    "enable_support_tools": False,
}


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def profile(limit):
    return {
        "provider": "ollama",
        "model": MODEL,
        "max_context_tokens": 16384,
        "max_request_output_tokens": 2048,
        "ollama_think": False,
        "max_tool_iterations": limit,
    }


def pinned_files(tasks):
    hashes = {
        "tasks.yaml": hashlib.sha256(Path("evals/real_tasks/tasks.yaml").read_bytes()).hexdigest()
    }
    for task in tasks:
        for kind in ("fixture", "acceptance", "reference"):
            root = getattr(task, kind)
            for path in sorted(root.rglob("*")):
                if path.is_file() and not any(
                    part in {"__pycache__", ".pytest_cache"} for part in path.parts
                ):
                    hashes[f"{task.id}/{kind}/{path.relative_to(root)}"] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
    return hashes


async def run(output):
    if git("rev-parse", "HEAD") != BASE or git("status", "--porcelain"):
        raise ValueError("Use the pinned clean harness checkout")
    if output.exists():
        raise ValueError("Use a new output directory; no reports are overwritten")
    tasks = [next(task for task in load_tasks() if task.id == task_id) for task_id in IDS]
    for task in tasks:
        await validate_task(task)
    before = pinned_files(tasks)
    defaults = {
        name: field.get_default(call_default_factory=True)
        for name, field in Settings.model_fields.items()
    }
    settings = {
        limit: Settings(
            _env_file=None,
            **{
                **defaults,
                **OPTIONS,
                "model_profiles": {"local": profile(limit)},
                "model_profile": "local",
            },
        )
        for limit in LIMITS
    }
    with httpx.Client(base_url=OPTIONS["ollama_host"], timeout=10) as client:
        installed = next(
            row for row in client.get("/api/tags").json()["models"] if row["name"] == MODEL
        )
        if installed["digest"] != DIGEST:
            raise ValueError("Installed model digest differs from frozen recipe")
        initially_loaded = any(
            row["name"] == MODEL for row in client.get("/api/ps").json()["models"]
        )
        version = client.get("/api/version").json()["version"]
    output.mkdir(parents=True)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    metadata = {
        "harness_revision": BASE,
        "started_at": started,
        "model": MODEL,
        "model_digest": DIGEST,
        "ollama_version": version,
        "model_initially_loaded": initially_loaded,
        "fixed_overrides": OPTIONS,
        "selected_profiles": {str(limit): profile(limit) for limit in LIMITS},
        "mode": "harness",
        "repeats": 1,
        "planned_attempts": 6,
        "manual_patch_repairs": 0,
        "pinned_file_sha256": before,
        "order": [
            {"task": task.id, "limits": list(LIMITS if i % 2 == 0 else LIMITS[::-1])}
            for i, task in enumerate(tasks)
        ],
        "active_model_observations": [],
        "cleanup": "not_started",
        "scope": (
            "Public trusted development fixtures; one attempt per task/limit, "
            "no held-out or causal quality claim. Reports remain separate by limit."
        ),
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    rows = {limit: [] for limit in LIMITS}
    provider = build_configured_provider("ollama", settings[8])
    try:
        for index, task in enumerate(tasks):
            for limit in LIMITS if index % 2 == 0 else LIMITS[::-1]:
                if git("rev-parse", "HEAD") != BASE or git("status", "--porcelain"):
                    raise RuntimeError("Harness checkout changed; experiment stopped")
                print(f"START {task.id} limit={limit}", flush=True)
                result = await run_task(
                    task, provider=provider, settings=settings[limit], mode="harness"
                )
                rows[limit].append(result)
                write_report(output / f"iterations-{limit}.json", started, rows[limit])
                with httpx.Client(base_url=OPTIONS["ollama_host"], timeout=10) as client:
                    metadata["active_model_observations"].append(
                        {
                            "task": task.id,
                            "limit": limit,
                            "models": [
                                row
                                for row in client.get("/api/ps").json()["models"]
                                if row["name"] == MODEL
                            ],
                        }
                    )
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "finished": sum(len(group) for group in rows.values()),
                            "task": task.id,
                            "limit": limit,
                            "accepted": result.passed,
                            "completion": result.completion_status,
                            "termination": result.termination,
                            "verification": result.verification_status,
                            "seconds": round(result.latency_ms / 1000, 1),
                            "tools": len(result.tool_trace),
                        }
                    ),
                    flush=True,
                )
                if pinned_files(tasks) != before:
                    raise RuntimeError("Pinned task/source/criteria/reference changed")
                if result.termination.startswith("error:"):
                    raise RuntimeError("Provider/runtime error; no retry or fallback")
    finally:
        await provider.aclose()
        if not initially_loaded:
            cleanup = subprocess.run(
                ["ollama", "stop", MODEL], capture_output=True, text=True, timeout=30
            )
            metadata["cleanup"] = (
                "unloaded_experiment_model" if cleanup.returncode == 0 else "unload_failed"
            )
        else:
            metadata["cleanup"] = "preexisting_model_left_loaded"
        with httpx.Client(base_url=OPTIONS["ollama_host"], timeout=10) as client:
            metadata["model_loaded_after_cleanup"] = any(
                row["name"] == MODEL for row in client.get("/api/ps").json()["models"]
            )
        metadata["pinned_files_unchanged"] = pinned_files(tasks) == before
        metadata["finished_attempts"] = sum(len(group) for group in rows.values())
        metadata["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: metadata[key]
                    for key in (
                        "cleanup",
                        "model_loaded_after_cleanup",
                        "pinned_files_unchanged",
                        "finished_attempts",
                    )
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    asyncio.run(run(parser.parse_args().output_dir))
