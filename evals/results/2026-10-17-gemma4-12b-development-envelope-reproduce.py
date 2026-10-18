"""Frozen six-attempt recipe; run from the pinned clean harness checkout."""

# Frozen recipe: synchronous local metadata/cleanup occur outside model requests.
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

from evals.real_run import MODES, load_tasks, run_task, validate_task, write_report
from harness.config import Settings
from harness.providers import build_configured_provider

BASE = "8e6e454c93bfa40ef0af6541de01feb71d18da66"
MODEL = "gemma4:12b"
MODEL_DIGEST = "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"
IDS = ("extract_label_normalizer", "positive_total_existing_failure", "retry_after_seconds")
OPTIONS = {
    "ollama_host": "http://127.0.0.1:11434",
    "ollama_model": MODEL,
    "ollama_num_ctx": 16384,
    "ollama_think": False,
    "coding_prompt_profile": "compact",
    "coding_toolset": "full",
    "max_tool_iterations": 8,
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


def pinned_files(tasks):
    evidence = {}
    for task in tasks:
        for kind in ("fixture", "acceptance", "reference"):
            root = getattr(task, kind)
            for path in sorted(root.rglob("*")):
                if path.is_file() and not any(
                    p in {"__pycache__", ".pytest_cache"} for p in path.parts
                ):
                    evidence[f"{task.id}/{kind}/{path.relative_to(root)}"] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
    return evidence


async def run(report):
    if git("rev-parse", "HEAD") != BASE or git("status", "--porcelain"):
        raise ValueError("Run only from the pinned clean harness checkout")
    metadata_path = report.with_suffix(".metadata.json")
    if report.exists() or metadata_path.exists():
        raise ValueError("Use a new report path; results are never overwritten")
    report.parent.mkdir(parents=True, exist_ok=True)
    tasks = [next(task for task in load_tasks() if task.id == task_id) for task_id in IDS]
    for task in tasks:
        await validate_task(task)
    before = pinned_files(tasks)
    defaults = {
        name: field.get_default(call_default_factory=True)
        for name, field in Settings.model_fields.items()
    }
    settings = Settings(_env_file=None, **{**defaults, **OPTIONS})
    with httpx.Client(base_url=OPTIONS["ollama_host"], timeout=10) as client:
        installed = next(
            row for row in client.get("/api/tags").json()["models"] if row["name"] == MODEL
        )
        if installed["digest"] != MODEL_DIGEST:
            raise ValueError("Installed model digest differs from frozen recipe")
        initially_loaded = any(
            row["name"] == MODEL for row in client.get("/api/ps").json()["models"]
        )
        version = client.get("/api/version").json()["version"]
    started = datetime.now(UTC).isoformat(timespec="seconds")
    metadata = {
        "harness_revision": BASE,
        "model": MODEL,
        "model_digest": MODEL_DIGEST,
        "ollama_version": version,
        "model_initially_loaded": initially_loaded,
        "started_at": started,
        "settings_overrides": OPTIONS,
        "repeats": 1,
        "manual_patch_repairs": 0,
        "planned_attempts": 6,
        "mode_order": [
            {"task": task.id, "modes": list(MODES if index % 2 == 0 else MODES[::-1])}
            for index, task in enumerate(tasks)
        ],
        "pinned_file_sha256": before,
        "cleanup": "not_started",
        "scope": (
            "Real tools in disposable trusted development fixtures; one sample per task/mode, "
            "no held-out or causal quality claim."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    results = []
    provider = build_configured_provider("ollama", settings)
    try:
        for index, task in enumerate(tasks):
            for mode in MODES if index % 2 == 0 else MODES[::-1]:
                print(f"START {task.id} {mode}", flush=True)
                result = await run_task(task, provider=provider, settings=settings, mode=mode)
                results.append(result)
                write_report(report, started, results)
                print(
                    json.dumps(
                        {
                            "finished": len(results),
                            "task": task.id,
                            "mode": mode,
                            "accepted": result.passed,
                            "termination": result.termination,
                            "completion": result.completion_status,
                            "verification": result.verification_status,
                            "latency_s": round(result.latency_ms / 1000, 1),
                            "tools": len(result.tool_trace),
                        }
                    ),
                    flush=True,
                )
                if pinned_files(tasks) != before:
                    raise RuntimeError(
                        "Pinned source/acceptance/reference changed; evaluation stopped"
                    )
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
        metadata["finished_attempts"] = len(results)
        metadata["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "cleanup": metadata["cleanup"],
                    "model_loaded": metadata["model_loaded_after_cleanup"],
                    "pinned_files_unchanged": metadata["pinned_files_unchanged"],
                    "finished": len(results),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    asyncio.run(run(parser.parse_args().report))
