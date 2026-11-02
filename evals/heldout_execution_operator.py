"""Explicit trusted-parent operator API; no CLI, approval generator or implicit launch."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import httpx

from evals.heldout_execution_admission import admit_execution
from evals.heldout_execution_cohort import run_admitted_cohort
from evals.heldout_execution_manifest import ExecutionManifest, ParentApproval
from evals.real_run import CohortLedger


def execute_selected(
    manifest_path: Path,
    *,
    manifest_sha256: str,
    approval_path: Path,
    approval_sha256: str,
    verify_parent: Callable[[ExecutionManifest, ParentApproval], None],
) -> CohortLedger:
    """Execute only after the caller independently verifies actual parent authority.

    A no-op verifier or arbitrary affirmative JSON is NOT approval. The trusted
    caller must have completed the exact freeze/review/CI/merge and fresh launch
    prerequisites. Calling this API can generate; it is not a preflight/rehearsal.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise ValueError("Enter selected execution outside an active event loop")
    admission = admit_execution(
        manifest_path,
        manifest_sha256=manifest_sha256,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        verify_parent=verify_parent,
    )

    def transport(ordinal: int, directory: Path) -> httpx.AsyncBaseTransport:
        # Constructed only inside a validated owned lifecycle, not during admission.
        return httpx.AsyncHTTPTransport(retries=0, trust_env=False)

    return run_admitted_cohort(admission, transport_factory=transport, scripted=False)
