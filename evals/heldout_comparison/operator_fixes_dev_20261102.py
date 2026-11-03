"""Machine-bound trusted-parent recipe: check only unless --execute is explicit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from evals.heldout_backend_rehearsal import _free_port
from evals.heldout_execution_admission import refuse_withdrawn_selection
from evals.heldout_execution_manifest import ExecutionManifest, ParentApproval, _selected_bytes
from evals.heldout_execution_operator import execute_selected
from evals.heldout_execution_preflight import preflight
from evals.heldout_guard import _read_regular
from evals.heldout_preparation import RUNTIME_ROOT, check_existing_interpreter
from evals.real_run import _harness_revision

ROOT = Path("/Users/jorge/.local/share/agent-harness/trials/gemma-fixes-dev-15yxnrks")
PYTHON = Path("/Users/jorge/Projects/agent-harness/.venv/bin/python")
RUNTIME = "cf52e2e723a93c570947bdb114d22dd2799d03b5"
MANIFEST_SHA = "8600423984f80fbd14ecaf74961a33fb51f409df24fdb67ebfb53c5b1631ef02"
SPEC_PATH = "evals/heldout_comparison/development-freeze-fixes-20261102.json"
GIT = "/opt/homebrew/bin/git"


def check_candidate() -> ExecutionManifest:
    check_existing_interpreter(PYTHON)
    if RUNTIME_ROOT != ROOT / "runtime" or _harness_revision() != {
        "commit": RUNTIME,
        "dirty": False,
    }:
        raise ValueError("Wrong retained source/runtime")
    path = ROOT / "candidate/manifest-candidate.json"
    refuse_withdrawn_selection(path)
    manifest = ExecutionManifest.model_validate_json(_selected_bytes(path, MANIFEST_SHA, 64_000))
    if manifest.runtime_commit != RUNTIME:
        raise ValueError("Wrong candidate runtime")
    preflight(manifest)
    raw, mode = _read_regular(ROOT / "g0/result.json", 64_000)
    gate = json.loads(raw)
    if mode != 0o600 or gate != {
        "runtime_commit": RUNTIME,
        "python": str(PYTHON),
        "pytest_passed": 2825,
        "pytest_failed": 0,
        "live_deselected": 5,
        "ruff": True,
        "format": True,
        "mypy": True,
        "coding_matrix": 90,
        "support_matrix": 90,
        "source_clean": True,
        "source_local_venv_absent": True,
    }:
        raise ValueError("Retained gate evidence is missing or not passing")
    _free_port()  # No request to an unknown listener; no reservation/ownership lease.
    return manifest


def git(*args: str) -> str:
    result = subprocess.run(
        [GIT, *args], cwd=RUNTIME_ROOT, capture_output=True, text=True, timeout=10, check=True
    )
    if len(result.stdout) > 64_000:
        raise ValueError("Git observation exceeds bound")
    return result.stdout.strip()


def verify_freeze(approval: ParentApproval, reviewed: str, merged: str) -> None:
    if any(not re.fullmatch("[0-9a-f]{40}", value) for value in (reviewed, merged)):
        raise ValueError("Select exact reviewed and merged commits")
    if (
        approval.manifest_sha256 != MANIFEST_SHA
        or approval.runtime_commit != RUNTIME
        or approval.freeze_reviewed_head != reviewed
        or approval.freeze_merge_commit != merged
    ):
        raise ValueError("Parent receipt differs from selected freeze")
    parents = git("show", "-s", "--format=%P", merged).split()
    if len(parents) != 2 or parents[1] != reviewed:
        raise ValueError("Freeze merge does not contain the exact reviewed head")
    git("merge-base", "--is-ancestor", RUNTIME, merged)
    name = f"{reviewed}:{SPEC_PATH}"
    if not 1 <= int(git("cat-file", "-s", name)) <= 64_000:
        raise ValueError("Freeze specification exceeds bound")
    spec = json.loads(git("show", name))
    own, _ = _read_regular(Path(__file__), 64_000)
    gate, _ = _read_regular(ROOT / "g0/result.json", 64_000)
    if (
        spec["manifest_sha256"] != MANIFEST_SHA
        or spec["runtime_commit"] != RUNTIME
        or spec["operator_recipe_sha256"] != hashlib.sha256(own).hexdigest()
        or spec["g0_sha256"] != hashlib.sha256(gate).hexdigest()
    ):
        raise ValueError("Recipe, gate or candidate differs from reviewed freeze")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-path", type=Path)
    parser.add_argument("--approval-sha256")
    parser.add_argument("--freeze-reviewed-head")
    parser.add_argument("--freeze-merge-commit")
    args = parser.parse_args()
    check_candidate()
    if args.check_only:
        print("Read-only candidate/gates match; no approval, provider or generation.")
        return
    if not all(
        (
            args.approval_path,
            args.approval_sha256,
            args.freeze_reviewed_head,
            args.freeze_merge_commit,
        )
    ):
        parser.error(
            "Execution requires the actual parent-selected receipt/hash and freeze commits"
        )

    # This command MUST come from the actual authorized parent after external
    # review/CI/merge/user-authority verification. Local Git/hash consistency is
    # NOT authentication of an issuer, review result or the source of arguments.
    def parent(manifest: ExecutionManifest, approval: ParentApproval) -> None:
        verify_freeze(approval, args.freeze_reviewed_head, args.freeze_merge_commit)

    execute_selected(
        ROOT / "candidate/manifest-candidate.json",
        manifest_sha256=MANIFEST_SHA,
        approval_path=args.approval_path,
        approval_sha256=args.approval_sha256,
        verify_parent=parent,
    )


if __name__ == "__main__":
    main()
