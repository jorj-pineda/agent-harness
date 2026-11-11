"""Fresh owned feedback rehearsal with mandatory in-memory generation responses."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from itertools import islice
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field

from evals.heldout_execution_manifest import Commit, Digest, FrozenModel
from evals.heldout_guard import _read_regular
from evals.heldout_inputs import _save
from evals.owned_inference_transport import ENDPOINT
from evals.public_feedback import FeedbackCell, validate_feedback_plan
from evals.public_feedback_execution import run_feedback_cohort
from evals.public_feedback_execution_admission import admit_feedback_execution
from evals.public_feedback_execution_manifest import (
    FeedbackExecutionManifest,
    FeedbackParentApproval,
)
from evals.public_feedback_preparation import prepare_feedback_candidate
from evals.real_run import load_tasks


class RehearsalCellEvidence(FrozenModel):
    cell: FeedbackCell
    session_sha256: Digest
    cleanup_sha256: Digest
    native_files: dict[str, Digest] = Field(min_length=9, max_length=9)


class OwnedFeedbackRehearsal(FrozenModel):
    format: Literal["public-feedback-owned-mock-rehearsal-v1"] = (
        "public-feedback-owned-mock-rehearsal-v1"
    )
    runtime_commit: Commit
    manifest_sha256: Digest
    preparation_sha256: Digest
    report_sha256: Digest
    cells: list[RehearsalCellEvidence] = Field(min_length=8, max_length=8)
    mock_generation_requests: Literal[24] = 24
    generation_forwarded: Literal[0] = 0
    parent_approval_issued: Literal[False] = False
    inference_ran: Literal[False] = False
    evidence_label: Literal["scripted-real-tools-rehearsal"] = "scripted-real-tools-rehearsal"


class _ReferenceTransport(httpx.MockTransport):
    """Three fixed native responses; no delegate, live option or network client."""

    def __init__(self, model: str, reference: str) -> None:
        self.model, self.reference = model, reference
        self.requests = 0
        self.closed = False
        super().__init__(self.reply)

    def reply(self, request: httpx.Request) -> httpx.Response:
        if self.closed or request.method != "POST" or request.url != ENDPOINT or self.requests >= 3:
            raise ValueError("Unexpected owned rehearsal mock request")
        payload = json.loads(request.content)
        names = {tool["function"]["name"] for tool in payload["tools"]}
        if not {"write_file", "run_project_check"} <= names:
            raise ValueError("Owned rehearsal tools differ from selected policy")
        self.requests += 1
        if self.requests == 1:
            calls = [("write_file", {"path": "pagination.py", "content": self.reference})]
        elif self.requests == 2:
            calls = [("run_project_check", {})]
        else:
            if json.loads(payload["messages"][-1]["content"])["exit_code"] != 0:
                raise ValueError("Owned rehearsal reference check did not pass")
            calls = []
        return httpx.Response(
            200,
            json={
                "model": self.model,
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 5,
                "message": {
                    "role": "assistant",
                    "content": "Scripted reference completion" if not calls else "",
                    "tool_calls": [
                        {"function": {"name": name, "arguments": arguments}}
                        for name, arguments in calls
                    ],
                },
            },
        )

    async def aclose(self) -> None:
        self.closed = True
        await super().aclose()


def _private_bytes(path: Path) -> bytes:
    raw, mode = _read_regular(path, 2_000_000)
    if mode != 0o600:
        raise ValueError("Owned rehearsal evidence must remain private")
    return raw


def _digest(path: Path) -> str:
    return hashlib.sha256(_private_bytes(path)).hexdigest()


def rehearse_feedback_execution(
    directory: Path,
    *,
    expected_python: Path,
    expected_commit: str,
    executables: dict[str, Path],
    dependency_files: list[Path],
    source_models: Path,
    command_env: Mapping[str, str],
) -> OwnedFeedbackRehearsal:
    """Create and consume only a new rehearsal bundle, never an existing candidate.

    Preparation is exclusive. Parent assertions below are a simulation override,
    bound to this new consumed bundle and zero freeze IDs, never actual authority.
    Metadata uses real owned backends; generation always uses _ReferenceTransport.
    There is no transport/provider/approval injection or live mode in this API.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise ValueError("Enter owned feedback rehearsal outside an active event loop")
    manifest = prepare_feedback_candidate(
        directory,
        expected_python=expected_python,
        expected_commit=expected_commit,
        executables=executables,
        dependency_files=dependency_files,
        source_models=source_models,
        command_env=command_env,
    )
    try:
        manifest_path = directory / "manifest-candidate.json"
        manifest_sha = _digest(manifest_path)
        fixture = FeedbackParentApproval(
            format="public-feedback-parent-approval-v1",
            approval_id="mandatory-mock-owned-rehearsal-only-NOT-actual-approval",
            manifest_sha256=manifest_sha,
            runtime_commit=manifest.runtime_commit,
            freeze_reviewed_head="0" * 40,
            freeze_merge_commit="0" * 40,
            exact_head_review_passed=True,
            ci_passed=True,
            merged=True,
            inference_authorized=True,
        )
        fixture_path = directory / "mock-parent-fixture.json"
        fixture_sha = _save(fixture_path, fixture.model_dump_json(indent=2).encode())

        def mock_parent(
            selected: FeedbackExecutionManifest, approval: FeedbackParentApproval
        ) -> None:
            # Explicit scripted provenance fixture, not a verifier of actual authority.
            if selected != manifest or approval != fixture:
                raise ValueError("Owned rehearsal fixture selection changed")

        admission = admit_feedback_execution(
            manifest_path,
            manifest_sha256=manifest_sha,
            approval_path=fixture_path,
            approval_sha256=fixture_sha,
            verify_parent=mock_parent,
        )
        plan = validate_feedback_plan()
        task = next(task for task in load_tasks() if task.id == plan.task.id)
        reference, _ = _read_regular(task.reference / "pagination.py", 64_000)
        transports: list[_ReferenceTransport] = []

        def mandatory_mock(ordinal: int, output: Path) -> httpx.AsyncBaseTransport:
            transport = _ReferenceTransport(plan.model, reference.decode("utf-8"))
            transports.append(transport)
            return transport

        ledger = run_feedback_cohort(admission, transport_factory=mandatory_mock, scripted=True)
        if (
            ledger.status != "finished"
            or len(transports) != 8
            or any(transport.requests != 3 or not transport.closed for transport in transports)
        ):
            raise ValueError("Owned rehearsal schedule or mock transport did not finish")
        output = Path(manifest.output_directory)
        report = json.loads(_private_bytes(output / "report.json"))
        if (
            report["mode"] != "scripted-real-tools-rehearsal"
            or len(report["results"]) != 8
            or report["cohort"] != ledger.model_dump(mode="json")
            or report["feedback_comparison"]["benefit_assessment"] != "not_assessed"
            or any(
                not row["passed"] or row["completion_status"] != "completed"
                for row in report["results"]
            )
        ):
            raise ValueError("Owned rehearsal reference results differ from its schedule")
        cells = []
        for cell in plan.cells:
            cell_output = output / f"cell-{cell.ordinal:02d}"
            cleanup_path = cell_output / "backend/cleanup.json"
            session_path = cell_output / "backend/session.json"
            cleanup = json.loads(_private_bytes(cleanup_path))
            session = json.loads(_private_bytes(session_path))
            if (
                cleanup.get("passed") is not True
                or cleanup.get("settled") is not True
                or cleanup.get("forced") is not False
                or cleanup.get("errors") != {}
                or session.get("format") != "heldout-owned-backend-session-v1"
                or session.get("inference_authorized") is not False
                or session.get("cleanup") != cleanup
            ):
                raise ValueError("Owned rehearsal settlement evidence did not pass")
            names = {
                f"request-{n:02d}.{suffix}"
                for n in range(1, 4)
                for suffix in ("request.bin", "response.bin", "json")
            }
            native = cell_output / "native"
            if {path.name for path in islice(native.iterdir(), 10)} != names:
                raise ValueError("Owned rehearsal native evidence inventory changed")
            cells.append(
                RehearsalCellEvidence(
                    cell=cell,
                    session_sha256=_digest(session_path),
                    cleanup_sha256=_digest(cleanup_path),
                    native_files={name: _digest(native / name) for name in sorted(names)},
                )
            )
        summary = OwnedFeedbackRehearsal(
            runtime_commit=manifest.runtime_commit,
            manifest_sha256=manifest_sha,
            preparation_sha256=_digest(directory / "preparation.json"),
            report_sha256=_digest(output / "report.json"),
            cells=cells,
            mock_generation_requests=sum(transport.requests for transport in transports),
        )
        _save(directory / "owned-rehearsal.json", summary.model_dump_json(indent=2).encode())
        return summary
    except BaseException as error:
        try:
            _save(directory / "WITHDRAWN.json", b'{"eligible_for_live_launch":false}')
            _save(
                directory / "owned-rehearsal-failure.json",
                json.dumps(
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "inference_authorized": False,
                    }
                ).encode(),
            )
        except BaseException as recording_error:
            error.add_note(
                f"Owned rehearsal failure recording also failed: {type(recording_error).__name__}"
            )
        raise
