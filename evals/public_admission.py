"""Public-pilot identity guards for the shared evaluator, not launch authorization."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from evals.heldout_comparison import ProtocolModel
from evals.public_envelope import PLAN_SHA256, policy_settings_values
from evals.public_workspaces import _selection
from evals.real_run import Task
from harness.config import Settings


class PublicCellWorkspace(ProtocolModel):
    ordinal: int = Field(ge=1, le=6)
    directory: Path


class PublicPilotDriftError(ValueError):
    """Observed public selection drift invalidates the attempt and latches closed."""


class PublicTaskGuard:
    def __init__(
        self, task: Task, cell: PublicCellWorkspace, *, mode: str, attempt: int, settings: Settings
    ) -> None:
        self.task = task
        self.cell = cell
        self.mode = mode
        self.attempt = attempt
        self.settings = settings
        self.failed = False
        self.failed_checkpoint: str | None = None
        self.checkpoints: list[str] = []
        self.expected_settings = policy_settings_values()
        self.check_at("initial")

    def check_at(self, stage: str) -> None:
        if self.failed:
            raise PublicPilotDriftError("Public pilot preservation previously failed")
        try:
            receipt, task, _ = _selection(self.cell.ordinal)
            if (
                self.task != task
                or self.mode != "harness"
                or type(self.attempt) is not int
                or self.attempt != receipt.repeat
            ):
                raise ValueError("Public task, mode or repeat differs from selected cell")
            actual = self.settings.model_dump()
            if any(actual[key] != value for key, value in self.expected_settings.items()):
                raise ValueError("Public runtime policy differs from selected pilot")
        except Exception as error:
            self.failed = True
            self.failed_checkpoint = stage
            raise PublicPilotDriftError("Public pilot selection or source changed") from error
        self.checkpoints.append(stage)

    def __call__(self) -> None:
        self.check_at("before_request")

    def evidence(self) -> dict[str, object]:
        return {
            "format": "public-pilot-preservation-v1",
            "plan_sha256": PLAN_SHA256,
            "ordinal": self.cell.ordinal,
            "status": "failed" if self.failed else "checks_passed",
            "checkpoints": list(self.checkpoints),
            "failed_checkpoint": self.failed_checkpoint,
        }
