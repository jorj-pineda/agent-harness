"""Small response enums kept separate from outcome harvesting to avoid import cycles."""

from typing import Literal

VerificationStatus = Literal["not_run", "passed", "failed", "stale"]
CompletionStatus = Literal["completed", "incomplete", "blocked", "budget_exhausted", "cancelled"]
