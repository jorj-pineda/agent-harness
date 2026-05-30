"""Long-term engineering memory for the harness.

SQLite-backed notes keyed by `user_id` — conventions, stack choices, review
preferences, and other durable context that should survive across sessions.
The agent writes via `remember_fact` / `remember` and reads via `recall_facts`
/ `recall`; the API layer injects notes into the system prompt each turn.
"""

from .store import DEFAULT_LIST_LIMIT, FACTS_HEADING, Fact, FactStore

__all__ = ["DEFAULT_LIST_LIMIT", "FACTS_HEADING", "Fact", "FactStore"]
