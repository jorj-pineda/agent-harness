-- Long-term personalization memory for the harness.
-- A separate DB file from the support schema: different access pattern
-- (read-write for the agent) and different trust boundary.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS memory_facts (
    id             INTEGER PRIMARY KEY,
    user_id        TEXT NOT NULL,
    fact           TEXT NOT NULL,
    source_turn_id TEXT,
    created_at     TEXT NOT NULL,
    UNIQUE(user_id, fact)
);

CREATE INDEX IF NOT EXISTS idx_memory_facts_user ON memory_facts(user_id);
