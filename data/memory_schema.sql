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
    updated_at     TEXT,
    UNIQUE(user_id, fact)
);

CREATE INDEX IF NOT EXISTS idx_memory_facts_user ON memory_facts(user_id);

-- Keep the legacy global table intact, including any operator schema additions.
CREATE TABLE IF NOT EXISTS workspace_memory_facts (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    workspace_root TEXT NOT NULL,
    fact TEXT NOT NULL,
    source_turn_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    UNIQUE(user_id, workspace_root, fact)
);

CREATE INDEX IF NOT EXISTS idx_workspace_memory_owner
ON workspace_memory_facts(user_id, workspace_root);

-- One monotonic ID space across both tables, including deleted notes.
CREATE TABLE IF NOT EXISTS memory_id_counter (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    last_id INTEGER NOT NULL
);
INSERT OR IGNORE INTO memory_id_counter VALUES (1, 0);
