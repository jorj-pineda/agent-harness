# agent-harness — Composer Super Prompts

Use this file to start a **new Composer chat per pivot phase**. Copy **one** phase block (from `## Paste from here` through `## Paste to here`) into the chat input.

**Roadmap detail:** [CODING_AGENT_PIVOT.md](CODING_AGENT_PIVOT.md)  
**Demo ops:** [demo.md](demo.md)

---

## Quick recap (for you, not the agent)

| Item | Status |
|------|--------|
| Support harness (steps 1–12) | Merged |
| Demo hardening (`feat/demo-readiness`) | Done — pytest/ruff/mypy, `demo.md`, eval `--live` |
| Current direction | Pivot → **senior-level coding agent** (phases 0–10 below) |
| Workflow | Agent completes **one phase**, pauses for your green light before the next |

**Which prompt to use**

| Phase | When |
|-------|------|
| **0** | Tag/merge support baseline before pivot |
| **1** | Rebrand README + API shape (no new tools yet) |
| **2** | Workspace jail + read-only code tools |
| **3** | Grounding → file:line citations |
| **4** | Write tools + verification loop |
| **5** | Repo memory (FactStore repurposed) |
| **6** | Coding eval harness |
| **7** | Indexing strategy (grep vs embed) |
| **8** | Docker + demo for coding |
| **9** | Senior polish (scope gate, edit budget, etc.) |
| **10** | Final gate |

---

## Shared context (included in every phase prompt)

Every phase block below repeats these rules so each chat is self-contained.

**Repo:** `/Users/jorge/Projects/agent-harness`

**Target product:** Local-first, pluggable-provider **senior coding agent** harness — hand-written ReAct loop (no LangChain/LlamaIndex/LangGraph). Portfolio piece.

**Headline features (post-pivot):**
1. **Grounded edits with confidence** — citations = file paths + line ranges; `escalated=True` when evidence is thin.
2. **Cross-session repo memory** — SQLite `FactStore` per `user_id`; conventions and prior decisions injected into system prompt.

**Response envelope (preserve; extend additively):**  
`{answer, confidence, citations, escalated, tool_calls, memory_writes, provider, latency_ms}`

**Architecture (layers depend only downward):**

```
api/            FastAPI — thin HTTP wrapper, per-request tool registry
  └── harness/  ReAct loop, session/turn state, provider router
        ├── grounding/   confidence heuristic, citations, escalation
        ├── memory/      FactStore, system-prompt injection
        ├── tools/       typed registry (coding tools replace support tools over phases)
        ├── workspace/   (Phase 2+) sandboxed repo root
        └── providers/   Ollama / Anthropic / OpenAI — SACRED abstraction
```

**Critical rules (non-negotiable):**
1. NO agent frameworks.
2. Nothing above `providers/` may import a specific provider SDK.
3. Every response ships the full metadata envelope.
4. Tests use VCR cassettes / FakeProvider — never hit live APIs in CI.
5. Secrets in `.env` via pydantic-settings; `.env.example` committed.
6. `pyproject.toml` edits: additive only, phase-labeled blocks.
7. Line length 100, `ruff format/check`, `mypy --strict` on core layers.
8. No comments that restate code; commits imperative present tense.
9. Do NOT commit unless Jorge explicitly says "commit" — DO propose commit messages each phase.
10. Workspace tools must jail paths; shell tools use argv allowlists, never `shell=True`.

**Baseline commands:**

```bash
uv sync --extra dev
pytest -m "not live"
ruff check .
mypy
python -m evals.run --providers ollama,anthropic,openai
```

**Phase workflow (MANDATORY):** Complete the assigned phase only. Then output:
1. **What I did** (3–6 bullets)
2. **Evidence** (test/lint counts)
3. **Files touched**
4. **Proposed commit message** (imperative, 1–2 sentences, WHY)
5. **Stop** — ask Jorge for green light before the next phase.

---

# Phase 0 — Freeze support baseline

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Phase 0 only:** Freeze the customer-support baseline before the coding-agent pivot. Do not start pivot code yet.

### Context

Support harness steps 1–12 are merged. Demo hardening landed on `feat/demo-readiness`: mypy fix, `demo.md`, eval `--live`, README/CLAUDE sync, `CODING_AGENT_PIVOT.md`.

### Your tasks

1. Confirm `main` (or intended base branch) includes demo-readiness work — read git log, run baseline commands.
2. Ensure support demo is recoverable:
   - Tag `v0.1-support` **or** document the merge commit hash in `CODING_AGENT_PIVOT.md` Phase 0 section.
3. Verify offline eval still matches README headline table.
4. List any uncommitted files; propose what to merge vs gitignore.

### Exit criteria

- Support baseline tagged or merge commit recorded.
- `pytest -m "not live"`, `ruff`, `mypy` green.
- No pivot code changes in this phase.

### Rules

Follow shared context above. Minimal diffs. Do not commit unless Jorge says "commit".

**Start Phase 0 now.** Report status, propose tag/merge steps, then STOP for green light.

## Paste to here

---

# Phase 1 — Narrative + API shape

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Phase 1 only:** Rebrand toward a senior coding agent **without** removing support tools yet. Read [CODING_AGENT_PIVOT.md](CODING_AGENT_PIVOT.md) Phase 1.

**Prerequisite:** Phase 0 complete (support baseline frozen).

### Your tasks

1. **README** — Rewrite headline features for coding agent:
   - Grounded edits with confidence (file:line citations).
   - Cross-session repo memory.
   - Keep architecture diagram accurate; note pivot in progress if support tools still present.
2. **`api/models.py` + `api/server.py`** — Extend requests:
   - Add `workspace_root` or `workspace_id` on session/chat (design: resolve to sandbox path).
   - Keep `user_id` for memory scoping.
3. **System prompt** — Replace `BASE_SYSTEM_PROMPT` in `api/server.py` with engineering contract: minimal diffs, cite files read, run verification before claiming done.
4. **Envelope (additive only)** — If needed, extend `TurnResponse` / API models with optional `files_touched`, `verification_ran` — do not remove existing fields evals use.

### Allowed file scope

`README.md`, `api/`, `harness/state.py` (only if envelope extended), `tests/api/`. Do not delete `tools/sql.py` or `tools/rag.py` yet.

### Exit criteria

- API boots; existing tests pass (update API tests for new optional fields).
- README describes coding agent intent.
- Support tools still registered — behavior may be unchanged for old curl demos.

### Rules

Follow shared context. Read before editing. Minimal diffs. Propose commit message; STOP for green light.

**Start Phase 1 now.**

## Paste to here

---

# Phase 2 — Workspace sandbox + read-only code tools

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Phase 2 only:** Workspace jail + read-only coding tools. Read [CODING_AGENT_PIVOT.md](CODING_AGENT_PIVOT.md) Phase 2.

**Prerequisite:** Phase 1 complete (API accepts workspace root).

### Your tasks

1. **`workspace/` module** (new package, add to `pyproject.toml` hatch packages):
   - `Workspace` dataclass: root path, ignore globs (`.git`, `node_modules`, `.venv`).
   - `resolve(path) -> Path` with jail — reject paths outside root.
2. **Read-only tools** in `tools/`:
   - `read_file(path, start_line?, end_line?)` — byte/line caps.
   - `grep_repo(pattern, path?, glob?)` — subprocess ripgrep or pure Python; timeout + hit limit.
   - `list_dir(path)` / optional `tree(path, depth)` — bounded listing.
   - `git_status`, `git_diff` — read-only git via argv allowlist.
3. **Registry** — `register_code_tools(registry, workspace=...)` called from `api/server.py` per request (bind workspace like memory tools bind `user_id`).
4. **Fixture repo** — `tests/fixtures/tiny_repo/` (minimal Python package + one module + test).
5. **Tests** — path escape attempts fail; happy-path read/grep on fixture.

### Allowed file scope

`workspace/`, `tools/`, `api/server.py`, `tests/`, `pyproject.toml` (additive), `tests/fixtures/`.

### Exit criteria

- Agent can explore fixture repo via new tools in integration test.
- SQL/RAG tools may remain registered alongside (no deletion required).

### Rules

Follow shared context. No `shell=True`. Log tool calls. STOP for green light after phase.

**Start Phase 2 now.**

## Paste to here

---

# Phase 3 — Grounding retarget (code citations)

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Phase 3 only:** Retarget grounding from RAG chunks to file:line citations. Read [CODING_AGENT_PIVOT.md](CODING_AGENT_PIVOT.md) Phase 3.

**Prerequisite:** Phase 2 complete (read tools emit structured results).

### Your tasks

1. **`harness/grounding.py`** — Accept evidence from `read_file` / `grep_repo` tool records (path, start_line, end_line, optional snippet).
2. **Heuristic** — Keep shape `top_score × coverage × health`; tune for code evidence not Chroma distances.
3. **Citation schema** — `{path, start_line, end_line, snippet?}` in grounding output; map to `TurnResponse.citations`.
4. **Tests** — Extend `tests/test_harness_grounding.py` with code-style tool call fixtures; keep or isolate legacy RAG scenarios.
5. **Do not delete RAG grounder path yet** if support evals still depend on it — branch on tool name or feature flag.

### Allowed file scope

`harness/grounding.py`, `harness/loop.py` (minimal), `harness/state.py`, `tests/test_harness_grounding.py`, related types.

### Exit criteria

- Grounding produces file:line citations from code tool traces.
- Escalation still fires on low-confidence coding scenarios in tests.

### Rules

Follow shared context. Minimal diffs. STOP for green light.

**Start Phase 3 now.**

## Paste to here

---

# Phase 4 — Write tools + verification loop

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Phase 4 only:** Write tools + allowlisted verification commands. Read [CODING_AGENT_PIVOT.md](CODING_AGENT_PIVOT.md) Phase 4.

**Prerequisite:** Phase 2–3 complete (read tools + code grounding).

### Your tasks

1. **`apply_patch` or `write_file`** — edits jailed to workspace; reject traversal.
2. **`run_command`** — allowlist only (`pytest`, `ruff`, `mypy`, `git diff`, etc.); timeout; `subprocess` with argument list, **never** `shell=True`.
3. **Integration test** — fixture repo with intentionally failing test; scripted or live FakeProvider path proves read → edit → pytest → answer.
4. **Optional setting** — `REQUIRE_VERIFICATION_BEFORE_FINISH` (default false) documented in `.env.example`.

### Allowed file scope

`tools/`, `workspace/`, `api/settings.py`, `tests/fixtures/`, `tests/test_tools_*`.

### Exit criteria

- End-to-end bugfix on `tests/fixtures/tiny_repo` in tests.
- Tool trace + verification visible in `TurnResponse`.

### Rules

Follow shared context. Security-first on shell. STOP for green light.

**Start Phase 4 now.**

## Paste to here

---

# Phase 5 — Memory for engineering context

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Phase 5 only:** Repurpose FactStore for repo/engineering memory. Read [CODING_AGENT_PIVOT.md](CODING_AGENT_PIVOT.md) Phase 5.

**Prerequisite:** Phase 1 complete (coding system prompt). Phases 2–4 recommended.

### Your tasks

1. **Document** facts as durable engineering notes (conventions, stack choices, review preferences).
2. **Tools** — keep `remember_fact` / `recall_facts` or add aliases `remember` / `recall`; update descriptions for coding context.
3. **System prompt** — clarify memory tool usage for cross-session repo preferences.
4. **Tests** — session A remembers "prefer async"; session B recall surfaces in system prompt or tool result.

### Allowed file scope

`memory/`, `tools/memory.py`, `api/server.py` (prompt + registry), `tests/test_tools_memory.py`, `tests/test_memory_store.py`.

### Exit criteria

- Cross-session memory test passes with engineering-style facts.
- No regression to memory isolation per `user_id`.

### Rules

Follow shared context. STOP for green light.

**Start Phase 5 now.**

## Paste to here

---

# Phase 6 — Eval harness for coding

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Phase 6 only:** Replace support eval scenarios with coding scenarios. Read [CODING_AGENT_PIVOT.md](CODING_AGENT_PIVOT.md) Phase 6.

**Prerequisite:** Phase 4 complete (at least one end-to-end fixture bugfix works).

### Your tasks

1. **`evals/scenarios.yaml`** — ~25–30 scenarios, categories:
   - `bugfix`, `feature_slice`, `refactor`, `explore_only`, `low_confidence`, `unsafe_request`
2. **`evals/scorers.py`** — add/adapt:
   - `patch_correctness`, `verification_ran`, code `faithfulness` (claims vs cited lines)
   - keep `escalation` scorer
3. **`evals/run.py`** — scripted tool results for fixture repos; keep offline FakeProvider default + `--live`.
4. **README** — new headline table from offline eval run.
5. **Tests** — update `tests/evals/test_run.py` for new schema/counts.

### Allowed file scope

`evals/`, `tests/evals/`, `tests/fixtures/`, `README.md` (eval table only).

### Exit criteria

- `python -m evals.run --providers ollama,anthropic,openai` completes offline.
- CI deterministic; live path documented.

### Rules

Follow shared context. Archive or move old support scenarios to `evals/scenarios_support.yaml` if useful. STOP for green light.

**Start Phase 6 now.**

## Paste to here

---

# Phase 7 — Indexing strategy

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Phase 7 only:** Choose and implement indexing for codebase search. Read [CODING_AGENT_PIVOT.md](CODING_AGENT_PIVOT.md) Phase 7.

**Prerequisite:** Phase 2 complete. Phase 6 recommended (evals tell you if grep is enough).

### Your tasks

1. **Decide** (document in README):
   - **A. Ripgrep only** (recommended v1), **B. Chroma over repo**, or **C. Hybrid**.
2. **Implement chosen path** — if staying grep-only, add optional `semantic_search` stub deferred; if Chroma, wire embedder through `providers/` only.
3. **Deprecate support data path** — archive `data/seed.py`, support corpus, SQL tools from default registry (feature flag or remove if Jorge approves).
4. **Update evals** if indexing changes explore_only scores.

### Allowed file scope

`tools/`, `data/`, `api/server.py`, `README.md`, `CODING_AGENT_PIVOT.md` (decision record).

### Exit criteria

- Decision documented with trade-offs.
- Default coding demo does not require support SQLite/doc corpus.

### Rules

Follow shared context. Do not add LangChain. STOP for green light.

**Start Phase 7 now.**

## Paste to here

---

# Phase 8 — Docker + demo for coding

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Phase 8 only:** Docker and demo docs for coding agent. Read [CODING_AGENT_PIVOT.md](CODING_AGENT_PIVOT.md) Phase 8 and existing [demo.md](demo.md).

**Prerequisite:** Phases 1–4 minimum; Phase 6 for eval demo commands.

### Your tasks

1. **`docker-compose.yml`** — mount fixture repo or `fixtures/` volume; env for `WORKSPACE_ROOT`.
2. **`Dockerfile`** — copy `workspace/`, `fixtures/` as needed.
3. **`demo.md`** — coding walkthrough:
   - create session with workspace
   - ask to fix failing test
   - interpret envelope (citations, confidence, escalated)
   - model options (Ollama vs cloud; RAM notes from support demo)
4. **Smoke test** — `docker compose build && up`; document commands even if gemma4 OOM on low-RAM Docker Desktop.

### Allowed file scope

`docker-compose.yml`, `Dockerfile`, `.dockerignore`, `demo.md`, `README.md` (Run it section).

### Exit criteria

- Reviewer can follow `demo.md` for coding demo in <15 minutes on a adequately provisioned machine.

### Rules

Follow shared context. STOP for green light.

**Start Phase 8 now.**

## Paste to here

---

# Phase 9 — Senior-level polish

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Phase 9 only:** Pick 2–3 senior differentiators. Read [CODING_AGENT_PIVOT.md](CODING_AGENT_PIVOT.md) Phase 9.

**Prerequisite:** Phases 1–6 complete.

### Options (pick 2–3 with Jorge if unclear)

1. **Scope gate** — classify task (bugfix / explore / out-of-scope); refuse huge rewrites.
2. **Edit budget** — max files/lines per turn; set `escalated` when exceeded.
3. **`emit_plan` tool** — structured plan in tool trace before edits.
4. **Diff-first API presentation** — enrich envelope with patch summary fields.
5. **Eval honesty** — README offline vs `--live` limits for coding metrics.

### Your tasks

1. Implement chosen items with tests.
2. Update README "What's novel" for each.
3. Add eval scenarios if scope gate / edit budget affect escalation.

### Exit criteria

- Each chosen feature has test coverage and README mention.
- No scope creep into IDE/LSP/full SWE-bench.

### Rules

Follow shared context. STOP for green light.

**Start Phase 9 now.** If Jorge didn't specify which 2–3 items, propose a recommendation first and wait.

## Paste to here

---

# Phase 10 — Final gate

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Phase 10 only:** Final quality gate for coding-agent "almost ready." Read [CODING_AGENT_PIVOT.md](CODING_AGENT_PIVOT.md) Phase 10.

**Prerequisite:** Phases 1–9 complete (or Jorge explicitly waives 9).

### Your tasks

1. Run full gate:
   ```bash
   pytest -m "not live"
   ruff check .
   mypy
   python -m evals.run --providers ollama,anthropic,openai
   ```
2. Regenerate eval report; confirm README headline table matches.
3. Verify `demo.md` coding path once (document any machine constraints).
4. Update `CLAUDE.md` / README deferred list.
5. Produce **reviewer checklist** (3–5 commands a hiring manager can run).
6. List explicit non-goals still deferred.

### Exit criteria

- All checks green.
- Reviewer checklist delivered.
- No uncommitted critical work unmentioned.

### Rules

Fix only blockers found in gate — minimal diffs. Propose final merge PR title/description. STOP after deliverable.

**Start Phase 10 now.**

## Paste to here

---

# Appendix — Support demo hardening (COMPLETE)

The 7-slice support hardening mission is **done** (May 2026). Do not re-run unless regressions appear. Summary:

| Slice | Outcome |
|-------|---------|
| 1 | Baseline verified |
| 2 | Chroma/mypy fix in `tools/rag.py` |
| 3 | Docker smoke + `demo.md` |
| 4 | Eval regenerated; README table matched |
| 5 | Eval `--live` + FakeProvider docs |
| 6 | README + CLAUDE sync |
| 7 | Final gate green |

For support-only debugging, use [demo.md](demo.md) and `python -m evals.run --providers ollama,anthropic,openai`.
