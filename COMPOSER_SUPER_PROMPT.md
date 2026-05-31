# agent-harness — Composer Super Prompts (post-pivot)

Use this file to start a **new Composer chat per mission**. Copy **one** mission block (from `## Paste from here` through `## Paste to here`) into the chat input.

**Roadmap:** [NEXT_STEPS.md](NEXT_STEPS.md)  
**Project rules:** [CLAUDE.md](CLAUDE.md)  
**Pivot history:** [CODING_AGENT_PIVOT.md](CODING_AGENT_PIVOT.md)  
**Demo ops:** [demo.md](demo.md)

---

## Quick recap (for you, not the agent)

| Item | Status |
|------|--------|
| Support harness (steps 1–12) | Merged |
| Demo hardening (`feat/demo-readiness`) | Done |
| Coding-agent pivot (phases 1–10) | **Merged to `main` (PR #13)** |
| Current direction | Post-pivot backlog in [NEXT_STEPS.md](NEXT_STEPS.md) |
| Workflow | Agent completes **one mission**, pauses for your green light before the next |

**Which prompt to use**

| Mission | When |
|---------|------|
| **0 — Context reload** | New chat; orient before picking work (read-only recon) |
| **1 — Ship + validate** | Tag release, reviewer checklist, Docker/live smoke |
| **2 — Live eval snapshot** | Run `--live`, document scores, optional README row |
| **3 — GitHub Actions CI** | Wire pytest/ruff/mypy/evals in CI |
| **4 — `emit_plan` tool** | Structured plan in tool trace before edits |
| **5 — Patch summary envelope** | `patch_summary` field from write results |
| **6 — Stronger scope gate** | Classify task type; refuse unbounded rewrites |
| **7 — README / portfolio polish** | FocusKPI write-up pass (minimal code) |
| **8 — Semantic search** | Only if grep-only explore evals fail |

Pivot phase prompts (0–10) are archived in [CODING_AGENT_PIVOT.md](CODING_AGENT_PIVOT.md).

---

## Shared context (included in every mission prompt)

Every mission block below repeats these rules so each chat is self-contained.

**Repo:** `/Users/jorge/Projects/agent-harness`  
**Branch:** `main` (pivot merged 2026-05, PR #13)

**Target product:** Local-first, pluggable-provider **senior coding agent** harness — hand-written ReAct loop (no LangChain/LlamaIndex/LangGraph). Portfolio piece for FocusKPI Junior AI/ML Engineer role.

**Headline features (shipped):**
1. **Grounded edits with confidence** — citations = file paths + line ranges; `escalated=True` when evidence is thin.
2. **Cross-session repo memory** — SQLite `FactStore` per `user_id`; engineering notes injected into system prompt.
3. **Workspace sandbox** — path-jailed code tools; ripgrep-first search; scope gate + edit budget.

**Response envelope (preserve; extend additively only):**  
`{answer, confidence, citations, escalated, tool_calls, memory_writes, files_touched, verification_ran, provider, latency_ms}`

**Architecture (layers depend only downward):**

```
api/            FastAPI — thin HTTP wrapper, per-request tool registry
  └── harness/  ReAct loop, session/turn state, provider router, policy gate
        ├── grounding/   confidence heuristic, file:line citations, escalation
        ├── memory/      FactStore, system-prompt injection
        ├── tools/       read/grep/edit/verify + memory (support tools optional)
        ├── workspace/   sandboxed repo root, path jail
        └── providers/   Ollama / Anthropic / OpenAI — SACRED abstraction
```

**Key defaults:**
- `ENABLE_SUPPORT_TOOLS=false` — coding demo default; support path needs seed/embed
- `DEFAULT_WORKSPACE_ROOT` — Docker: `/app/fixtures/tiny_repo`
- `MAX_FILES_TOUCHED_PER_TURN=5`
- Offline eval: 28 coding scenarios; README headline table; threshold 0.50 (API: 0.55)

**Critical rules (non-negotiable):**
1. NO agent frameworks.
2. Nothing above `providers/` may import a specific provider SDK.
3. Every response ships the full metadata envelope.
4. Tests use VCR cassettes / FakeProvider — never hit live APIs in CI.
5. Secrets in `.env` via pydantic-settings; `.env.example` committed.
6. `pyproject.toml` edits: additive only, step-labeled blocks.
7. Line length 100, `ruff format/check`, `mypy --strict` on core layers.
8. No comments that restate code; commits imperative present tense.
9. Do NOT commit unless Jorge explicitly says "commit" — DO propose commit messages each mission.
10. Workspace tools must jail paths; shell tools use argv allowlists, never `shell=True`.

**Baseline commands:**

```bash
uv sync --extra dev
pytest -m "not live"                    # ~337 tests
ruff check .
mypy
python -m evals.run --providers ollama,anthropic,openai
```

**Mission workflow (MANDATORY):** Complete the assigned mission only. Then output:
1. **What I did** (3–6 bullets)
2. **Evidence** (test/lint counts, command output)
3. **Files touched**
4. **Proposed commit message** (imperative, 1–2 sentences, WHY)
5. **Stop** — ask Jorge for green light before the next mission.

---

# Mission 0 — Context reload (read-only)

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Mission 0 only:** Read-only reconnaissance. Do **not** change code unless Jorge explicitly asks.

### Context

Coding-agent pivot (phases 1–10) is **merged to `main`** (PR #13). Post-pivot work is tracked in [NEXT_STEPS.md](NEXT_STEPS.md). Project rules live in [CLAUDE.md](CLAUDE.md).

### Your tasks

1. Read [NEXT_STEPS.md](NEXT_STEPS.md), [README.md](README.md) (eval table + reviewer checklist), and [CLAUDE.md](CLAUDE.md).
2. Confirm repo state: `git log -5 --oneline`, `git status`.
3. Run baseline gate (report counts, don't fix unless broken):
   ```bash
   uv run pytest -m "not live" -q
   uv run ruff check .
   uv run mypy
   ```
4. Summarize: what's shipped, what's next (Tier A/B/C from NEXT_STEPS), and recommend **one** mission for Jorge to pick next.

### Exit criteria

- Accurate status report (test count, eval scenario count, key env vars).
- No code diffs unless baseline is red and Jorge approves fixes.

### Rules

Follow shared context above. STOP after deliverable — wait for Jorge to pick Mission 1–8.

**Start Mission 0 now.**

## Paste to here

---

# Mission 1 — Ship + validate

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Mission 1 only:** Finish post-merge ship checklist from [NEXT_STEPS.md](NEXT_STEPS.md) §1–3. Minimal code; docs/tags/smoke only.

### Prerequisite

Pivot merged to `main`. Mission 0 optional.

### Your tasks

1. **Tag** — If `v0.2-coding-agent` missing, propose tag on merge commit `acb8cbe` (or current merge tip); do not force-push.
2. **Reviewer checklist** — Run README checklist end-to-end; fix only blockers found.
3. **Docker smoke** — Follow [demo.md](demo.md) coding path; document any machine constraints in demo.md if needed.
4. **Update [NEXT_STEPS.md](NEXT_STEPS.md)** — Mark merge/tag/checklist items done; note live eval as next.
5. Optional: tag `v0.1-support` at pre-pivot commit if not already tagged.

### Allowed file scope

`NEXT_STEPS.md`, `demo.md` (smoke notes only), git tags (with Jorge approval).

### Exit criteria

- Baseline gate green.
- NEXT_STEPS reflects current ship status.
- Docker demo documented (pass or documented constraint).

### Rules

Follow shared context. No feature work. STOP for green light.

**Start Mission 1 now.**

## Paste to here

---

# Mission 2 — Live eval snapshot

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Mission 2 only:** Run live provider evals and document results. Read [NEXT_STEPS.md](NEXT_STEPS.md) Tier A (live eval headline row).

### Prerequisite

Ollama running with `gemma4` pulled, **or** Anthropic/OpenAI keys in `.env`.

### Your tasks

1. Run live eval (start narrow if full matrix is slow):
   ```bash
   python -m evals.run --live --providers ollama
   ```
2. Capture score spread vs offline README table; note which scenarios diverge and why.
3. Add **`evals/LIVE.md`** (or README subsection) documenting:
   - command used, date, provider, scenario subset if partial
   - live vs offline interpretation (scripted vs real model)
4. Optional: add separate **"Live snapshot"** row to README eval table (do not overwrite offline table).

### Allowed file scope

`evals/LIVE.md`, `README.md` (eval section only), `NEXT_STEPS.md` (mark done).

### Exit criteria

- At least one successful `--live` run with documented output.
- Clear eval honesty note for reviewers.

### Rules

Follow shared context. Do not change scorers unless live run exposes a harness bug. STOP for green light.

**Start Mission 2 now.** If no live provider available, report what's missing and STOP.

## Paste to here

---

# Mission 3 — GitHub Actions CI

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Mission 3 only:** Add CI workflow per [NEXT_STEPS.md](NEXT_STEPS.md) Tier C.

### Your tasks

1. Add `.github/workflows/ci.yml`:
   - `uv sync --extra dev`
   - `pytest -m "not live"`
   - `ruff check .`
   - `mypy`
   - `python -m evals.run --providers ollama,anthropic,openai` (offline FakeProvider)
2. Python 3.11 matrix (single version fine for portfolio).
3. Optional second job: support regression with `ENABLE_SUPPORT_TOOLS=true` + `evals/scenarios_support.yaml` (only if seed data can run in CI without Ollama — document if skipped).
4. Update README reviewer checklist to mention CI badge if added.

### Allowed file scope

`.github/workflows/`, `README.md` (CI note), `NEXT_STEPS.md`, maybe `data/` if CI needs fixture DB.

### Exit criteria

- Workflow file valid; local commands match CI steps.
- No live provider calls in CI.

### Rules

Follow shared context. STOP for green light.

**Start Mission 3 now.**

## Paste to here

---

# Mission 4 — `emit_plan` tool

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Mission 4 only:** Add structured planning before edits. Deferred item from pivot Phase 9 / [NEXT_STEPS.md](NEXT_STEPS.md) Tier A.

### Your tasks

1. Add `emit_plan(steps: list[str], ...)` tool — writes structured plan to tool trace (no filesystem side effects).
2. Register in code tools when workspace is bound.
3. Optional setting: `REQUIRE_PLAN_BEFORE_EDIT` (default false) — if true, escalate when agent calls `write_file` without prior `emit_plan` in same turn.
4. Tests: plan appears in `tool_calls`; optional gate test.
5. README "What's novel" bullet + eval scenario if gate enabled.

### Allowed file scope

`tools/code.py`, `api/server.py`, `api/settings.py`, `.env.example`, `tests/`, `README.md`, `evals/` (if new scenario).

### Exit criteria

- Plan tool callable in integration test.
- No regression to existing 337 tests.

### Rules

Follow shared context. Minimal API surface. STOP for green light.

**Start Mission 4 now.**

## Paste to here

---

# Mission 5 — Patch summary in envelope

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Mission 5 only:** Enrich response envelope with edit summary. [NEXT_STEPS.md](NEXT_STEPS.md) Tier A.

### Your tasks

1. Extend `TurnResponse` / API models with optional `patch_summary: list[str]` (or structured `{path, lines_added, lines_removed}` — pick simpler v1).
2. Harvest from `write_file` tool results in [harness/outcome.py](harness/outcome.py) or loop.
3. API tests + one harness test proving field populated after edit turn.
4. Document field in README envelope description.

### Allowed file scope

`harness/state.py`, `harness/outcome.py`, `harness/loop.py`, `api/models.py`, `tests/`.

### Exit criteria

- Additive envelope field only; evals still pass.
- Backward compatible (optional field).

### Rules

Follow shared context. STOP for green light.

**Start Mission 5 now.**

## Paste to here

---

# Mission 6 — Stronger scope gate

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Mission 6 only:** Improve [harness/policy.py](harness/policy.py) task classification. [NEXT_STEPS.md](NEXT_STEPS.md) Tier A.

### Prerequisite

Read current scope gate tests and eval `unsafe_request` scenarios.

### Your tasks

1. Classify incoming message: `bugfix | explore | refactor | out_of_scope` (heuristic or lightweight rules — no second LLM call unless justified).
2. Refuse unbounded requests ("rewrite entire repo", "delete all tests") with `provider="policy"` early return (existing pattern).
3. Allow normal bugfix/explore/refactor through.
4. Tests + 2–3 eval scenarios if behavior changes.
5. README note under scope gate.

### Allowed file scope

`harness/policy.py`, `api/server.py`, `tests/`, `evals/scenarios.yaml` (additive scenarios only).

### Exit criteria

- False positive rate low on existing evals.
- New refusal cases covered by tests.

### Rules

Follow shared context. STOP for green light.

**Start Mission 6 now.**

## Paste to here

---

# Mission 7 — README / portfolio polish

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Mission 7 only:** FocusKPI write-up pass — **docs only**, no new features. [NEXT_STEPS.md](NEXT_STEPS.md) §4.

### Your tasks

1. Read README as a hiring manager (3–6 paragraph pitch). Tighten:
   - Opening hook (why this repo, not another agent tutorial)
   - "What's novel" — lead with grounded confidence + memory + eval honesty
   - Clear "Run it in 5 minutes" path
2. Sync any stale numbers (337 tests, 28 scenarios, PR #13 merged).
3. Optional: add link to `evals/LIVE.md` if Mission 2 ran.
4. Do **not** rewrite architecture unless inaccurate.

### Allowed file scope

`README.md`, `NEXT_STEPS.md` (portfolio section only).

### Exit criteria

- README stands alone as application write-up draft.
- No code changes.

### Rules

Follow shared context. STOP for green light.

**Start Mission 7 now.**

## Paste to here

---

# Mission 8 — Semantic codebase search

## Paste from here

You are working on `agent-harness` at `/Users/jorge/Projects/agent-harness`.

**Mission 8 only:** Implement deferred semantic search. **Only run if Jorge confirms grep-only explore evals are insufficient.**

### Prerequisite

Mission 2 live eval or manual testing shows explore failures ripgrep cannot fix.

### Your tasks

1. Implement `semantic_search` in [tools/semantic.py](tools/semantic.py):
   - Embed via provider boundary only (no direct ollama import outside `providers/`)
   - Chroma collection scoped to workspace root
   - Ignore globs match workspace module
2. Index on first use or explicit `index_workspace` tool — document trade-off.
3. Register alongside `grep_repo`; eval scenario updates if scores shift.
4. README: document when to use grep vs semantic.

### Allowed file scope

`tools/semantic.py`, `workspace/`, `providers/` (if embed hook needed), `api/server.py`, `tests/`, `evals/`, `README.md`.

### Exit criteria

- Semantic search returns file:line citations compatible with grounder.
- Offline evals still deterministic (script semantic tool in YAML if needed).

### Rules

Follow shared context. Do not add LangChain. STOP for green light.

**Start Mission 8 only if Jorge confirmed the trigger.** Otherwise propose grep-first mitigations and STOP.

## Paste to here

---

# Appendix — Pivot phases 0–10 (COMPLETE)

Coding-agent pivot shipped on `main` (PR #13). Phase-by-phase history: [CODING_AGENT_PIVOT.md](CODING_AGENT_PIVOT.md).

| Phase | Outcome |
|-------|---------|
| 0 | Support baseline frozen |
| 1 | README + API shape (`workspace_root`, envelope fields) |
| 2 | Workspace jail + read-only code tools |
| 3 | Grounding → file:line citations |
| 4 | Write + verify tools |
| 5 | Engineering memory |
| 6 | Coding eval harness (28 scenarios) |
| 7 | Ripgrep-first; `ENABLE_SUPPORT_TOOLS=false` default |
| 8 | Docker + demo.md coding walkthrough |
| 9 | Scope gate, edit budget, eval honesty |
| 10 | Final gate (~337 tests), reviewer checklist |

For support-only debugging: `ENABLE_SUPPORT_TOOLS=true`, seed/embed, `evals/scenarios_support.yaml`.
