# Pivot roadmap: support harness → senior coding agent

Steps to take this repo from **demo-ready customer-support harness** (post `feat/demo-readiness`, May 2026) to **almost-ready senior-level coding agent harness** — still hand-written loop, no LangChain, provider abstraction intact.

The goal is not “add a file-read tool and call it a coding agent.” Senior-level means: multi-step repo reasoning, safe edits with verification, grounded citations to code, confidence/escalation when the agent is guessing, and an eval suite that scores real engineering outcomes — not support-policy overlap.

---

## What you have today (keep)

| Asset | Why it survives the pivot |
|-------|---------------------------|
| `harness/loop.py` ReAct controller | Tool-agnostic; swap tools, keep loop |
| `providers/` abstraction + cassettes | Same backends; coding agents need them more |
| `TurnResponse` metadata envelope | `{answer, confidence, citations, escalated, tool_calls, memory_writes, provider, latency_ms}` maps cleanly to “patch + evidence + uncertainty” |
| `harness/grounding.py` | Retarget from RAG chunks → file/line citations; heuristic stays |
| `memory/FactStore` | Repo conventions, user prefs, past decisions across sessions |
| `evals/` matrix + scorers pattern | Replace scenarios; keep offline `FakeProvider` + optional `--live` |
| `demo.md`, Docker, pytest/mypy/ruff gate | Operational baseline from demo-readiness pass |
| Provider rule | Nothing above `providers/` imports a specific SDK |

---

## What must change (support → coding)

| Support layer | Coding replacement |
|---------------|-------------------|
| `data/` mock SQLite + Chroma doc corpus | Target **workspace** (git repo on disk); optional indexed codebase (embeddings or ripgrep-first) |
| `tools/sql.py`, `tools/rag.py` | **Code tools**: read/grep/list/tree, edit/patch, run tests/lint, git status/diff (bounded shell) |
| Support system prompt | **Engineering system prompt**: scope discipline, minimal diffs, run verification before claiming done |
| 30 support `evals/scenarios.yaml` | **Coding scenarios**: bugfix, feature slice, refactor, test-add, adversarial (bad edit request), escalation on low confidence |
| README narrative | FocusKPI support story → senior agent story (grounded edits + cross-session repo memory) |
| Default demo (`return window?`) | Demo: clone repo → fix failing test → show citations + confidence |

---

## Definition of “almost ready” (coding pivot)

A reviewer can:

1. `docker compose up` (or local uvicorn) with **one** model path documented (Ollama *or* cloud).
2. Point the agent at a **small fixture repo** ( vendored under `fixtures/` or mounted volume).
3. Ask for a multi-file task; get a response with **tool trace**, **file:line citations**, **confidence**, and **escalated** when evidence is thin.
4. Run `pytest` + offline eval matrix; see scores on **patch correctness**, **test pass**, **faithfulness to cited lines**, **escalation accuracy**.
5. Read README as a 3–6 paragraph write-up of a **coding agent**, not support.

Not required for “almost ready”: full SWE-bench leaderboard, autonomous multi-hour runs, IDE plugin, or production sandboxing.

---

## Phased steps (execute in order)

Each phase ends with: tests green, proposed commit message, pause for green light.

### Phase 0 — Freeze the support baseline

**Status:** Done on `feat/demo-readiness`.

- [x] pytest / ruff / mypy clean on core layers
- [x] Docker smoke documented in `demo.md`
- [x] Offline eval regenerates README headline table
- [x] `--live` eval path for real provider comparison

**Before pivot:** merge `feat/demo-readiness` → `main` (or tag `v0.1-support`) so support demo is recoverable.

---

### Phase 1 — Narrative + API shape (no tools yet)

Rebrand without breaking the loop.

1. **README** — rewrite headline features:
   - *Grounded edits with confidence* (citations = file paths + line ranges)
   - *Cross-session repo memory* (conventions, stack choices, prior review notes)
2. **`api/server.py`** — extend request model:
   - `workspace_id` or `repo_path` (sandbox root)
   - keep `user_id` for memory scoping
3. **System prompt** — replace support assistant with coding-agent contract (minimal diffs, verify with tests, cite files read).
4. **Envelope** — add optional fields if needed (`files_touched`, `verification_commands`) without removing existing keys evals depend on.

**Exit:** API boots; old tools still work; README describes coding agent intent.

---

### Phase 2 — Workspace sandbox + read-only code tools

Senior agents read before they write.

1. **`workspace/` module** (or `data/workspace.py`):
   - Resolve and **jail** all paths to a configured root (no `/etc/passwd` escapes).
   - `Workspace` dataclass: root, ignore globs (`.git`, `node_modules`, `.venv`).
2. **Tools (read-only first):**
   - `read_file(path, start_line?, end_line?)` — capped bytes/lines
   - `grep_repo(pattern, path?, glob?)` — wrap ripgrep or pure Python; hard timeout + hit limit
   - `list_dir(path)` / `tree(path, depth)` — bounded listing
   - `git_diff`, `git_status` — read-only git subprocess with allowlist args
3. **Registry factory** — per request: bind workspace root (like memory tools bind `user_id`).
4. **Tests** — fixture repo under `tests/fixtures/tiny_repo/`; assert path jail + row limits.

**Exit:** Agent can explore a fixture repo via tools; SQL/RAG tools optional or deprecated behind flag.

---

### Phase 3 — Grounding retarget (code citations)

Reuse `Grounder`; change inputs.

1. **`harness/grounding.py`** — accept citations from:
   - `read_file` / `grep_repo` tool results (file, line range, snippet hash)
   - not Chroma chunk IDs
2. **Heuristic** — same shape: `top_evidence_score × coverage × health`; escalate when agent edits without reading cited regions.
3. **`TurnResponse.citations`** — schema: `{path, start_line, end_line, snippet?}`.
4. **Tests** — port `test_harness_grounding.py` scenarios to code-style tool records.

**Exit:** Confidence/escalation fires on coding tool traces; support RAG tests removed or isolated.

---

### Phase 4 — Write tools + verification loop

Senior = edit **and** prove.

1. **`apply_patch` / `write_file`** — unified diff or full-file replace; reject edits outside workspace.
2. **`run_command`** — allowlist only: `pytest`, `ruff`, `mypy`, `npm test`, `git diff`, etc.; timeout, no shell injection (`exec` argv list, never `shell=True`).
3. **Harness policy** — encourage loop: read → edit → run tests → summarize (max iterations already capped).
4. **Optional:** block `finish` until at least one verification command succeeded (config flag).

**Exit:** End-to-end fix on `tests/fixtures/tiny_repo` broken test; metadata shows tool chain + verification.

---

### Phase 5 — Memory for engineering context

Repurpose `FactStore`; don’t throw away SQLite layer.

1. Rename/document facts as **durable engineering notes** (not “user likes blue”).
2. Tools: keep `remember_fact` / `recall_facts` or rename to `remember` / `recall`.
3. System prompt injection — repo conventions (“use ruff”, “no force-push”), prior session decisions.
4. **Tests** — cross-session recall: session 1 remembers “prefer async”; session 2 applies it.

**Exit:** Personalization eval category becomes “repo memory recall.”

---

### Phase 6 — Eval harness for coding

Replace support scenarios; keep runner architecture (`evals/run.py`).

1. **`evals/scenarios.yaml`** — new categories (~25–30 scenarios):
   - `bugfix` — failing test in fixture repo; gold = test passes
   - `feature_slice` — add function + test; gold = diff contains symbol + test pass
   - `refactor` — rename/extract; gold = behavior preserved
   - `explore_only` — answer question about codebase; gold = citations cover claims
   - `low_confidence` — ambiguous spec; gold = `escalated=true`
   - `unsafe_request` — delete `.git`, exfiltrate; gold = refusal
2. **Scorers:**
   - `patch_correctness` — apply agent diff to fixture; run gold test command
   - `faithfulness` — claims ⊆ cited file regions (line overlap)
   - `verification` — did agent run allowed check?
   - keep `escalation` scorer
3. **Offline mode** — scripted tool results + FakeProvider (CI deterministic).
4. **`--live`** — run against real provider on fixture repos (slow, manual).

**Exit:** New README headline table; offline eval CI-safe.

---

### Phase 7 — Indexing strategy (pick one, don’t boil ocean)

For senior agents, **ripgrep-first** is enough for v1; embeddings optional.

| Option | Pros | Cons |
|--------|------|------|
| **A. Ripgrep only** | No embed model; fast; works offline | Weak semantic search |
| **B. Chroma over repo** | Reuses embed pipeline | Needs Ollama embed or cloud embed API |
| **C. Hybrid** | grep + embed on demand | More code |

**Recommendation:** Phase 2 ships **A**; add **B** only if evals show explore failures.

Remove or archive `data/seed.py`, support DB, doc corpus when coding path is default.

---

### Phase 8 — Docker + demo for coding

Update `demo.md` and compose.

1. Mount **fixture repo** or sample project volume into app container.
2. Document model options:
   - Full local: Ollama + `gemma4` (GPU / RAM notes from support demo)
   - Cloud chat + Ollama embed only (if keeping Chroma)
   - Cloud-only smoke: small tasks with grep-only tools
3. Demo script:
   ```bash
   curl -X POST .../sessions -d '{"user_id":"dev1","workspace":"/app/fixtures/tiny_repo"}'
   curl -X POST .../chat -d '{"message":"Fix the failing test in test_foo.py", ...}'
   ```

**Exit:** Reviewer reproduces coding demo in &lt;15 minutes with doc alone.

---

### Phase 9 — Senior-level polish (differentiators)

What separates “junior script kiddie agent” from “senior” in the portfolio:

1. **Scope gate** — router or pre-loop classifier: bugfix vs architecture question vs out-of-scope; refuse huge rewrites.
2. **Edit budget** — max files/lines changed per turn; escalate when exceeded.
3. **Structured plan tool** — optional `emit_plan` before edits (inspectable in envelope).
4. **Diff-first UX** — API returns patch summary + citations, not wall of text.
5. **Eval honesty** — README states offline vs live limits (already done for support; extend for coding).

Pick **2–3** for “almost ready”; defer the rest.

---

### Phase 10 — Final gate

Same bar as demo-readiness Slice 7:

- [ ] `pytest -m "not live"` full pass
- [ ] `ruff` + `mypy` on core layers
- [ ] Offline eval → README table updated
- [ ] `demo.md` coding walkthrough verified once on your machine
- [ ] Deferred list updated in README

---

## Suggested merge order (dependency chain)

```
Phase 0 (tag support baseline)
  → 1 narrative/API
  → 2 workspace + read tools
  → 3 grounding retarget
  → 4 write + verify tools
  → 5 memory repurposed
  → 6 evals
  → 7 indexing choice
  → 8 docker/demo
  → 9 polish (parallelizable)
  → 10 final gate
```

Phases 3–4 can overlap after Phase 2 lands. Phase 6 should wait until at least one fixture task runs end-to-end in Phase 4.

---

## Explicit non-goals (stay out unless forced)

- LangChain / LangGraph / CrewAI
- Full IDE / LSP integration
- Unbounded shell (`rm -rf`, curl pipes)
- Multi-tenant auth
- SWE-bench full leaderboard run
- Autonomous long-horizon runs (&gt;8 tool iterations without human)
- Fine-tuning custom models

---

## Risk register (from support demo work)

| Risk | Mitigation |
|------|------------|
| Gemma 4 OOM in Docker Desktop | Document RAM; default cloud provider for coding demo; smaller local model for smoke |
| Embeddings tie you to Ollama | Phase 7: ripgrep-first; or add OpenAI embed behind `providers/` |
| Eval columns identical (FakeProvider) | Accept for CI; use `--live` for provider shootouts; coding scorers add patch/test signal |
| Session loss on restart | Still deferred; document for demo; use single uvicorn process |
| Scope creep into “IDE replacement” | Phase 9 scope gate + edit budget |

---

## First PR after pivot (suggested scope)

**Phase 1 only:** README rebrand + `workspace_id` in API + coding system prompt. No tool deletion yet. Lets you iterate narrative while support tools remain for comparison.

---

## Reference commands (current baseline)

```bash
# Verify support baseline before pivot
pytest -m "not live"
python -m evals.run --providers ollama,anthropic,openai

# After Phase 6 (coding evals)
python -m evals.run --providers ollama,anthropic,openai   # offline
python -m evals.run --live --providers anthropic            # optional
```
