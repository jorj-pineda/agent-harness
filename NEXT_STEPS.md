# Next steps after coding-agent pivot (phases 1–10)

**Status (2026-06-01):** Phases 1–10 **merged to `main`** (PR #13). Missions **1–7 done** (ship, live eval, CI, `emit_plan`, patch summary, scope gate, README polish). Ollama `tool_name` multi-turn fix merged. **Mission 9 agent panel complete** on `feat/gui-integration` — slices **9a (Typer CLI)**, **9b (static panel)**, **9c (SSE live tool cards)**, **9d (polish + docs)** all shipped; only the real `docs/panel.png` capture is pending ([GUI-integ.md](GUI-integ.md)). Suite at **381 tests** (`pytest -m "not live"`). Mission 8 semantic search remains conditional.

This doc is the post-pivot backlog — what to do **after** merge, in priority order.

**New agent chats:** paste a mission from [COMPOSER_SUPER_PROMPT.md](COMPOSER_SUPER_PROMPT.md) (Mission 0 = context reload).

---

## 1. Ship (do first)

| Task | Why | Done when |
|------|-----|-----------|
| ~~Merge `feat/pivot-harness` → `main`~~ | Locks the portfolio narrative | ✅ PR #13 merged |
| ~~**Tag release**~~ | Recoverable snapshot | **Proposed** (awaiting approval): `v0.2-coding-agent` → `acb8cbe` |
| ~~**Run reviewer checklist locally once**~~ | Catch env-specific gaps | ✅ 337 passed, ruff/mypy clean, offline eval 84/84; Docker `/chat` blocked by RAM (see [demo.md](demo.md)) |
| **Optional: tag support baseline** | Pre-pivot snapshot | **Proposed** (awaiting approval): `v0.1-support` → `1ce9f9e` (pre-pivot `main`) |

```bash
git checkout main && git pull
# after merge:
git tag v0.2-coding-agent
pytest -m "not live" && ruff check . && mypy
python -m evals.run --providers ollama,anthropic,openai
```

---

## 2. Docs sync (same week as merge)

| Task | Owner | Notes |
|------|-------|-------|
| ~~**Sync `CLAUDE.md`**~~ | Done | Coding-agent status, 337 tests, new defaults |
| ~~**Sync `COMPOSER_SUPER_PROMPT.md`**~~ | Done | Post-pivot missions 0–9; pivot phases archived |
| ~~**FocusKPI README pass**~~ | Done | Mission 7 on `chore/readme-portfolio` |

---

## 3. Validate beyond offline CI (high value, low code)

| Task | Effort | Outcome |
|------|--------|---------|
| ~~**Live eval smoke**~~ | ~30 min | ✅ 3-scenario smoke documented in [evals/LIVE.md](evals/LIVE.md); `gemma4` OOM, `llama3.2:1b` fallback |
| ~~**Docker coding demo on your machine**~~ | ~15 min | ✅ Stack up; `/sessions` OK; `/chat` OOM on gemma4 with 8 GiB Docker RAM — constraint documented in [demo.md](demo.md) |
| **Cloud provider demo** ← **optional next** | ~15 min | `.env` with Anthropic/OpenAI; same bugfix curl; confirm tool-call wire format |

Offline eval proves **harness shape**; live runs prove **model behavior**. Both belong in the portfolio story.

---

## 4. Portfolio / application (parallel track)

| Task | Notes |
|------|-------|
| ~~**GitHub repo polish**~~ | README FocusKPI pass (Mission 7); eval table + [demo.md](demo.md) + [evals/LIVE.md](evals/LIVE.md) linked |
| **FocusKPI application** | Resume + GitHub + **this README** as the 3–6 paragraph write-up → `danz@focuskpi.com` |
| **Demo video (optional)** | 3–5 min: session → bugfix curl → envelope fields; Anthropic recommended for full edit→verify chain |
| **Second project** | Duodoro or another repo — don’t let agent-harness block the application window |
| **Tag release (optional)** | `v0.2-coding-agent` at post-pivot merge — proposed in §1 |

---

## 5. Technical follow-ups (prioritized backlog)

Pick **one at a time** after merge. Each should be a small PR with tests.

### Tier A — strengthens the “senior agent” story

| Item | What | Trigger |
|------|------|---------|
| ~~**Live eval headline row**~~ | Run matrix live on Ollama; add “live snapshot” row to README (separate from offline table) | Partial — 3-scenario smoke + 4070 native gemma4 notes in [evals/LIVE.md](evals/LIVE.md) |
| ~~**`emit_plan` tool**~~ | Structured plan in tool trace before edits | Shipped; optional `REQUIRE_PLAN_BEFORE_EDIT` gate (default off) |
| ~~**Diff summary in envelope**~~ | `patch_summary` field from `write_file` results | Shipped (PR #15) |
| ~~**Stronger scope gate**~~ | Classify bugfix vs explore vs refactor; refuse unbounded rewrites | Shipped (PR #15) |
| **Agent panel demo UI** | Typer CLI + local web panel (Cursor-style tool trace) | [GUI-integ.md](GUI-integ.md) Mission 9 — post-application “wow” demo |
| **SSE streaming `/chat`** | Live tool cards in UI | Slice 9c in [GUI-integ.md](GUI-integ.md); pairs with agent panel |

### Tier B — deferred from README (only if evals demand it)

| Item | What | Trigger |
|------|------|---------|
| **Semantic search** | Implement `semantic_search` behind `providers/` embed; Chroma over repo | Explore-only eval failures with grep-only |
| **Session persistence** | Redis or SQLite `sessions` table | Multi-worker or demo across restarts |
| **LLM-judge confidence** | Plug behind `Grounder.ground()` | Heuristic systematically wrong on live runs |
| **Per-sentence citations** | NLI or span attribution | Faithfulness scorer needs it |

### Tier C — ops / hardening

| Item | What |
|------|------|
| ~~**CI workflow**~~ | GitHub Action: `pytest -m "not live"`, ruff, mypy, `evals.run` — [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |
| **Provider failover** | Router retry on timeout (see CLAUDE.md deferred) |
| ~~**Support path CI job**~~ | Offline `scenarios_support.yaml` matrix in CI (scripted FakeProvider; no Ollama/seed) |

---

## 6. Explicit non-goals (don’t start unless requirements change)

- LangChain / LangGraph / CrewAI
- IDE / LSP plugin
- SWE-bench leaderboard runs
- Multi-tenant auth
- Unbounded shell
- Fine-tuning custom models
- Autonomous multi-hour runs without human checkpoint

---

## 7. Suggested timeline

```
Week 1   Tag v0.2-coding-agent, reviewer checklist, live smoke + Docker demo
Week 2   FocusKPI submit (README = write-up); optional demo video
Week 3+  Mission 9 agent panel ([GUI-integ.md](GUI-integ.md)) for demo UX
         Mission 8 semantic search only when live eval proves grep insufficient
```

---

## 8. Quick reference

| Goal | Command |
|------|---------|
| Offline eval (README table) | `python -m evals.run --providers ollama,anthropic,openai` |
| Support regression eval | `python -m evals.run --scenarios evals/scenarios_support.yaml` (with support tools seeded) |
| Live provider comparison | `python -m evals.run --live --providers ollama` |
| Coding demo | [demo.md](demo.md) |
| Agent panel plan (Mission 9) | [GUI-integ.md](GUI-integ.md) |
| Pivot history | [CODING_AGENT_PIVOT.md](CODING_AGENT_PIVOT.md) |
| Composer missions (new chats) | [COMPOSER_SUPER_PROMPT.md](COMPOSER_SUPER_PROMPT.md) |

---

## Decision log

| Date | Decision |
|------|----------|
| 2026-05 | Pivot phases 1–10 merged to `main` (PR #13) |
| 2026-05 | Mission 1 ship checklist: baseline green; Docker gemma4 OOM documented; tags proposed |
| 2026-05 | Mission 2 live eval: 3-scenario smoke in [evals/LIVE.md](evals/LIVE.md); gemma4 OOM, llama3.2:1b fallback |
| 2026-05 | Mission 3 CI: [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — baseline + support offline eval |
| 2026-05 | Mission 4 `emit_plan` tool + optional `REQUIRE_PLAN_BEFORE_EDIT` gate |
| 2026-05 | `CLAUDE.md` + `COMPOSER_SUPER_PROMPT.md` synced for post-pivot missions |
| 2026-05 | Indexing: **ripgrep-first** (A); semantic search deferred |
| 2026-05 | Default registry: **coding tools on**, support tools off (`ENABLE_SUPPORT_TOOLS=false`) |
| 2026-05 | Missions 5–6: `patch_summary`, scope gate (PR #15) |
| 2026-05 | Mission 7 README portfolio pass |
| 2026-05 | Ollama multi-turn fix: `tool_name` on tool messages; 4070 live notes in [evals/LIVE.md](evals/LIVE.md) |
| 2026-05 | Mission 9 agent panel planned — [GUI-integ.md](GUI-integ.md) (demo UI OK; full IDE still out of scope) |
| 2026-06-01 | Mission 9 slice 9a: Typer CLI (`agent-harness serve`/`chat`); fixed `.env` test-isolation leak (`_env_file=None`) |
| 2026-06-01 | Mission 9 slice 9b: static demo panel (`ui/`) mounted at `/` — tool cards + envelope rail |
| 2026-06-01 | Mission 9 slice 9c: SSE `GET /chat/stream` + `harness/stream.py` `on_event` hook; live tool cards (no loop duplication) |
| 2026-06-01 | Mission 9 slice 9d: demo.md panel walkthrough, README panel blurb + ASCII preview, `docs/` screenshot recipe — **Mission 9 complete** (real `docs/panel.png` capture pending Jorge) |

Update this table when Tier A/B items land.
