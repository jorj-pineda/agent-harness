# Next steps after coding-agent pivot (phases 1–10)

**Status:** Phases 1–10 **merged to `main`** (PR #13, merge `acb8cbe`). Support baseline recoverable via `evals/scenarios_support.yaml`.

This doc is the post-pivot backlog — what to do **after** merge, in priority order.

**New Composer chats:** paste a mission from [COMPOSER_SUPER_PROMPT.md](COMPOSER_SUPER_PROMPT.md) (Mission 0 = context reload).

---

## 1. Ship (do first)

| Task | Why | Done when |
|------|-----|-----------|
| ~~Merge `feat/pivot-harness` → `main`~~ | Locks the portfolio narrative | ✅ PR #13 merged |
| **Tag release** | Recoverable snapshot | e.g. `v0.2-coding-agent` on merge commit |
| **Run reviewer checklist locally once** | Catch env-specific gaps | [README reviewer checklist](README.md#reviewer-checklist) all green |
| **Optional: tag support baseline** | Pre-pivot snapshot | `v0.1-support` points at pre-pivot merge |

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
| ~~**Sync `COMPOSER_SUPER_PROMPT.md`**~~ | Done | Post-pivot missions 0–8; pivot phases archived |
| **FocusKPI README pass** | You or Mission 7 | README is the 3–6 paragraph write-up — see [COMPOSER_SUPER_PROMPT.md](COMPOSER_SUPER_PROMPT.md) Mission 7 |

---

## 3. Validate beyond offline CI (high value, low code)

| Task | Effort | Outcome |
|------|--------|---------|
| **Live eval smoke** | ~30 min | `python -m evals.run --live --providers ollama` (or anthropic) on 2–3 scenarios; document score spread in README or a short `evals/LIVE.md` note |
| **Docker coding demo on your machine** | ~15 min | Follow [demo.md](demo.md) end-to-end with real Gemma 4 tool-calling |
| **Cloud provider demo** | ~15 min | `.env` with Anthropic/OpenAI; same bugfix curl; confirm tool-call wire format |

Offline eval proves **harness shape**; live runs prove **model behavior**. Both belong in the portfolio story.

---

## 4. Portfolio / application (parallel track)

| Task | Notes |
|------|-------|
| **GitHub repo polish** | Pin README eval table; ensure demo.md is linked from repo description |
| **FocusKPI application** | Resume + GitHub + README as write-up to `danz@focuskpi.com` |
| **Demo video (optional)** | 3–5 min: session → bugfix curl → envelope fields on screen |
| **Second project** | Duodoro or another repo — don’t let agent-harness block the application window |

---

## 5. Technical follow-ups (prioritized backlog)

Pick **one at a time** after merge. Each should be a small PR with tests.

### Tier A — strengthens the “senior agent” story

| Item | What | Trigger |
|------|------|---------|
| **Live eval headline row** | Run matrix live on Ollama; add “live snapshot” row to README (separate from offline table) | After one successful `--live` run |
| **`emit_plan` tool** | Structured plan in tool trace before edits | If demo feels too “black box” |
| **Diff summary in envelope** | Optional `patch_summary` field from `write_file` results | UX polish for API consumers |
| **Stronger scope gate** | Classify bugfix vs explore vs refactor; refuse “rewrite entire repo” with tests | If manual testing shows false negatives |

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
| **CI workflow** | GitHub Action: `pytest -m "not live"`, ruff, mypy, `evals.run` |
| **Provider failover** | Router retry on timeout (see CLAUDE.md deferred) |
| **Support path CI job** | Optional matrix job with `ENABLE_SUPPORT_TOOLS=true` + `scenarios_support.yaml` |

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
Week 2   Application materials (Mission 7 README pass, submit FocusKPI)
Week 3+  One Tier-A PR if motivated (Mission 2/3/4/5/6 from COMPOSER_SUPER_PROMPT)
         Tier B (Mission 8 semantic search) only when live eval proves grep insufficient
```

---

## 8. Quick reference

| Goal | Command |
|------|---------|
| Offline eval (README table) | `python -m evals.run --providers ollama,anthropic,openai` |
| Support regression eval | `python -m evals.run --scenarios evals/scenarios_support.yaml` (with support tools seeded) |
| Live provider comparison | `python -m evals.run --live --providers ollama` |
| Coding demo | [demo.md](demo.md) |
| Pivot history | [CODING_AGENT_PIVOT.md](CODING_AGENT_PIVOT.md) |
| Composer missions (new chats) | [COMPOSER_SUPER_PROMPT.md](COMPOSER_SUPER_PROMPT.md) |

---

## Decision log

| Date | Decision |
|------|----------|
| 2026-05 | Pivot phases 1–10 merged to `main` (PR #13) |
| 2026-05 | `CLAUDE.md` + `COMPOSER_SUPER_PROMPT.md` synced for post-pivot missions |
| 2026-05 | Indexing: **ripgrep-first** (A); semantic search deferred |
| 2026-05 | Default registry: **coding tools on**, support tools off (`ENABLE_SUPPORT_TOOLS=false`) |
| 2026-05 | Senior polish shipped: **scope gate**, **edit budget**, **eval honesty** |

Update this table when Tier A/B items land.
