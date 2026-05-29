# agent-harness — Composer 2.5 Super Prompt

Use this file to start a new Composer 2.5 chat. Copy everything under **"Paste from here"** into the chat input.

---

## Quick recap (for you, not the agent)

- **What it is:** Local-first ReAct agent harness for **enterprise customer support** (not a coding agent).
- **Headline features:** Grounded answers with confidence scoring + cross-session personalization memory.
- **Status:** Steps 1–12 merged (Docker + README). ~293 tests. Portfolio hardening is what's left.
- **Workflow:** Agent works in 7 slices, pauses after each for your green light before continuing.

---

## Paste from here

You are working on the `agent-harness` repo at /Users/jorge/Projects/agent-harness.

## Project summary

A local-first, pluggable-provider agent harness for **enterprise customer support** — NOT a coding/SWE agent. Built from scratch (no LangChain/LlamaIndex/LangGraph). Portfolio piece for a Junior AI/ML Engineer application.

Two headline features:
1. **Grounded answers with confidence scoring** — deterministic heuristic over RAG chunks; `escalated=True` below threshold; citations on every grounded turn.
2. **Cross-session personalization memory** — SQLite `FactStore` per user_id; facts injected into system prompt + remember/recall tools.

Every response envelope: `{answer, confidence, citations, escalated, tool_calls, memory_writes, provider, latency_ms}`.

Stack: Python 3.11+, FastAPI, Pydantic v2, SQLite, Chroma, Ollama (Gemma 4 E4B default), Anthropic, OpenAI. Target hardware: RTX 4070 laptop, 8GB VRAM. Docker: two services (Ollama + app), `OLLAMA_KV_CACHE_TYPE=q8_0`.

Architecture layers (each depends only on layers below):

    api/            FastAPI — thin HTTP wrapper
      └── harness/  ReAct loop, state, router
            ├── grounding/   confidence, citations, escalation
            ├── memory/      FactStore, system-prompt injection
            ├── tools/       registry + SQL (read-only) + RAG + memory
            ├── data/        SQLite mock support DB + Chroma corpus
            └── providers/   Ollama / Anthropic / OpenAI (SACRED abstraction)

## Current status (as of 2026-05-29)

Steps 1–12 MERGED. Latest: PR #11 Docker + README.
- 293 pytest tests, ruff + mypy clean on core layers
- README.md IS the FocusKPI write-up (includes eval headline table)
- evals/: 30 scenarios, scorers (faithfulness, correctness, memory recall, escalation), run.py CLI
- evals/report.md is GITIGNORED — regenerate with `python -m evals.run --providers ollama,anthropic,openai`
- evals/run.py `_build_provider()` still uses FakeProvider with SHARED scripts for all providers — real provider comparison not wired yet
- Sessions in-memory only (deferred)
- CLAUDE.md may be stale (gitignored); trust README + code

Critical rules (non-negotiable):
1. NO agent frameworks (LangChain, etc.)
2. Nothing above providers/ may import a specific provider
3. SQL tool is read-only (pragma, parameterized, row limit, no DDL/DML, log queries)
4. Every response ships full metadata envelope
5. Tests use VCR cassettes / FakeProvider — never hit live APIs in CI
6. Secrets in .env via pydantic-settings; .env.example committed
7. pyproject.toml edits: additive only, step-labeled blocks
8. Line length 100, ruff format/check, mypy --strict on core layers
9. No comments that restate code; commits imperative present tense
10. Do NOT commit unless Jorge explicitly asks — but DO propose commit messages each slice

Out of scope unless Jorge asks: auth, chat UI, fine-tuning, LangChain, OpenTelemetry, coding-agent features.

## Your mission: post-MVP portfolio hardening

Jorge is returning after ~5 weeks away. The core build is done. Your job is to get the repo demo-ready and eval-credible again.

### Slice plan (execute in order)

**Slice 1 — Baseline verification**
- Read README.md, pyproject.toml, docker-compose.yml, api/server.py, harness/loop.py, evals/run.py
- Run: `pytest` (or `pytest -m "not slow"`), `ruff check`, `mypy` on core packages if configured
- Note any failures, broken imports, or drift from README claims
- Do NOT fix yet unless trivial — report first, then fix in Slice 2

**Slice 2 — Fix anything broken from Slice 1**
- Fix test/lint/type failures with minimal diffs
- Match existing conventions in surrounding code

**Slice 3 — Docker end-to-end smoke**
- Verify Dockerfile + docker-compose.yml build and start
- Document exact commands for: model pull (gemma4, nomic-embed-text), data.seed, data.embed, curl /sessions + /chat
- Fix any compose/env/path issues found (SQLITE_DB_PATH, CHROMA_PATH, OLLAMA_HOST, etc.)

**Slice 4 — Eval report regeneration**
- Run `python -m evals.run --providers ollama,anthropic,openai`
- Compare output to README headline table; update README table + narrative if numbers/metrics changed
- Note: report.md stays gitignored — README is the committed source of truth for the headline table

**Slice 5 — Eval provider story (pick the minimal honest improvement)**
- Read evals/run.py `_build_provider` and tests/cassettes/
- Either:
  (A) Wire per-provider cassette replay into eval matrix so columns CAN diverge, OR
  (B) Add clear README/docs explaining FakeProvider limitation + add a `@pytest.mark.live` optional path for real provider evals
- Choose (A) if cassettes already cover scenario shapes; choose (B) if wiring is large — be honest in README either way
- Keep CI offline/deterministic

**Slice 6 — Doc sync**
- If CLAUDE.md exists locally: update "Current status" to reflect steps 1–12 complete
- Ensure README "Run it" and "Eval results" sections match reality after your changes
- Remove any stale TODO references

**Slice 7 — Final gate**
- Full pytest pass
- Summarize what a reviewer should try: `docker compose up`, curl example, one eval command
- List deferred items still untouched (session persistence, LLM-judge confidence, etc.)

## Workflow rules (MANDATORY)

**Split the task into slices, after each slice pause, and give the user a commit msg and description of what was done in that slice, wait for a green light from user before continuing.**

Per slice output format:
1. **What I did** — 3-6 bullet points
2. **Evidence** — command output summaries (test counts, pass/fail)
3. **Files touched** — list
4. **Proposed commit message** — imperative mood, present tense, 1-2 sentences focusing on WHY
5. **Stop** — ask Jorge for green light before starting the next slice

Do NOT:
- Create git commits unless Jorge explicitly says "commit"
- Expand scope into deferred features without asking
- Import provider SDKs outside providers/
- Add frameworks
- Edit unrelated files

Do:
- Read before editing
- Minimal focused diffs
- Reuse existing patterns (Pydantic models, async, ToolRegistry, FakeProvider, cassettes)
- Web-fetch if discussing Gemma/Ollama model specs (versions change)

Start with Slice 1 now. Read the key files, run pytest, report baseline status, then STOP and wait for green light.

---

## Paste to here
