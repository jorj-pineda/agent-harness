# agent-harness

A local-first, pluggable-provider agent harness optimized for enterprise customer support. Two headline features differentiate it from a toy ReAct loop: **grounded answers with confidence scoring** and **cross-session personalization memory**. Built from scratch (no LangChain/LlamaIndex) to demonstrate understanding of agent internals.

This is a portfolio project — architectural decisions should be defensible in a 3–6 paragraph written write-up, and the README is the source of that write-up.

## Current status

Steps 1–12 merged as of 2026-04-23 (Docker + README, PR #11). Post-MVP demo hardening on `feat/demo-readiness` (2026-05): `demo.md`, eval `--live` path, Chroma typing fix.

Test suite: **293 collected** (289 pass + 4 live-provider skips in default CI). `ruff` clean; `mypy --strict` clean on core layers (`harness`, `grounding`, `memory`, `tools`, `providers`).

README is the FocusKPI write-up and holds the **offline eval headline table**. `evals/report.md` is gitignored — regenerate with `python -m evals.run` and update README if metrics change.

## Architecture

Layered, each layer depends only on the ones below it.

```
api/            FastAPI server — thin HTTP wrapper over harness
  └── harness/  ReAct loop, state, router
        ├── grounding/   confidence scoring, citations, escalation
        ├── memory/      short-term window, summarizer, long-term facts
        ├── tools/       typed registry + SQL tools + RAG tool + memory tool
        ├── data/        SQLite schema, seed, embedded doc corpus
        └── providers/   Ollama / Anthropic / OpenAI behind one interface
```

Rule: **nothing above `providers/` may import a specific provider.** Model-specific quirks (Gemma 4 tool-call format vs OpenAI's) are normalized at the provider boundary.

## Stack

- **Python 3.11+** (match-statement used liberally, strict type hints)
- **Ollama** for local inference; primary model is **Gemma 4 E4B** (Apache 2.0, released 2026-04-02). `ollama run gemma4`.
- **Cloud providers:** Anthropic Claude, OpenAI — behind the same `Provider` interface.
- **Embeddings:** `nomic-embed-text` via Ollama (runs alongside Gemma 4, no extra deps).
- **Vector store:** Chroma (simpler than FAISS for a demo; persistent client on disk).
- **Database:** SQLite for both mock support data and long-term memory. Read-only DB user for the SQL tool.
- **API:** FastAPI + Pydantic v2.
- **Tests:** pytest + pytest-asyncio. Coverage target ~70% on harness/grounding/memory/tools.

## Hardware constraint

Primary target: **RTX 4070 laptop, 8GB VRAM**. Every default must run comfortably here.

- Gemma 4 E4B (Q4_K_M) is the primary model — safe fit.
- Gemma 4 26B MoE is an upgrade path; document as "requires 16GB+ VRAM workstation" in README.
- Set `OLLAMA_KV_CACHE_TYPE=q8_0` in the docker-compose env to keep 32k-context memory ~5GB instead of ~15GB. Call this out in the README as a deliberate optimization.

## Critical rules

1. **No agent frameworks.** No LangChain, LlamaIndex, LangGraph, CrewAI, Haystack. Building the loop from scratch is the _point_. If you catch yourself wanting a framework helper, write the 10 lines yourself — the code is the portfolio.
2. **Provider abstraction is sacred.** The harness must not `import ollama` or `import anthropic` anywhere outside `providers/`. Adding a new backend should be a single-file change.
3. **Always web-fetch when a local/open-weights model is discussed.** Versions and sizes change monthly. Never answer from training knowledge about Gemma/Llama/Qwen/etc. specs.
4. **SQL tool is read-only.** Separate DB connection with a read-only SQLite pragma; queries parameterized; row limit enforced server-side; no DDL/DML tokens in generated SQL. Log every query.
5. **Every response ships with metadata.** `{answer, confidence, citations, escalated, tool_calls, memory_writes, provider, latency_ms}` — not just a string. The eval harness depends on this shape.
6. **Determinism in tests.** Provider calls in tests go through a recorded-response fake (VCR-style cassettes in `tests/cassettes/`). Never hit live APIs in CI. Eval matrix defaults to offline `FakeProvider`; use `evals.run --live` only for optional manual provider comparison.
7. **Secrets live in `.env`**, loaded via `pydantic-settings`. `.env.example` is committed, `.env` is gitignored. No hardcoded keys, ever.

## Eval harness

The eval suite is not optional — it's the differentiator for the job pitch. Minimum:

- `evals/scenarios.yaml` — ~30 scripted support conversations covering: grounded factual Q&A, personalization recall across sessions, off-topic refusal, low-confidence escalation triggers, adversarial prompt injection attempts.
- `evals/scorers.py` — faithfulness (every factual claim covered by a cited chunk), correctness (vs gold answer), memory-recall accuracy, escalation precision/recall.
- `evals/run.py` — runs the full matrix (every scenario × every provider). Default: offline `FakeProvider` (deterministic, CI-safe). `--live` calls real backends. Emits `evals/report.md`.

The README headline table is the committed offline snapshot. `evals/report.md` is gitignored. Re-run default evals after harness changes; update README if metrics shift. Eval runner default escalation threshold is **0.50**; the API default is **0.55** (`CONFIDENCE_ESCALATION_THRESHOLD`).

## Commands

```bash
# install
uv sync --extra dev              # or: pip install -e .[dev]

# docker demo (see demo.md)
docker compose up --build -d
docker exec agent-harness-ollama ollama pull gemma4
docker exec agent-harness-ollama ollama pull nomic-embed-text
docker exec agent-harness-app python -m data.seed
docker exec agent-harness-app python -m data.embed

# pull local models (dev without Docker)
ollama pull gemma4               # Gemma 4 E4B
ollama pull nomic-embed-text

# seed data + embed corpus
python -m data.seed
python -m data.embed

# run
uvicorn api.server:app --reload

# tests
pytest -m "not live"             # CI default; skips live provider tests
pytest -m live                   # optional: real Ollama / API keys

# evals (offline — README headline table)
python -m evals.run --providers ollama,anthropic,openai

# evals (live — non-deterministic, manual only)
python -m evals.run --live --providers ollama
```

## Conventions

- **Type hints everywhere.** `mypy --strict` compatible for the core layers (harness, grounding, memory, tools, providers). API/data/evals can be looser.
- **Pydantic models at layer boundaries.** Tool inputs/outputs, API requests/responses, Message/Turn/Session — all Pydantic. No stringly-typed dicts flowing across modules.
- **Async all the way from API down to providers.** Ollama and cloud providers are I/O-bound; the harness loop is async.
- **Line length 100.** Formatter: `ruff format`. Linter: `ruff check` with a strict ruleset.
- **No comments that restate code.** Comments only for _why_ — a constraint, a workaround, a non-obvious invariant. Never a WHAT comment on a well-named function.
- **Commits: imperative mood, present tense.** "Add confidence scoring" not "Added" or "Adds".

## Parallel collaboration (when multiple agents are active)

When Jorge is running several Claude instances at once (Claude Code, Antigravity, Cursor), coordination is enforced by file-scope discipline, not locking. Rules:

- **Stay in your file scope.** Drive-by edits in other layers cause merge pain — stop and ping Jorge instead.
- **Claude Code owns `CLAUDE.md`.** Other agents record decisions in commit messages; Claude Code syncs the docs.
- **`pyproject.toml` edits are additive only** — append to a step-labeled block (`# step-10 api`, etc.). No reorders, no re-pins of existing deps.
- **Pause-per-substep still applies** on large branches: summary + commit message + Jorge's greenlight before moving on.

## Deferred (revisit when there's a real need)

Scope expansions we considered but postponed. Revisit only when a concrete use case forces them — premature abstractions here would be pure overhead.

- **Router fallback / retry across providers.** `ProviderRouter` is a plain dispatch table today. Adding failover (primary flakes → try backup) needs real error patterns to design against; don't speculate on shape.
- **Embedder routing through the router.** Only one consumer (RAG tool) and it takes an embedder directly. Add a second consumer before indirecting.
- **Per-provider ToolSpec translation as a shared layer.** Each provider converts `ToolSpec` into its own tool-call format internally — lifting that into a shared mapper only pays off if translation grows non-trivial.
- **LLM-judge confidence.** `Grounder` today uses a deterministic heuristic (`top_score * coverage * health`). An LLM self-assessment judge could plug in behind the same `Grounder.ground()` interface once the heuristic baseline underperforms on evals — but self-reports are systematically over-confident, so validate on scenarios before swapping.
- **Per-sentence citation attribution.** Grounding emits citations at the turn level (which chunks the answer draws from). Mapping individual claims to specific chunks needs a post-generation pass (spans, NLI, or a cited-output schema) — defer until the eval harness has a faithfulness metric that rewards it.
- **Rewriting the answer on escalation.** Today `escalated=True` is a flag; the raw answer is preserved so the API layer owns presentation. Swapping in a templated "I'm not sure — let me hand you off" message is a UX decision, not a harness one.
- **Session persistence.** Sessions live in an in-memory `dict[session_id, Session]` inside the FastAPI process. Survives neither restarts nor a multi-worker deployment. Swap for Redis or a `sessions` SQLite table when the demo grows beyond a single uvicorn process.

## Out of scope (explicitly)

Things that sound like they belong but don't — noting them here so we don't accidentally build them:

- User authentication / multi-tenant session isolation (overlaps with your Duodoro project; not a differentiator here).
- A chat UI (decided: FastAPI-only; demo via curl or an HTTP client).
- Fine-tuning Gemma 4 (impressive-sounding but a different project).
- LangChain/LlamaIndex integration (see rule #1).
- Production observability stack (OpenTelemetry, Prometheus). A structured `logging` setup with JSON output is enough for a portfolio piece.

## Links

- FocusKPI role this project supports: Junior AI/ML Engineer — applications go to `danz@focuskpi.com` with resume, GitHub, and a 3–6 paragraph write-up of one project. The README of this repo _is_ that write-up draft.
- Gemma 4 model card: https://ai.google.dev/gemma/docs/core/model_card_4
- Ollama library: https://ollama.com/library/gemma4
