# agent-harness

A local-first, pluggable-provider agent harness optimized for enterprise customer support. Two headline features differentiate it from a toy ReAct loop: **grounded answers with confidence scoring** and **cross-session personalization memory**. Built from scratch (no LangChain/LlamaIndex) to demonstrate understanding of agent internals.

This is a portfolio project — architectural decisions should be defensible in a 3–6 paragraph written write-up, and the README is the source of that write-up.

## Current status

Step 5a complete as of 2026-04-19. Read-only SQL tools landed on `tools/sql.py` via a `build_sql_tools(db_path)` factory, with 20 tests in `tests/test_tools_sql.py`. Test suite: **116 passed**, ruff clean.

Next up: **5b — RAG tool** (Chroma nearest-neighbor query returning chunks with citations). See `TODO.md` for the full remaining work and the handoff brief for the next session.

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
5. **Every response ships with metadata.** `{answer, confidence, citations, tool_calls, memory_writes, provider, latency_ms}` — not just a string. The eval harness depends on this shape.
6. **Determinism in tests.** Provider calls in tests go through a recorded-response fake (VCR-style cassettes in `tests/cassettes/`). Never hit live APIs in CI.
7. **Secrets live in `.env`**, loaded via `pydantic-settings`. `.env.example` is committed, `.env` is gitignored. No hardcoded keys, ever.

## Eval harness

The eval suite is not optional — it's the differentiator for the job pitch. Minimum:

- `evals/scenarios.yaml` — ~30 scripted support conversations covering: grounded factual Q&A, personalization recall across sessions, off-topic refusal, low-confidence escalation triggers, adversarial prompt injection attempts.
- `evals/scorers.py` — faithfulness (every factual claim covered by a cited chunk), correctness (vs gold answer), memory-recall accuracy, escalation precision/recall.
- `evals/run.py` — runs the full matrix (every scenario × every provider), emits `evals/report.md` with a provider-comparison table.

The README's headline table comes from `evals/report.md`. If you change the harness, re-run evals and commit the new report.

## Commands

Filled in as we build. Placeholder entries:

```bash
# install
uv sync                          # or: pip install -e .

# pull local models
ollama pull gemma4               # Gemma 4 E4B
ollama pull nomic-embed-text

# seed data + embed corpus
python -m data.seed
python -m data.embed

# run
uvicorn api.server:app --reload

# tests
pytest
pytest -m "not slow"             # skip eval integration tests

# evals
python -m evals.run --providers ollama,anthropic
```

## Conventions

- **Type hints everywhere.** `mypy --strict` compatible for the core layers (harness, grounding, memory, tools, providers). API/data/evals can be looser.
- **Pydantic models at layer boundaries.** Tool inputs/outputs, API requests/responses, Message/Turn/Session — all Pydantic. No stringly-typed dicts flowing across modules.
- **Async all the way from API down to providers.** Ollama and cloud providers are I/O-bound; the harness loop is async.
- **Line length 100.** Formatter: `ruff format`. Linter: `ruff check` with a strict ruleset.
- **No comments that restate code.** Comments only for _why_ — a constraint, a workaround, a non-obvious invariant. Never a WHAT comment on a well-named function.
- **Commits: imperative mood, present tense.** "Add confidence scoring" not "Added" or "Adds".

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
