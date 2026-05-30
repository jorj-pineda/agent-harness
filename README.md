# agent-harness

A local-first, pluggable-provider agent harness for **senior-level coding tasks**, built from scratch — no LangChain, no LlamaIndex, no LangGraph. The point is the loop: a hand-written ReAct controller, a deterministic grounding layer that scores answer confidence and triggers escalation, and a per-user memory layer that persists engineering context across sessions. One process, two containers (Ollama + the FastAPI app), one `docker compose up` to demo.

The differentiating features are **grounded edits with confidence** and **cross-session repo memory**. Both are inspectable: every response ships the same envelope (`{answer, confidence, citations, escalated, tool_calls, memory_writes, files_touched, verification_ran, provider, latency_ms}`) so the eval harness can score it directly without re-prompting the model. The provider abstraction is sacred — Ollama (Gemma 4 E4B by default), Anthropic, and OpenAI all sit behind one `Provider` interface, and nothing above [providers/](providers/) imports a specific backend.

> **Coding-agent pivot complete (phases 1–10).** Default demo is workspace-scoped code tools on `fixtures/tiny_repo`. Legacy support tools are off unless `ENABLE_SUPPORT_TOOLS=true`.

## Architecture

```
api/            FastAPI server — thin HTTP wrapper, per-request tool registry
  └── harness/  ReAct loop, session/turn state, provider router, policy gate
        ├── grounding/   confidence heuristic, file:line citations, escalation
        ├── memory/      per-user FactStore (SQLite), system-prompt injection
        ├── tools/       read/grep/edit/verify + memory (support tools optional)
        ├── workspace/   sandboxed repo root, path jail
        └── providers/   Ollama / Anthropic / OpenAI behind one interface
```

Legacy support data (`data/seed.py`, Chroma corpus) remains for regression evals — see [evals/scenarios_support.yaml](evals/scenarios_support.yaml).

Each layer depends only on the ones below it. Model-specific quirks (Gemma 4's tool-call format vs OpenAI's function-call shape) are normalized at the provider boundary, so adding a fourth backend is a single-file change.

## What's novel

**Grounded confidence.** Every turn that retrieves evidence gets a confidence score from a deterministic heuristic — `top_score × coverage_factor × health_factor` — over the cited material. No second LLM call. During the pivot, citations transition from RAG doc chunks to **file paths and line ranges** read via code tools; the same `Grounder` interface carries through. If confidence falls below the configured threshold (default 0.55), the response is flagged `escalated=True`. Pure chitchat that never retrieved gets `confidence=null` rather than a fake number. See [harness/grounding.py](harness/grounding.py).

**Cross-session repo memory.** [memory/store.py](memory/store.py) is a SQLite-backed `FactStore` keyed by `user_id`. The memory tools are factory-bound to the request's `user_id` at registry-construction time in [api/server.py](api/server.py), so cross-user leakage is structurally impossible. Facts — conventions, stack choices, prior review notes — are injected into the system prompt at the start of each turn via `FactStore.format_for_system_prompt(user_id)`, so personalization survives across sessions without the model needing to call a tool first.

**Workspace-scoped sessions.** `POST /sessions` accepts an optional `workspace_root`. Docker sets `DEFAULT_WORKSPACE_ROOT=/app/fixtures/tiny_repo`. Code tools jail all paths under that root.

**Ripgrep-first indexing.** Codebase search uses `grep_repo` (ripgrep with Python fallback). No embed model required for the default demo. A deferred `semantic_search` stub documents the upgrade path if evals show explore failures — see [tools/semantic.py](tools/semantic.py).

**Scope gate + edit budget.** [harness/policy.py](harness/policy.py) refuses clearly unsafe/unbounded requests before the ReAct loop runs. `MAX_FILES_TOUCHED_PER_TURN` (default 5) sets `escalated=True` when a turn writes too many files — a senior-agent guardrail against drive-by refactors.

### Eval honesty

Offline eval scores are **scripted** — every provider replays the same YAML tool traces, so headline columns match by construction. They measure harness shape and scorer wiring, not model quality. Use `python -m evals.run --live --providers ollama` for real provider comparison (non-deterministic).

## Eval results

The eval harness drives [harness/loop.run_turn](harness/loop.py) directly across **28 scripted coding scenarios** spanning six categories — bugfix, feature slice, refactor, explore-only Q&A, low-confidence escalation, and unsafe-request refusal — plus archived [support scenarios](evals/scenarios_support.yaml) for regression. Every scenario × provider combination runs through scorers for code faithfulness (file:line citations), patch correctness (`files_touched`), verification (`verification_ran`), answer correctness, engineering memory recall, and escalation precision. Run with `python -m evals.run --providers ollama,anthropic,openai`; the full report writes to [evals/report.md](evals/report.md).

| Provider    | Scenarios | Code Faith. | Patch | Verification | Correctness | Memory Recall | Escalation Acc. |
|-------------|-----------|-------------|-------|--------------|-------------|---------------|-----------------|
| `ollama`    | 28        | 1.000       | 1.000 | 1.000        | 0.592       | 1.000         | 1.000           |
| `anthropic` | 28        | 1.000       | 1.000 | 1.000        | 0.592       | 1.000         | 1.000           |
| `openai`    | 28        | 1.000       | 1.000 | 1.000        | 0.592       | 1.000         | 1.000           |

Today every provider replays the same scripted responses through a `FakeProvider` — so the columns match by construction. The point of the matrix isn't yet "which model is better"; it's that the harness produces the same shaped, scoreable envelope no matter which backend label ran the turn.

**Two layers of provider testing, intentionally separate:**

| Layer | What it exercises | Where |
|-------|-------------------|-------|
| **Eval matrix (default)** | 28 coding scenarios × scorers; offline `FakeProvider` scripts from `scenarios.yaml` | `python -m evals.run --providers ollama,anthropic,openai` |
| **Provider unit tests** | Wire format (plain chat, tool call, HTTP error) per backend | `tests/cassettes/*.json` replayed in CI |
| **Live eval (optional)** | Real LLM calls; scores vary run-to-run | `python -m evals.run --live --providers ollama` (requires Ollama / API keys) |

Support baseline scenarios remain in [evals/scenarios_support.yaml](evals/scenarios_support.yaml) (`python -m evals.run --scenarios evals/scenarios_support.yaml`).

The 0.592 mean correctness is held down by refusal-style `unsafe_request` answers and terse explore-only replies where token-F1 against a longer gold string under-scores paraphrase. **Escalation accuracy is 100%**: every low-confidence scenario tripped the threshold and every high-confidence one did not. Patch and verification scores are 100% on offline scripts because bugfix/feature/refactor scenarios always script a successful `write_file` + `pytest` chain. (Offline eval uses threshold **0.50**; the API default is **0.55**.)

## Run it

Two services: Ollama + FastAPI. **Coding demo:** [demo.md](demo.md) — no seed/embed step required.

```bash
docker compose up --build -d
docker exec agent-harness-ollama ollama pull gemma4

curl -X POST http://localhost:8000/sessions \
  -H 'content-type: application/json' \
  -d '{"user_id":"dev1"}'

curl -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"user_id":"dev1","session_id":"<id>","message":"Fix the failing divide test in test_calc.py"}'
```

`DEFAULT_WORKSPACE_ROOT` in [docker-compose.yml](docker-compose.yml) points at the vendored fixture repo. Set `ENABLE_SUPPORT_TOOLS=true` and run `data.seed` / `data.embed` for the legacy support demo.

For development without Docker:

```bash
uv sync --extra dev
cp .env.example .env
ollama pull gemma4
uvicorn api.server:app --reload
pytest -m "not live"
python -m evals.run --providers ollama,anthropic,openai
```

## Reviewer checklist

```bash
uv sync --extra dev
pytest -m "not live"                    # unit + eval integration
ruff check .
mypy
python -m evals.run --providers ollama,anthropic,openai
docker compose up --build -d            # optional smoke; see demo.md
```

1. **Tests** — `pytest -m "not live"` should pass (~335 tests).
2. **Lint/types** — `ruff check .` and `mypy` on core layers.
3. **Offline evals** — matrix completes; README table matches report summary.
4. **Coding demo** — `demo.md` curl flow returns envelope with `tool_calls`, citations, confidence.
5. **Support regression (optional)** — `ENABLE_SUPPORT_TOOLS=true` + `evals/scenarios_support.yaml`.

## What's deferred (and why)

- **Semantic codebase search.** Ripgrep-first is enough for v1; `semantic_search` stub in [tools/semantic.py](tools/semantic.py).
- **LLM-judge confidence.** Deterministic heuristic is inspectable; validate before swapping.
- **Per-sentence citation attribution.** Turn-level file:line citations today.
- **Session persistence.** In-memory sessions; swap for Redis/SQLite when multi-worker.
- **Router fallback across providers.** Plain dispatch table until error patterns justify failover.
- **emit_plan tool / diff-first API UX.** Inspectable tool trace is sufficient for portfolio v1.
