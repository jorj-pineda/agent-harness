# agent-harness

A local-first, pluggable-provider agent harness for **senior-level coding tasks**, built from scratch — no LangChain, no LlamaIndex, no LangGraph. The point is the loop: a hand-written ReAct controller, a deterministic grounding layer that scores answer confidence and triggers escalation, and a per-user memory layer that persists engineering context across sessions. One process, two containers (Ollama + the FastAPI app), one `docker compose up` to demo.

The differentiating features are **grounded edits with confidence** and **cross-session repo memory**. Both are inspectable: every response ships the same envelope (`{answer, confidence, citations, escalated, tool_calls, memory_writes, files_touched, verification_ran, provider, latency_ms}`) so the eval harness can score it directly without re-prompting the model. The provider abstraction is sacred — Ollama (Gemma 4 E4B by default), Anthropic, and OpenAI all sit behind one `Provider` interface, and nothing above [providers/](providers/) imports a specific backend.

> **Pivot in progress (Phase 1).** The codebase is transitioning from a customer-support demo to a coding agent. Support tools (SQL, RAG doc search) remain registered alongside the new API shape; coding tools (read/grep/edit/verify) land in Phases 2–4. The eval table below still reflects the frozen support baseline until Phase 6 replaces scenarios.

## Architecture

```
api/            FastAPI server — thin HTTP wrapper, per-request tool registry
  └── harness/  ReAct loop, session/turn state, provider router
        ├── grounding/   confidence heuristic, citations, escalation flag
        ├── memory/      per-user FactStore (SQLite), system-prompt injection
        ├── tools/       typed registry — coding tools (Phases 2–4) + legacy support tools*
        ├── workspace/   (Phase 2+) sandboxed repo root, path jail
        ├── data/        legacy support DB + Chroma corpus (archived in Phase 7)
        └── providers/   Ollama / Anthropic / OpenAI behind one interface
```

\* Support SQL/RAG/memory tools still registered during the pivot; default demo curl may ask policy questions until [demo.md](demo.md) is updated for coding (Phase 8).

Each layer depends only on the ones below it. Model-specific quirks (Gemma 4's tool-call format vs OpenAI's function-call shape) are normalized at the provider boundary, so adding a fourth backend is a single-file change.

## What's novel

**Grounded confidence.** Every turn that retrieves evidence gets a confidence score from a deterministic heuristic — `top_score × coverage_factor × health_factor` — over the cited material. No second LLM call. During the pivot, citations transition from RAG doc chunks to **file paths and line ranges** read via code tools; the same `Grounder` interface carries through. If confidence falls below the configured threshold (default 0.55), the response is flagged `escalated=True`. Pure chitchat that never retrieved gets `confidence=null` rather than a fake number. See [harness/grounding.py](harness/grounding.py).

**Cross-session repo memory.** [memory/store.py](memory/store.py) is a SQLite-backed `FactStore` keyed by `user_id`. The memory tools are factory-bound to the request's `user_id` at registry-construction time in [api/server.py](api/server.py), so cross-user leakage is structurally impossible. Facts — conventions, stack choices, prior review notes — are injected into the system prompt at the start of each turn via `FactStore.format_for_system_prompt(user_id)`, so personalization survives across sessions without the model needing to call a tool first.

**Workspace-scoped sessions.** `POST /sessions` accepts an optional `workspace_root` (absolute path to a repo sandbox). The path is resolved and stored on the session; Phase 2+ code tools jail all file access under that root.

## Eval results

The eval harness drives [harness/loop.run_turn](harness/loop.py) directly across 30 scripted **support** scenarios (frozen baseline; coding scenarios replace these in Phase 6) spanning five categories — grounded factual Q&A, cross-session personalization recall, off-topic refusal, low-confidence escalation, and prompt-injection attempts. Every scenario × provider combination runs through scorers for faithfulness (every claim covered by a cited chunk), correctness (vs gold answer), memory recall, and escalation precision. Run with `python -m evals.run --providers ollama,anthropic,openai`; the full report writes to [evals/report.md](evals/report.md).

| Provider    | Scenarios | Faithfulness | Correctness | Memory Recall | Escalation Acc. |
|-------------|-----------|--------------|-------------|---------------|-----------------|
| `ollama`    | 30        | 1.000        | 0.497       | 1.000         | 1.000           |
| `anthropic` | 30        | 1.000        | 0.497       | 1.000         | 1.000           |
| `openai`    | 30        | 1.000        | 0.497       | 1.000         | 1.000           |

Today every provider replays the same scripted responses through a `FakeProvider` — so the columns match by construction. The point of the matrix isn't yet "which model is better"; it's that the harness produces the same shaped, scoreable envelope no matter which backend label ran the turn.

**Two layers of provider testing, intentionally separate:**

| Layer | What it exercises | Where |
|-------|-------------------|-------|
| **Eval matrix (default)** | 30 support scenarios × scorers; offline `FakeProvider` scripts from `scenarios.yaml` | `python -m evals.run --providers ollama,anthropic,openai` |
| **Provider unit tests** | Wire format (plain chat, tool call, HTTP error) per backend | `tests/cassettes/*.json` replayed in CI |
| **Live eval (optional)** | Real LLM calls; scores vary run-to-run | `python -m evals.run --live --providers ollama` (requires Ollama / API keys) |

The VCR cassettes do **not** cover eval scenario shapes — wiring them into the matrix would need ~90 scenario-specific recordings. For provider comparison against gold answers, use `--live`; for CI and the README headline table, use the default offline mode.

The 0.497 mean correctness is held down by the off-topic and prompt-injection categories, where a "good" answer is a refusal rather than a high-overlap match against a gold string. **Escalation accuracy is 100%**: every low-confidence scenario tripped the threshold and every high-confidence one did not. That's the load-bearing claim of the grounding layer, and it's the metric a support team would actually act on. (Offline eval uses threshold **0.50**; the API default is **0.55** — both produce 100% escalation accuracy on the current scenarios.)

## Run it

The full stack is two services: a local Ollama runtime and the FastAPI app. Chroma is *not* a separate service — the harness uses `chromadb.PersistentClient` (in-process), so the corpus rides on the app container's data volume rather than a Chroma server.

**Step-by-step demo walkthrough:** [demo.md](demo.md) (model pulls, memory requirements, troubleshooting).

Compose starts Ollama but **does not auto-download models** — pull `gemma4` and `nomic-embed-text` into the Ollama container after `up`. On Docker Desktop without GPU, allocate **≥ 12 GB RAM** or Gemma 4 may OOM on `/chat`.

```bash
docker compose up --build -d

# pull models into the running Ollama container (one-time per ollama_models volume)
docker exec agent-harness-ollama ollama pull gemma4
docker exec agent-harness-ollama ollama pull nomic-embed-text

# seed the support DB and embed the doc corpus (one-time per app_data volume)
docker exec agent-harness-app python -m data.seed
docker exec agent-harness-app python -m data.embed

# hit the API
curl -X POST http://localhost:8000/sessions \
  -H 'content-type: application/json' \
  -d '{"user_id":"u1","workspace_root":"/path/to/repo"}'
curl -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"user_id":"u1","session_id":"<id>","message":"what is your return window?"}'
```

`workspace_root` is optional; omit it until code tools land (Phase 2). The support demo message above still works during the pivot.

`OLLAMA_KV_CACHE_TYPE=q8_0` is set in [docker-compose.yml](docker-compose.yml) — it cuts a 32k-context KV cache from ~15 GB to ~5 GB, which is the difference between Gemma 4 E4B fitting on an 8 GB-VRAM laptop and OOM-ing. The 26B-MoE upgrade path is documented but assumes a 16 GB+ workstation. To use Anthropic or OpenAI instead, copy [.env.example](.env.example) to `.env` and fill in keys; without them the app boots Ollama-only.

For development without Docker:

```bash
pip install -e .[dev]
cp .env.example .env

ollama pull gemma4
ollama pull nomic-embed-text

python -m data.seed
python -m data.embed

uvicorn api.server:app --reload
pytest -m "not live"                            # 289 pass, 4 live skips
python -m evals.run --providers ollama,anthropic,openai   # offline; update README if metrics change
```

## What's deferred (and why)

- **LLM-judge confidence.** The deterministic heuristic is cheap and inspectable. An LLM self-assessment judge plugs in behind the same `Grounder.ground()` interface, but self-reports are systematically over-confident — validate on evals before swapping.
- **Per-sentence citation attribution.** Citations live at the turn level today (which chunks the answer drew from). Mapping individual claims to specific chunks needs a post-generation pass (NLI or a cited-output schema) — defer until the faithfulness metric rewards it.
- **Session persistence.** Sessions live in an in-memory `dict[session_id, Session]` inside the FastAPI process. Survives neither restarts nor a multi-worker deployment. Swap for Redis or a `sessions` SQLite table when the demo grows beyond a single uvicorn process.
- **Router fallback / retry across providers.** [`ProviderRouter`](harness/router.py) is a plain dispatch table. Adding failover needs real error patterns to design against; not worth speculating on shape now.
- **Answer rewriting on escalation.** `escalated=True` is a flag; the raw answer is preserved so the API layer owns presentation. Templating the handoff message is a UX decision, not a harness one.

The full deferred-vs-out-of-scope rationale lives in [CLAUDE.md](CLAUDE.md). The short version: the loop is the portfolio piece — every abstraction in this repo earns its line count by being exercised end-to-end in the eval harness.
