# agent-harness

A local-first, pluggable-provider agent harness for enterprise customer support, built from scratch — no LangChain, no LlamaIndex, no LangGraph. The point is the loop: a hand-written ReAct controller, a deterministic grounding layer that scores answer confidence and triggers escalation, and a per-user memory layer that persists facts across sessions. One process, two containers (Ollama + the FastAPI app), one `docker compose up` to demo.

The differentiating features are **grounded answers with confidence scoring** and **cross-session personalization memory**. Both are inspectable: every response ships the same envelope (`{answer, confidence, citations, escalated, tool_calls, memory_writes, provider, latency_ms}`) so the eval harness can score it directly without re-prompting the model. The provider abstraction is sacred — Ollama (Gemma 4 E4B by default), Anthropic, and OpenAI all sit behind one `Provider` interface, and nothing above [providers/](providers/) imports a specific backend.

## Architecture

```
api/            FastAPI server — thin HTTP wrapper, per-request tool registry
  └── harness/  ReAct loop, session/turn state, provider router
        ├── grounding/   confidence heuristic, citations, escalation flag
        ├── memory/      per-user FactStore (SQLite), system-prompt injection
        ├── tools/       typed registry + SQL (read-only) + RAG + memory tools
        ├── data/        SQLite mock support DB + Chroma-embedded doc corpus
        └── providers/   Ollama / Anthropic / OpenAI behind one interface
```

Each layer depends only on the ones below it. Model-specific quirks (Gemma 4's tool-call format vs OpenAI's function-call shape) are normalized at the provider boundary, so adding a fourth backend is a single-file change.

## What's novel

**Grounded confidence.** Every turn that uses [search_docs](tools/rag.py) gets a confidence score from a deterministic heuristic — `top_score * coverage_factor * health_factor` — over the retrieved chunks. No second LLM call. If confidence falls below the configured threshold (default 0.55), the response is flagged `escalated=True` and the API layer can route it to a human. Pure chitchat that never retrieved gets `confidence=null` rather than a fake number, so the eval harness can distinguish "the model was unsure" from "this question didn't need grounding." See [harness/grounding.py](harness/grounding.py) for the full heuristic and the deferred upgrade path (LLM-judge confidence, per-sentence attribution).

**Per-user memory.** [memory/store.py](memory/store.py) is a SQLite-backed `FactStore` keyed by `user_id`. The `remember_fact` and `recall_facts` tools are factory-bound to the request's `user_id` at registry-construction time in [api/server.py](api/server.py), so cross-user leakage is structurally impossible — there is no code path where a tool call could read another user's facts even if the model tried. Facts are injected into the system prompt at the start of each turn via `FactStore.format_for_system_prompt(user_id)`, so personalization survives across sessions without the model needing to call a tool first.

## Eval results

The eval harness drives [harness/loop.run_turn](harness/loop.py) directly across 30 scripted scenarios spanning five categories — grounded factual Q&A, cross-session personalization recall, off-topic refusal, low-confidence escalation, and prompt-injection attempts. Every scenario × provider combination runs through scorers for faithfulness (every claim covered by a cited chunk), correctness (vs gold answer), memory recall, and escalation precision. Run with `python -m evals.run --providers ollama,anthropic,openai`; the full report writes to [evals/report.md](evals/report.md).

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
  -d '{"user_id":"u1"}'
curl -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"user_id":"u1","session_id":"<id>","message":"what is your return window?"}'
```

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
