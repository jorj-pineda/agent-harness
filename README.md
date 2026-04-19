# agent-harness

Local-first, pluggable-provider agent harness built from scratch (no LangChain / LlamaIndex) to demonstrate agent internals end-to-end. Optimized for enterprise customer support via **grounded answers with confidence scoring** and **cross-session personalization memory**.

Status: pre-scaffold. See [CLAUDE.md](CLAUDE.md) for architecture and locked decisions.

## Quick start

```bash
uv sync
cp .env.example .env

ollama pull gemma4
ollama pull nomic-embed-text

python -m data.seed
python -m data.embed

uvicorn api.server:app --reload
```

## Layout

- `api/` — FastAPI server, thin HTTP wrapper
- `harness/` — ReAct loop, state, router
- `grounding/` — confidence scoring, citations, escalation
- `memory/` — short-term window, summarizer, long-term per-user facts
- `tools/` — typed tool registry + SQL + RAG + memory tools
- `providers/` — Ollama / Anthropic / OpenAI behind one interface
- `data/` — SQLite mock support DB + embedded doc corpus
- `evals/` — scenarios, scorers, provider-comparison report
- `tests/` — pytest + VCR-style provider cassettes

## Evaluation

`evals/report.md` is the source for the headline provider-comparison table. Re-run after any harness change:

```bash
python -m evals.run --providers ollama,anthropic
```
