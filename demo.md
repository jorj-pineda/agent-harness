# Demo guide — coding agent

How to run the **coding-agent** harness end-to-end. The default path uses the
vendored `fixtures/tiny_repo` workspace (no support DB seed, no Chroma embed).

## Quick start (Docker — recommended)

```bash
docker compose up --build -d

# Pull chat model (embed model only needed if ENABLE_SUPPORT_TOOLS=true)
docker exec agent-harness-ollama ollama pull gemma4

# Create session — workspace defaults to /app/fixtures/tiny_repo in compose
curl -s -X POST http://localhost:8000/sessions \
  -H 'content-type: application/json' \
  -d '{"user_id":"dev1"}' | tee /tmp/session.json

SESSION_ID=$(python3 -c "import json; print(json.load(open('/tmp/session.json'))['session_id'])")

# Ask for a bugfix (model must support tool calling)
curl -s -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d "{\"user_id\":\"dev1\",\"session_id\":\"$SESSION_ID\",\"message\":\"Fix the failing divide test in test_calc.py\"}" \
  | python3 -m json.tool
```

### Reading the envelope

Every `/chat` response includes:

| Field | Meaning |
|-------|---------|
| `answer` | Final assistant text |
| `citations` | File:line keys from `read_file` / `grep_repo` (e.g. `calc.py:4-6`) |
| `confidence` | Grounding score; `null` when no evidence tools ran |
| `escalated` | `true` when confidence is low, verification missing (if required), or edit budget exceeded |
| `files_touched` | Repo paths written this turn |
| `verification_ran` | `true` after a successful allowlisted `pytest`/`ruff`/`mypy` run |
| `tool_calls` | Full tool trace |

### Hardware / memory

- **GPU laptop (8 GB VRAM):** Gemma 4 E4B with `OLLAMA_KV_CACHE_TYPE=q8_0` (set on the Ollama service).
- **Docker Desktop without GPU:** allocate **≥ 12 GB RAM** or use a cloud provider in `.env` (`DEFAULT_PROVIDER=anthropic`).

## Quick start (local, no Docker)

```bash
uv sync --extra dev
cp .env.example .env

ollama pull gemma4

uvicorn api.server:app --reload
```

Create a session with an explicit workspace:

```bash
curl -X POST http://localhost:8000/sessions \
  -H 'content-type: application/json' \
  -d "{\"user_id\":\"dev1\",\"workspace_root\":\"$(pwd)/tests/fixtures/tiny_repo\"}"
```

## Eval matrix (offline)

```bash
python -m evals.run --providers ollama,anthropic,openai
```

Scores are deterministic (scripted `FakeProvider`). Use `--live` only for real provider comparison — scores vary run-to-run.

## Optional: legacy support demo

Set `ENABLE_SUPPORT_TOOLS=true`, pull `nomic-embed-text`, then seed:

```bash
docker exec agent-harness-ollama ollama pull nomic-embed-text
docker exec agent-harness-app python -m data.seed
docker exec agent-harness-app python -m data.embed
curl -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"user_id":"u1","session_id":"<id>","message":"what is your return window?"}'
```

Support eval scenarios: `python -m evals.run --scenarios evals/scenarios_support.yaml`

## Optional: cloud providers

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`. Pass `"provider":"anthropic"` on `/chat`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Ollama 500 / OOM on `/chat` | More Docker RAM, smaller model, or cloud provider |
| `read_file` not in tool list | Session needs `workspace_root` (or `DEFAULT_WORKSPACE_ROOT`) |
| Empty citations | Model skipped read/grep tools — check `tool_calls` in response |
| `escalated: true` after edit | Edit budget (`MAX_FILES_TOUCHED_PER_TURN`, default 5) or low grounding confidence |
