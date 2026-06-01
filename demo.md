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

**Smoke-tested constraint (2026-05, Apple Silicon + Docker Desktop):** With Docker limited to ~8 GiB total, Ollama fails to load `gemma4` (`model requires more system memory (9.7 GiB) than is available`). `/sessions` succeeds but `/chat` returns HTTP 500. Fix: raise Docker Desktop **Settings → Resources → Memory** to **≥ 12 GiB**, or set `DEFAULT_PROVIDER=anthropic` / `OPENAI_API_KEY` in `.env` and pass `"provider":"anthropic"` on `/chat`.

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

## Agent panel + CLI (curl-free)

The harness ships a **Typer CLI** and a **static demo panel** so you can drive a
turn without curl. Both are thin HTTP clients — no agent logic lives in them.

### Start the server

```bash
uv sync --extra dev
agent-harness serve                 # = uvicorn api.server:app; --host/--port/--reload
```

### Browser panel

Open **http://127.0.0.1:8000/**. The panel is served from `ui/` (static mount):

1. (Optional) paste an absolute `workspace_root` and pick a provider in the top bar.
2. Click **New session**.
3. Type a coding task and press **Enter**.

Tool calls stream in **live as cards** (name, arguments, result/error, latency)
via Server-Sent Events (`GET /chat/stream`); the right-hand **envelope rail**
shows the confidence badge, escalation pill, provider, latency, citations,
`files_touched`, `patch_summary`, and `memory_writes`. If the browser can't open
an `EventSource`, the panel falls back to a single `POST /chat`.

> **Screenshot (30-second capture):** with the server running and a turn sent,
> screenshot `http://127.0.0.1:8000/` and save it to `docs/panel.png` — the
> README references that path. A populated capture needs a tool-calling provider
> (cloud key or a model that completes read→edit→verify; gemma4 may stop early on
> 8 GB VRAM — see the hardware note above).

### Terminal REPL

```bash
agent-harness chat \
  --workspace "$(pwd)/tests/fixtures/tiny_repo" \
  --provider anthropic            # omit for the configured default
```

`chat` creates a session, then loops on stdin: each line is a turn, and the reply
prints the answer plus a one-line envelope summary (confidence, escalation,
provider, latency, `files_touched`). Type `/quit` or Ctrl-D to exit.

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

See **Agent panel + CLI** above for the curl-free demo — plan in [GUI-integ.md](GUI-integ.md).

## Optional: cloud providers

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`. Pass `"provider":"anthropic"` on `/chat`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Ollama 500 / OOM on `/chat` | More Docker RAM (**≥ 12 GiB** for gemma4), smaller model, or cloud provider |
| `model requires more system memory (9.7 GiB)` in Ollama logs | Same as above — gemma4 weights alone need ~9.4 GiB on CPU |
| `read_file` not in tool list | Session needs `workspace_root` (or `DEFAULT_WORKSPACE_ROOT`) |
| Empty citations | Model skipped read/grep tools — check `tool_calls` in response |
| Model reads files then answers without editing | After merging `fix/ollama-tool-loop`, multi-turn tools work; gemma4 may still stop with text instead of `write_file` — try `"provider":"anthropic"` for edit→verify demos |
| `escalated: true` after edit | Edit budget (`MAX_FILES_TOUCHED_PER_TURN`, default 5) or low grounding confidence |
| Panel loads but Send is disabled | Click **New session** first — the input enables once a session exists |
| Panel tool cards never fill in | Stream interrupted; the panel falls back to `POST /chat`. Check the server log and that the provider supports tool calls |
