# Demo guide

How to run the agent-harness stack end-to-end for a portfolio walkthrough or local smoke test.

## Do I need to download Gemma 4 locally?

**It depends how you run the stack.**

| Setup | Who pulls models? | What you run |
|-------|-------------------|--------------|
| **Docker (recommended demo)** | The **Ollama container** — not your Mac's host Ollama | After `docker compose up`, exec into the Ollama service and pull there (see below). You do **not** need `ollama pull` on the host unless you also run Ollama outside Docker. |
| **Dev without Docker** | Your **local Ollama install** | `ollama pull gemma4` and `ollama pull nomic-embed-text` on the machine where Ollama is running. |

Docker Compose starts two services (Ollama + FastAPI app). It does **not** auto-download models. The Ollama container starts empty until you pull into its volume (`ollama_models`). Pull once; the volume keeps models across restarts.

**Two models are required:**

1. **`gemma4`** — chat / tool-calling (default `OLLAMA_MODEL`)
2. **`nomic-embed-text`** — embeddings for RAG (`OLLAMA_EMBED_MODEL`)

## Hardware / memory

- **Target:** RTX 4070 laptop, 8 GB VRAM — Gemma 4 E4B (Q4) fits with `OLLAMA_KV_CACHE_TYPE=q8_0` (already set on the Ollama service in `docker-compose.yml`).
- **Docker Desktop (Mac/Windows, CPU-only):** allocate **≥ 12 GB RAM** to Docker. Gemma 4 needs ~9.7 GiB to load; default Docker memory often causes `/chat` to return 500 from Ollama OOM. Linux + GPU passthrough or a smaller dev model (e.g. `llama3.2:1b` via `OLLAMA_MODEL` in `.env`) are fallbacks for low-memory hosts.

## Quick start (Docker)

```bash
# 1. Build and start
docker compose up --build -d

# 2. Pull models into the Ollama container (one-time per volume)
docker exec agent-harness-ollama ollama pull gemma4
docker exec agent-harness-ollama ollama pull nomic-embed-text

# 3. Seed mock support DB + embed doc corpus (one-time per app_data volume)
docker exec agent-harness-app python -m data.seed
docker exec agent-harness-app python -m data.embed

# 4. Create a session
curl -X POST http://localhost:8000/sessions \
  -H 'content-type: application/json' \
  -d '{"user_id":"u1"}'

# 5. Chat (replace <session_id> from step 4)
curl -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"user_id":"u1","session_id":"<session_id>","message":"what is your return window?"}'
```

Every `/chat` response includes the full envelope: `answer`, `confidence`, `citations`, `escalated`, `tool_calls`, `memory_writes`, `provider`, `latency_ms`.

## Quick start (no Docker)

Requires Ollama installed and running on the host.

```bash
pip install -e .[dev]
cp .env.example .env

ollama pull gemma4
ollama pull nomic-embed-text

python -m data.seed
python -m data.embed

uvicorn api.server:app --reload
```

Then use the same `curl` commands against `http://localhost:8000`.

## Optional: cloud providers

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`. The app registers whichever providers have keys. Pass `"provider":"anthropic"` or `"openai"` on `/chat` to select one; default is `ollama`.

## Data persistence

- **SQLite + Chroma + memory DB** live on the Docker volume `app_data` (`/app/data` in the app container). Re-running `seed` / `embed` after the first setup is only needed if you wipe the volume (`docker compose down -v`).
- **Ollama models** live on volume `ollama_models`. Pull once unless you remove that volume.

## Eval report (offline by default)

**Offline (CI-safe, deterministic)** — replays scripted answers; produces the README headline table:

```bash
python -m evals.run --providers ollama,anthropic,openai
```

**Live (optional)** — calls real providers; scores vary and need API keys / running Ollama:

```bash
# Ollama must be reachable (OLLAMA_HOST in .env or environment)
python -m evals.run --live --providers ollama

# Cloud keys in .env
python -m evals.run --live --providers anthropic,openai
```

Writes `evals/report.md` (gitignored). The README headline table is the committed offline snapshot; regenerate after harness changes and update README if metrics shift.

Unit-test cassettes in `tests/cassettes/` cover provider wire format only (plain / tool-call / error per backend) — not the 30 eval scenarios.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `/chat` → 500, Ollama logs mention memory | Model too large for Docker RAM | Increase Docker Desktop memory, use GPU host, or set `OLLAMA_MODEL=llama3.2:1b` in `.env` for smoke only |
| `/chat` → 404 session | Unknown or expired `session_id` | `POST /sessions` again (sessions are in-memory; lost on app restart) |
| Empty RAG answers | Corpus not embedded | Run `python -m data.embed` (or docker exec equivalent) |
| Embed fails | Missing `nomic-embed-text` | Pull embed model in Ollama (container or host) |

## Stop / reset

```bash
docker compose down          # keep volumes (data + models)
docker compose down -v       # wipe volumes — re-seed and re-pull models
```
