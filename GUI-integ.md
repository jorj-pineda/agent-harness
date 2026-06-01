# GUI integration plan — agent panel (Mission 9)

Thin **demo UI** + optional **Typer CLI** over the existing FastAPI harness. Goal: a Cursor / Claude Code–style **agent panel only** — not an IDE, not a second ReAct loop.

**Roadmap slot:** Mission 9 (after Mission 7 README polish; parallel or after FocusKPI submit).  
**Related:** Mission 10 / Slice 9c — SSE streaming for live tool cards.

---

## Why

- The harness already returns a rich envelope (`tool_calls`, `citations`, `confidence`, `files_touched`, `patch_summary`, …) — but curl/JSON hides the story.
- Hiring managers and demo videos need **visible tool traces** and grounding metadata in ~30 seconds.
- v1 deliberately skipped UI ([CLAUDE.md](CLAUDE.md) out-of-scope: full IDE). This is a **local demo shell**, same contract as `/chat`.

---

## Non-goals

| Skip | Reason |
|------|--------|
| Monaco / LSP / file tree editor | Different product; overlaps Cursor |
| Electron / Tauri desktop app | Heavy; web panel + browser is enough |
| Second agent loop in the frontend | UI calls HTTP only |
| Auth / multi-tenant | Portfolio scope |
| LangChain / Gradio / Streamlit | Framework demo aesthetic; build thin Vite panel |

---

## Architecture

```
cli/                 Typer entry: serve | chat | ui
  └── (calls uvicorn, opens browser)

api/                 FastAPI (unchanged loop)
  └── optional: mount ui/dist static files at GET /
  └── later: GET /chat/stream (SSE)

ui/                  Vite + React (or plain HTML v0)
  └── fetch POST /sessions, POST /chat
  └── render message list + tool cards + envelope rail

harness/ …           No imports from ui/ or cli/
```

**Rule:** `ui/` and `cli/` may only talk to `api/` over HTTP (or spawn uvicorn). No `import ollama` / provider SDKs in the frontend.

---

## CLI commands (target UX)

```bash
# Install (after pyproject entry point added)
uv sync --extra dev
agent-harness serve              # uvicorn api.server:app --reload
agent-harness chat               # terminal REPL → POST /chat (optional in 9a)
agent-harness ui                 # serve + open http://127.0.0.1:8000/
agent-harness ui --no-open       # serve static UI without launching browser
```

Today: `uv run uvicorn api.server:app --reload` + curl/PowerShell. CLI wraps that.

---

## Agent panel layout (MVP)

```
┌─────────────────────────────────────────────────────────────┐
│  workspace: tiny_repo ▼   provider: ollama ▼   [New session] │
├──────────────────────────────────────┬──────────────────────┤
│  Chat                                │  Envelope            │
│  ─────                               │  confidence: 1.0     │
│  User: Fix divide test…              │  escalated: false    │
│                                      │  citations: …        │
│  Assistant: …                        │  files_touched: []   │
│    ▼ list_dir  (1.2ms)               │  verification: false │
│    ▼ read_file test_calc.py          │  patch_summary: []   │
│                                      │  provider / latency  │
│  [ message input………………… ] [Send]   │                      │
└──────────────────────────────────────┴──────────────────────┘
```

**Tool cards:** expandable name, arguments (JSON), result preview, `error`, `latency_ms`.  
**Phase 1:** full turn renders after POST completes (spinner).  
**Phase 2 (9c):** cards appear live via SSE as each tool finishes.

---

## Slices (one PR each; pause for green light)

### Slice 9a — CLI shell

| Task | Detail |
|------|--------|
| Add `cli/` | Typer app; `[project.scripts] agent-harness = "cli.main:app"` in `pyproject.toml` (additive block) |
| `serve` | Wrap uvicorn with host/port from env |
| `chat` | Optional minimal REPL: create session, loop on stdin |
| Tests | Invoke Typer with `CliRunner`; no network |

**Exit:** `agent-harness serve` starts API; existing pytest green.

---

### Slice 9b — Static agent panel (one-shot `/chat`)

| Task | Detail |
|------|--------|
| Add `ui/` | Vite + React (or single `ui/static/index.html` for fastest v0) |
| Wire API | `POST /sessions`, `POST /chat`; show full envelope |
| Mount | FastAPI `StaticFiles` at `/` when `ENABLE_DEMO_UI=true` or always in dev |
| Styling | Dark theme; collapsible tool sections |

**Exit:** Browser demo shows tool trace + envelope rail; [demo.md](demo.md) documents `agent-harness ui`.

---

### Slice 9c — SSE streaming

| Task | Detail |
|------|--------|
| API | `POST /chat/stream` or SSE on existing route — emit events: `tool_start`, `tool_end`, `assistant_delta`, `turn_complete` |
| Harness | Hook in [harness/loop.py](harness/loop.py) via callback/async generator (no duplicate loop) |
| UI | Append tool cards as events arrive |

**Exit:** Live tool trace during long gemma4 turns; tests with fake provider + SSE client.

---

### Slice 9d — Polish

| Task | Detail |
|------|--------|
| Provider / workspace pickers | Dropdowns wired to session create + chat body |
| Demo video script | 3–5 min walkthrough in [demo.md](demo.md) |
| README | One paragraph + screenshot; link here |
| PowerShell | Document `agent-harness ui` alongside Invoke-RestMethod |

---

## API contract (unchanged for 9b)

Use existing [api/models.py](api/models.py) request/response shapes. UI is a consumer — do not fork the envelope.

Optional additive fields later: `Accept: text/event-stream` on chat.

---

## Portfolio pitch (one line)

> The harness exposes grounded confidence and tool traces over HTTP; the demo UI lets reviewers watch an agent turn unfold without reading raw JSON.

---

## Dependencies (additive only)

```toml
# pyproject.toml — example block label: # mission-9 cli
dependencies = [..., "typer>=0.12"]

[project.optional-dependencies]
ui = []  # ui/ uses npm separately; document in ui/README.md
```

Frontend: Node 20+, `npm create vite@latest ui -- --template react-ts` (or equivalent). CI: build `ui/dist` in optional job or document manual build for demo.

---

## Verification

| Gate | Command |
|------|---------|
| Backend unchanged | `pytest -m "not live"` (~356 tests) |
| Lint | `ruff check .` ; `mypy` on core layers |
| Manual | `agent-harness ui` → send bugfix prompt → tool cards + envelope visible |

---

## Decision log

| Date | Decision |
|------|----------|
| 2026-05 | Agent panel only (not IDE); Mission 9; plan in this file |
| 2026-05 | Phase 1 = one-shot `/chat`; SSE in 9c |
| 2026-05 | Revisit v1 “no chat UI” — demo shell OK; full IDE still out of scope |

---

## Handoff (Claude Code / new chat)

Branch suggestion: `feat/agent-panel-ui`  
Start with **Slice 9a** or **9b** depending on whether CLI or visual demo is higher priority for the application video.

Paste Mission 9 block from [COMPOSER_SUPER_PROMPT.md](COMPOSER_SUPER_PROMPT.md) when ready.
