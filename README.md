# agent-harness

**Current direction:** a practical coding harness for lower-tier local and hosted
models. See [roadmap.md](roadmap.md) for the implementation plan and
[AGENTS.md](AGENTS.md) for contributor instructions. The portfolio description and
historical results below describe the project's starting point, not demonstrated
improvements in model coding ability.

[![CI](https://github.com/jorj-pineda/agent-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/jorj-pineda/agent-harness/actions/workflows/ci.yml)

Most agent tutorials stop at `LangChain.AgentExecutor`. **agent-harness** is the opposite: a hand-written ReAct loop, grounding layer, and memory store you can read in an afternoon — built to show how a **senior coding agent** is wired, not how to import a framework. No LangChain, LlamaIndex, or LangGraph. One FastAPI process, pluggable providers (Ollama / Anthropic / OpenAI), workspace-scoped code tools, and an eval harness that scores every turn from the same metadata envelope.

Three ideas carry the portfolio story. **Grounded confidence:** every evidence-backed turn gets a deterministic score and file:line citations; below threshold → `escalated=true` without a second LLM judge. **Cross-session repo memory:** SQLite facts per `user_id` injected into the system prompt so conventions survive across sessions. **Eval honesty:** offline scores replay scripted tool traces (30 scenarios); live Ollama runs are documented separately — the README table measures harness shape, not which model wins.

**Run it in five minutes:** `docker compose up --build -d`, `ollama pull gemma4`, then `POST /sessions` + `POST /chat` against the vendored `tiny_repo` fixture. Step-by-step curls, envelope field guide, and Windows PowerShell notes: [demo.md](demo.md). Live provider behavior (Mac Docker OOM, 4070 gemma4 multi-turn validation): [evals/LIVE.md](evals/LIVE.md).

> **Coding-agent pivot complete (phases 1–10, PR #13).** Default demo: workspace code tools on `fixtures/tiny_repo`. Legacy support tools off unless `ENABLE_SUPPORT_TOOLS=true`.

## Architecture

```
api/            FastAPI server — thin HTTP wrapper, per-request tool registry
  └── harness/  ReAct loop, session/turn state, provider router, policy gate
        ├── grounding/   confidence heuristic, file:line citations, escalation
        ├── memory/      per-user FactStore (SQLite), system-prompt injection
        ├── tools/       read/grep/edit/verify + memory (support tools optional)
        ├── workspace/   repo path guard and disposable workspace copies
        └── providers/   Ollama / Anthropic / OpenAI behind one interface
```

Legacy support data (`data/seed.py`, Chroma corpus) remains for regression evals — see [evals/scenarios_support.yaml](evals/scenarios_support.yaml).

Each layer depends only on the ones below it. Model-specific quirks (Gemma 4's tool-call format vs OpenAI's function-call shape) are normalized at the provider boundary, so adding a fourth backend is a single-file change.

## What's novel

Every response ships one envelope — `{answer, confidence, citations, escalated, tool_calls, memory_writes, files_touched, verification_ran, patch_summary, provider, latency_ms}` — so consumers and eval scorers never re-parse prose.

**Grounded confidence.** [harness/grounding.py](harness/grounding.py) scores evidence turns with `top_score × coverage × health` over cited file spans. Pure chitchat → `confidence=null`. Threshold breach → `escalated=true`.

**Cross-session repo memory.** [memory/store.py](memory/store.py) persists facts per `user_id`; [api/server.py](api/server.py) injects them at turn start. Memory tools are factory-bound to the session user — no cross-user leakage.

**Workspace path guard + policy.** Code tools resolve paths under `workspace_root`; [harness/policy.py](harness/policy.py) classifies task kind and flags unsafe scope; `MAX_FILES_TOUCHED_PER_TURN` flags oversized edits after the turn. These controls do not isolate code executed by tests.

**Ripgrep-first search.** Default demo needs no embed model; deferred semantic path in [tools/semantic.py](tools/semantic.py).

**Targeted edits.** `read_file` now returns a whole-file `sha256`, including when
reading a small range of a large file. `replace_text` takes that hash and one
exact `old_text` span, refuses stale or ambiguous edits, and returns a diff.
`write_file` remains available for new files or deliberate full replacements.
File-tool edit metadata includes either operation. These mechanics have offline
regression coverage; model-level success still requires real coding evaluations.

**Managed commands and disposable copies.** Code-tool commands now run as async
processes with bounded output. Timeouts and cancellation terminate their process
group on POSIX. Command results report when stdout or stderr was truncated.
`workspace.disposable_workspace` copies a trusted fixture and compares filesystem
snapshots to find edits, new files, and deletions, including changes made by tests.
It excludes `.env` files, and commands receive a small default environment rather
than provider API keys. The five-task real evaluator uses this copy with actual
coding tools and external acceptance checks. A copy is not a security boundary:
executed code can still access the host filesystem and network. Do not run
unfamiliar repository code with this helper until an isolated execution mode exists.

**Planning + patch trace.** `emit_plan` records steps before edits;
`patch_summary` lists successful writes. When enabled,
`REQUIRE_PLAN_BEFORE_EDIT` blocks premature file edits, and
`REQUIRE_VERIFICATION_BEFORE_FINISH` gives an edited turn one bounded chance to
run a fresh check. The API response adds `verification_status`,
`completion_status`, and `completion_reason`; the older `verification_ran` flag
remains for compatibility. File count limits reject additional file-tool edits
before mutation. A passing check becomes stale after a later edit, and
version/help commands do not count as checks. Recognized checks are currently
pytest, Ruff check, and mypy. Set `PROJECT_CHECK_ARGV='["pytest","-q"]'`
and `REQUIRE_VERIFICATION_BEFORE_FINISH=true` to require that exact command
after the latest edit. The runtime includes it in the system message, and the
same setting applies to the API and real-task evaluator. Without an explicit
project check, the previous recognized-command heuristic remains in use; a
passing unrelated command may therefore appear verified.

The configured runtime also caps tool attempts per turn with
`MAX_TOOL_CALLS_PER_TURN` and stops before repeating a tool call whose previous
`MAX_IDENTICAL_TOOL_CALLS` consecutive outcomes were unchanged. Rejected calls
remain in the trace with an error, and the response reports `budget_exhausted`
or `blocked` with a reason. These limits prevent further tool dispatch; they do
not yet impose a total wall-time or model-token budget.

**Local agent panel.** A Typer CLI (`agent-harness serve`/`chat`) and a zero-build static panel (`ui/`) make the envelope legible — tool cards stream in live over SSE (`GET /chat/stream`). Both are thin HTTP clients; the ReAct loop is never duplicated in the frontend.

### Eval honesty

The legacy scenario runner has two modes, neither of which executes real coding
tasks:

- **`scripted-contract`** (default): fake model responses and simulated tools
  exercise the low-level loop. Historical scenario behavior is preserved.
- **`live-model-simulated-tools`** (`--live`): real inference uses the shared
  application prompt, coding tool definitions, memory injection, and policy
  settings, but file reads, edits, and commands still return canned results.
  Unscripted coding calls fail explicitly; they never execute on disk.

The shared setup lives in `harness/runtime.py`, `harness/prompts.py`, and
`harness/config.py`; the CLI uses it through the API. Live smoke tests now use the
application escalation threshold unless `--escalation-threshold` overrides it.
Offline contracts retain their 0.50 default. Historical live results predate this
shared setup and should not be compared directly with new runs.

The application and real-task evaluator also share chat provider configuration.
`OLLAMA_HOST` and `OLLAMA_MODEL` select a local model. For an OpenAI-compatible
chat endpoint, set `OPENAI_COMPATIBLE_BASE_URL` and
`OPENAI_COMPATIBLE_MODEL`, plus `OPENAI_COMPATIBLE_API_KEY` if required; then
set `DEFAULT_PROVIDER=openai_compatible` for the app or pass
`--provider openai_compatible` to the evaluator. Endpoint URLs must not contain
credentials. This path uses chat completions with function tools; endpoint
compatibility should be checked with a real model before relying on it.

Reports identify these modes and explain the proxy metrics: expected-path recall
is not patch correctness, and a scripted verification flag is not a passing test
suite. The separate [five-task real evaluator](evals/real_tasks/README.md) runs
the shared application runtime with real tools on trusted disposable fixtures,
then grades the final files with external acceptance tests. It has a same-model
minimal-prompt mode. In the first local smoke comparison, `gemma4:12b` passed
`divide_zero` once in each mode; this does not establish an overall advantage.

Offline eval scores are **scripted** — every provider replays the same YAML tool traces, so headline columns match by construction. They measure harness shape, not model quality. Live runs: `python -m evals.run --live --providers ollama` and [evals/LIVE.md](evals/LIVE.md).

## Eval results

The eval harness drives [harness/loop.run_turn](harness/loop.py) directly across **30 scripted coding scenarios** spanning six categories — bugfix, feature slice, refactor, explore-only Q&A, low-confidence escalation, and unsafe-request refusal — plus archived [support scenarios](evals/scenarios_support.yaml) for regression. Every scenario × provider combination runs through scorers for code faithfulness (file:line citations), patch correctness (`files_touched`), verification (`verification_ran`), answer correctness, engineering memory recall, and escalation precision. Run with `python -m evals.run --providers ollama,anthropic,openai`; the full report writes to [evals/report.md](evals/report.md).

| Provider    | Scenarios | Code Faith. | Patch | Verification | Correctness | Memory Recall | Escalation Acc. |
|-------------|-----------|-------------|-------|--------------|-------------|---------------|-----------------|
| `ollama`    | 30        | 1.000       | 1.000 | 1.000        | 0.592       | 1.000         | 1.000           |
| `anthropic` | 30        | 1.000       | 1.000 | 1.000        | 0.592       | 1.000         | 1.000           |
| `openai`    | 30        | 1.000       | 1.000 | 1.000        | 0.592       | 1.000         | 1.000           |

Today every provider replays the same scripted responses through a `FakeProvider` — so the columns match by construction. The point of the matrix isn't yet "which model is better"; it's that the harness produces the same shaped, scoreable envelope no matter which backend label ran the turn.

**Two layers of provider testing, intentionally separate:**

| Layer | What it exercises | Where |
|-------|-------------------|-------|
| **Eval matrix (default)** | 30 coding scenarios × scorers; offline `FakeProvider` scripts from `scenarios.yaml` | `python -m evals.run --providers ollama,anthropic,openai` |
| **Provider unit tests** | Wire format (plain chat, tool call, HTTP error) per backend | `tests/cassettes/*.json` replayed in CI |
| **Live eval (optional)** | Real LLM calls; scores vary run-to-run | `python -m evals.run --live --providers ollama` — see [evals/LIVE.md](evals/LIVE.md) |

Support baseline scenarios remain in [evals/scenarios_support.yaml](evals/scenarios_support.yaml) (`python -m evals.run --scenarios evals/scenarios_support.yaml`).

The 0.592 mean correctness is held down by refusal-style `unsafe_request` answers and terse explore-only replies where token-F1 against a longer gold string under-scores paraphrase. **Escalation accuracy is 100%**: every low-confidence scenario tripped the threshold and every high-confidence one did not. Patch and verification scores are 100% on offline scripts because bugfix/feature/refactor scenarios always script a successful `write_file` + `pytest` chain. (Offline eval uses threshold **0.50**; the API default is **0.55**.)

### Live snapshot (2026-05-31)

Not comparable to the offline table — real Ollama, non-deterministic. Mac Docker: `gemma4` OOM → `llama3.2:1b` fallback (3-scenario smoke). 4070 laptop: native `gemma4:e4b` completes multi-turn tool chains after the Ollama `tool_name` wire fix; full read→write→pytest still depends on model choice. Details: [evals/LIVE.md](evals/LIVE.md).

| Provider | Scenarios | Code Faith. | Patch | Verification | Correctness | Escalation Acc. |
|----------|-----------|-------------|-------|--------------|-------------|-----------------|
| `ollama` (live) | 3 | 0.333 | 0.667 | 0.667 | 0.131 | 1.000 |

Escalation wiring held; patch/faithfulness dropped because the fallback model skipped or mishandled tool calls on bugfix/explore scenarios.

## Run it

Full walkthrough: [demo.md](demo.md) (envelope fields, hardware notes, PowerShell curls).

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

Local dev (no Docker):

```bash
uv sync --extra dev
cp .env.example .env   # set DEFAULT_WORKSPACE_ROOT to tests/fixtures/tiny_repo
ollama pull gemma4
uvicorn api.server:app --reload
pytest -m "not live"
python -m evals.run --providers ollama,anthropic,openai
```

### Local agent panel

Prefer not to read raw JSON? Start the server and open **http://127.0.0.1:8000/** —
or drive it from the terminal. Full walkthrough in [demo.md](demo.md#agent-panel--cli-curl-free).

```bash
agent-harness serve                                   # browser panel at /
agent-harness chat --workspace "$(pwd)/tests/fixtures/tiny_repo"   # terminal REPL
```

The panel calls the same `/sessions` + `/chat` API; tool calls stream in live as
cards over SSE (`GET /chat/stream`), with the response envelope on a side rail.

![agent-harness panel after a bugfix turn — live tool cards (read_file, write_file, run_command/pytest), the answer, and the grounding envelope rail (confidence 1.00, verified, citations, files touched, patch summary).](docs/panel.png)

_Capture above is offline-deterministic — the `scripted` provider chip in the rail
is the test `FakeProvider`; the workspace edit and pytest run are real. Reproduce
or refresh it via the recipe in [docs/README.md](docs/README.md)._

## Reviewer checklist

CI runs the same offline gate on every push/PR ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)): pytest, ruff, mypy, coding eval matrix, and support scenario regression — no live providers.

```bash
uv sync --extra dev
pytest -m "not live"                    # unit + eval integration
ruff check .
mypy
python -m evals.run --providers ollama,anthropic,openai
docker compose up --build -d            # optional smoke; see demo.md
```

1. **Tests** — `pytest -m "not live"` should pass (381 tests; 5 live tests deselected in CI).
2. **Lint/types** — `ruff check .` and `mypy` on core layers.
3. **Offline evals** — matrix completes; README table matches report summary.
4. **Coding demo** — `demo.md` curl flow returns envelope with `tool_calls`, citations, confidence.
5. **Support regression (optional)** — `ENABLE_SUPPORT_TOOLS=true` + `evals/scenarios_support.yaml`.

## What's deferred (and why)

- **Semantic codebase search.** Ripgrep-first is enough for v1; Mission 8 / [tools/semantic.py](tools/semantic.py) when explore evals fail grep-only.
- **Agent panel demo UI.** Mission 9 — Typer CLI + local web panel over `/chat`; plan in [GUI-integ.md](GUI-integ.md). Not a full IDE.
- **Streaming `/chat` (SSE).** Slice 9c; live tool-trace in the agent panel.
- **Session persistence.** In-memory sessions; swap for Redis/SQLite when multi-worker or restart-safe demos matter.
- **LLM-judge confidence.** Deterministic heuristic is inspectable; validate before swapping.
- **Per-sentence citation attribution.** Turn-level file:line citations today.
- **Router fallback across providers.** Plain dispatch table until error patterns justify failover.
