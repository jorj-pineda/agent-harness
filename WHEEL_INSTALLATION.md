# Wheel installation validation — 2026-10-20

At runtime revision `326e69c865940419505cac9804a4769ef7ed7874`, the built wheel
installed and served the CLI/API/panel outside the checkout in fresh macOS arm64
virtual environments using Python **3.11.15 and 3.14.5**. Neither environment
enabled system site packages or inherited the development environment's packages.
This is installation/startup evidence, not release approval or coding-quality evidence.

[Structured observations](evals/results/2026-10-20-wheel-install-smoke.json) record
the wheel/sdist/constraint/lock hashes, all 86 installed distribution versions in
each environment, served route hashes, recipe/log hashes and operator corrections.
Both installs used wheel-declared dependencies constrained by the committed lock
export. This does not validate unconstrained future dependency resolution.

## Build and install into a separate environment

The following reproduces the dependency policy from a checkout with uv installed.
Choose an installed Python; automatic Python downloads were disabled in the audit.

```sh
install_dir="$(mktemp -d /tmp/agent-harness-install.XXXXXX)"
uv build --out-dir "$install_dir/dist"
uv export --locked --no-dev --no-emit-project --no-hashes \
  --format requirements-txt --output-file "$install_dir/constraints.txt"
uv venv --python 3.11 "$install_dir/venv"
uv pip install --python "$install_dir/venv/bin/python" \
  --constraint "$install_dir/constraints.txt" \
  "$install_dir/dist/agent_harness-0.1.0-py3-none-any.whl"
uv pip check --python "$install_dir/venv/bin/python"
cd "$install_dir"
"$install_dir/venv/bin/agent-harness" --help
```

The versioned filename describes the audited `0.1.0` build. Run commands from the
separate directory so checkout imports and its `.env` cannot supply the result.
The CLI's `serve` command starts the installed API, with the same explicit limits,
workspace rules and selected provider settings as a source installation. The
panel is served at `/`; its JavaScript/CSS assets are packaged at `api/ui`.
See [demo.md](demo.md) for the source-checkout coding walkthrough and budget limits.
Project check dependencies remain the owner's responsibility: this runtime-only
install does not include pytest or the project's Node dependencies. Startup success
does not prove those commands or a model task can complete.

## What was exercised

Both fresh environments passed dependency consistency checks and resolved `api`,
`cli` and `harness` inside their own environment. The actual installed console
entry point provided help and launched the API process from a separate directory
with an allowlisted environment and temporary stores. No user credentials or `.env`
were used; the configured Ollama endpoint was deliberately unreachable, and no
coding request was submitted.

HTTP requests served `/`, `/app.js`, `/memory.js`, `/style.css`, `/openapi.json`
and `/docs`. The actual CLI created an inspection-only session without a turn,
captured a manual checkpoint, inspected configuration, then read the same
checkpoint and configuration after an API process restart. The fixture's original
bytes and `0640` mode survived. SQLite contained **zero coding runs**. A separate
isolated import probe confirmed the coding API did not import Chroma; the normal
server smoke created no support storage. Every owned API completed application
shutdown, exited and released its listener. No model was loaded or called.

This manual checkpoint/restart sequence is not a demonstration of cancellation
after a partial model edit, safe coding continuation or real single-file reversal.
The panel checks cover serving assets, not a fresh interactive browser workflow.
Support retrieval startup and a full OS/Python compatibility matrix were not run.

## Setup corrections and retained evidence

The first offline installs failed because pinned Chroma (Python 3.11) and PyYAML
(Python 3.14) were unavailable in the local cache. Registry access then installed
the **same wheel and constrained versions** into each environment. These cache
misses are environmental observations, not product dependency incompatibilities.

The first Python 3.11 smoke reached asset/session/checkpoint/configuration checks,
but the operator recipe wrongly required exit code zero after SIGTERM. Uvicorn
logged completed application shutdown and then propagated SIGTERM. Its listener
was confirmed absent. The partial directory and original recipe were retained;
the corrected smoke requires exit zero or SIGTERM, a completed shutdown log and
listener absence. The full smoke was then run on both environments. No inference
or coding task was repeated or repaired.

Raw environments, artifacts, constraints, logs and recipes remain at
`~/.local/share/agent-harness/trials/wheel-install-6o2gi8i2/`. The public observations
are not a portable source/wheel bundle. Nothing was published to a package index.
No Duodoro edits, paid calls, Docker actions or existing Ollama service changes
occurred. G2–G5 and the original ten-task no-go remain unchanged; this closes
the narrow fresh lock-constrained installation observation on this host only.
