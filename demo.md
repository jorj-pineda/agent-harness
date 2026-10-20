# Local coding workflow

Use the shared CLI/API/panel runtime on a disposable trusted fixture first. This
example uses Jorge's already selected `gemma4:12b` on a local Ollama service. It
creates no hosted-model calls or premium supervisor. The recorded model results
and limits are in [evals/BASELINES.md](evals/BASELINES.md); completing this walkthrough
is not a model-quality comparison.

## Start the API with explicit limits

From the agent-harness checkout, install dependencies and ensure your existing
Ollama service has the selected model. In one terminal:

```sh
uv sync --extra dev
DEFAULT_PROVIDER=ollama OLLAMA_MODEL=gemma4:12b \
OLLAMA_NUM_CTX=16384 OLLAMA_THINK=false ENABLE_SUPPORT_TOOLS=false \
MAX_CONTEXT_TOKENS=16000 MAX_TOTAL_TOKENS_PER_TURN=64000 \
MAX_COMPLETION_TOKENS_PER_TURN=4096 MAX_TOOL_CALLS_PER_TURN=20 \
MAX_TURN_WALL_SECONDS=300 REQUEST_TIMEOUT_SECONDS=120 \
REQUIRE_VERIFICATION_BEFORE_FINISH=true \
PROJECT_CHECK_ARGV='["python","-m","pytest","-q","test_calc.py"]' \
uv run agent-harness serve
```

These are explicit example limits, not measured optimal settings. The default
prompt remains `legacy`; `CODING_PROMPT_PROFILE=compact` is opt-in with limited
paired observations. Prompt/context and total-token gates use estimates; reported
usage can exceed an estimate. Setup consumes the dispatch deadline, while owned
cleanup, final review and storage may exceed it. Thinking/context settings are
requested backend options, not backend attestation. The check command is global
server configuration: change it for a different project before submitting work.

Settings can also come from a local `.env`, which may contain credentials. Do not
commit or include that file in model-visible artifacts. No support corpus or
embedding model is needed with support tools disabled. Choose hosted endpoints
explicitly with a separate call budget; there is no automatic paid fallback.

## Open a disposable fixture

In another terminal, from the same checkout:

```sh
task_workspace="$(mktemp -d /tmp/agent-harness-demo.XXXXXX)"
cp tests/fixtures/tiny_repo/calc.py tests/fixtures/tiny_repo/test_calc.py "$task_workspace/"
uv run agent-harness chat --user-id demo-user --provider ollama \
  --workspace "$task_workspace" --checkpoints
```

Ask: `Fix divide in calc.py so the failing divide test passes. Preserve add and run
the configured check.` The copy protects the vendored fixture from demo edits.
Checkpoints opt into bounded plaintext source copies in local SQLite; keep this
disposable path until you finish inspecting or reversing its changes.

For the panel, open **http://127.0.0.1:8000/**, enter that absolute disposable path,
choose `ollama`, opt into turn checkpoints if wanted and create a new session.
The panel and CLI are thin HTTP clients of the same runtime.

## Review and recover explicitly

Review completion/check status, changed paths, bounded diffs and tool errors.
A `passed` check is observed execution evidence, not independent task acceptance.
`verification_ran` is retained as a legacy pass indicator; `confidence` is a
read/tool heuristic and cannot establish answer or patch correctness. Unavailable,
stale, failed, not-run and missing evidence stay distinct. Compare the artifact
against the requested behavior, including cases beyond the visible fixture test.

The CLI prints the session ID and submission key. `/check` inspects a pending
submission; `/retry` repeats only its exact retained input/key. New messages are
blocked after an uncertain outcome. Type `/quit` to exit. The separate
`uv run agent-harness cancel SESSION_ID --user-id demo-user` command requests
cancellation and returns before final settlement. The server waits for owned
cleanup before final reporting; it cannot promise remote inference/billing or
synchronous workers stop immediately.

The panel's **Cancel turn**, **Check run** and explicit retry controls use
the API boundaries. A stream disconnect leaves app-owned work running under its
budgets; it never triggers an automatic edit-producing POST fallback. Reload
restores pending references without submitting them. Forgetting a reference does
not cancel server work.

After an API restart, inspect **Saved runs** in the panel, or:

```sh
uv run agent-harness runs --user-id demo-user
uv run agent-harness run RUN_ID --user-id demo-user --json
uv run agent-harness memory --user-id demo-user
```

Replace `RUN_ID` with an actual printed ID. Saved history stays read-only. Explicit
continuation forks require fresh scoped workspace/configuration/transcript/context
checks and acknowledgment of uncaptured state; older/changed/unknown profiles may
require a fresh session. Explicit reversal is single-file, hash/mode guarded and
preserves other paths. Checkpoint/reversal controls and CLI commands are documented
in [README.md](README.md). Never reset a dirty repository to recover a run.

## Keep evaluation evidence separate

```sh
report_dir="$(mktemp -d /tmp/agent-harness-contracts.XXXXXX)"
uv run python -m evals.run --providers ollama,anthropic,openai \
  --report "$report_dir/coding.md"
uv run python -m evals.run --scenarios evals/scenarios_support.yaml \
  --providers ollama,anthropic,openai --report "$report_dir/support.md"
```

Those matrices use scripted responses/tools and validate mechanics. Adding
`--live` uses inference with simulated tools; it still does not produce or grade
real repository patches. Real coding evaluations use [evals/real_tasks](evals/real_tasks),
disposable workspaces and immutable external acceptance. Existing raw results are
preserved by using unique report paths. No offline test proves model improvement.

## Execution limits and troubleshooting

| Observation | Next action |
|---|---|
| Model unavailable or load failure | Inspect your selected local backend/model; stop and resolve setup before another run. |
| No file tools | Create a session with the intended absolute workspace path. |
| Budget-exhausted/incomplete | Inspect preserved diffs/errors; narrow the task or explicitly revise limits. |
| Unavailable/failed/stale check | Inspect the attempt and configured argv; rerun the relevant check after the latest edit. |
| Lost response or stream | Inspect the exact saved submission; use an explicit same-key retry only if appropriate. |
| Saved continuation blocked | Inspect review diagnostics; changed/unknown evidence does not grant permission to edit. |

Path validation and command allowlists are not process isolation. Tests/npm scripts
execute trusted repository code with host permissions. Admission coordinates one
harness process, not editors, other API workers or external writers. Checkpoints
exclude some paths/metadata and are not an atomic whole-workspace restore.
Cross-process coordination, untrusted-code isolation, full reconciliation, context
compaction, runtime model switching and broader held-out/project acceptance remain
roadmap work. Named
profiles can already be configured and selected at server startup; see the
[README profile configuration](README.md#named-startup-model-profiles).

This walkthrough assumes a source checkout with development dependencies. Wheels
now include the panel; [fresh installation observations](WHEEL_INSTALLATION.md)
cover lock-constrained runtime installs outside the checkout on macOS/Python 3.11
and 3.14. Project check dependencies still need separate installation. See the
[release-readiness audit](RELEASE_READINESS.md) for the unmet release gates.

Legacy Docker/support smoke observations remain in [evals/LIVE.md](evals/LIVE.md).
They are historical environment observations, not a recommendation to spend money
or change your running containers. This walkthrough uses neither Docker nor support
resource initialization.
