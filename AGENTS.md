# Agent instructions

## Product goal and roadmap

Build a practical coding harness for Jorge's lower-tier local and hosted models.
Improve bounded task completion through reliable tools, focused prompts, useful
context, and recovery from mistakes. A premium model must not be a required
supervisor. Model capability improvements must be measured, not assumed.

Read [roadmap.md](roadmap.md) before planning substantial work. It defines the
current direction, implementation sequence, milestone acceptance criteria, and
deferred work. Start with its **First implementation sequence** and keep progress
accurate. Complete the requested slice; do not silently expand into all milestones.

`NEXT_STEPS.md`, `CODING_AGENT_PIVOT.md`, `COMPOSER_SUPER_PROMPT.md`, and portfolio
sections in `CLAUDE.md` describe previous goals. Use them for historical context;
this file and the roadmap govern new work when that context conflicts with the
lower-tier-model direction. Explicit user instructions take precedence.

## Architecture

- `providers/`: backend adapters and provider-neutral message/tool types.
- `tools/`, `workspace/`: validated tool execution and workspace access.
- `harness/`: shared runtime configuration, prompt construction, loop, policy,
  outcomes, memory integration, and events.
- `api/`: HTTP transport and application resource lifecycle.
- `cli/`, `ui/`: clients of the API; do not duplicate agent behavior here.
- `evals/`: evaluation runners and scorers; reuse the application runtime where
  testing application behavior.
- `tests/`: deterministic tests, integration fixtures, and provider cassettes.

Keep the handwritten loop and provider abstraction. Provider-specific SDKs belong
in `providers/`. Shared harness code must not depend on HTTP handlers. Prefer
small, explicit seams over framework migrations or speculative abstractions.

## Implementation rules

- Inspect relevant code and tests before editing. Preserve unrelated user changes.
- Keep changes focused and typed. Use Pydantic at validated boundaries and match
  the existing Python style (100-column Ruff configuration, strict core typing).
- Put enforceable budgets and preconditions in code. A prompt instruction is not
  an enforcement mechanism; an after-the-fact flag is not a preventive gate.
- Do not call file-read evidence a probability of correctness. Report observed
  changes, check results, unresolved failures, and completion status honestly.
- Keep tool definitions, runtime prompts, and configuration shared between
  application and real-task evaluations. Make simulation overrides explicit.
- Preserve the existing response contract during incremental changes unless the
  requested task calls for a migration. Document intentional behavior changes.
- Keep local-only and provider-specific dependencies out of paths that do not
  need them where practical. Never print `.env` contents or credentials.
- Do not add agents, semantic search, model judges, or a larger prompt merely
  because they are available. Use observed failures and evaluations to justify them.

## Evaluations and tests

Keep these three kinds of evidence separate:

1. **Scripted contract tests:** fake model responses and simulated tool results;
   validate harness mechanics, not model quality.
2. **Live-model simulated-tool smoke tests:** real inference with canned tools;
   validate limited model/tool interaction, not working patches.
3. **Real coding evaluations:** real tools in disposable workspaces, the shared
   application runtime, and independent acceptance checks on the final artifact.

Do not score filename overlap, prose similarity, or citations as proof that a
patch works. Never let editable tests redefine authoritative acceptance checks.
Compare the same model on the same tasks with recorded budgets and configurations.
Keep network/provider tests marked `live`; do not introduce paid calls into the
default test suite. Use isolated settings and temporary storage in tests.

For code changes, run focused regression tests followed by the relevant project
checks. With dependencies installed, the normal offline gate is:

```sh
uv run pytest -m 'not live'
uv run ruff check .
uv run mypy
```

When changing eval behavior, also run both scripted scenario sets and inspect the
report labels. Use temporary report paths to avoid overwriting a user's results:

```sh
uv run python -m evals.run --providers ollama,anthropic,openai --report /tmp/agent-harness-coding-report.md
uv run python -m evals.run --scenarios evals/scenarios_support.yaml --providers ollama,anthropic,openai --report /tmp/agent-harness-support-report.md
```

If using `.venv/bin/pytest` directly, put `.venv/bin` on `PATH` as well: integration
tests launch `pytest` as a subprocess. Report environmental failures separately
from product regressions. Documentation-only edits do not require the test suite.

## Execution and user work

Path validation and command allowlists are not process isolation. Preserve that
distinction in implementation and documentation. Use disposable workspaces for
real-model evaluation, never the user's working repository as an eval fixture.
Manage subprocess cancellation explicitly; timing out an await does not terminate
the underlying process. Never automatically reset a dirty repository or discard
user changes. Keep credentials out of model-visible traces and test artifacts.

## Reporting progress

### Commits and pull requests

- Make frequent, small, cohesive commits so the history stays easy to review.
- Jorge is the sole commit author. Use his configured Git identity; do not add
  Codex or another assistant as an author or co-author.
- Codex assistance may be acknowledged in the PR description.
- When asked to open a PR, push the branch and create the PR, but do not merge it.
  Jorge merges his own PRs.

Explain what changed, why it matters to lower-tier-model usability, what was
validated, and what remains incomplete. Update roadmap progress only for work
actually completed. Keep historical results labeled with their original runtime
and evaluation limitations. Do not claim model-quality improvements from offline
tests alone. Do not commit, publish, or run a paid comparison unless requested.
