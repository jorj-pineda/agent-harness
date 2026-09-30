# Roadmap: a coding harness for lower-tier models

Updated: 2026-09-29

## Direction

Build a coding harness Jorge can use on real repositories with affordable hosted models and locally runnable models that struggle with coding-agent workflows. The harness should help those models inspect the right code, make focused edits, recover from errors, and verify their work within an explicit budget.

This changes the project's original emphasis. A portfolio demonstration of agent architecture is no longer the destination. Frontier-model harnesses are not the primary target: Jorge already finds those models sufficiently capable, and access to their surrounding products can be restrictive or expensive. The opportunity here is to improve the usefulness of models that need more assistance.

“Lower-tier” describes observed task performance, not a parameter-count cutoff or a permanent list of model names. Local small models and less reliable hosted models may need different profiles. Support configurable endpoints and evaluate exact model identifiers rather than baking today's model choices into the architecture.

**Working hypothesis:** a compact coding prompt, reliable tools, controlled context, and targeted recovery can improve a weaker model's task success enough to make it useful for everyday bounded coding work. This is a hypothesis to measure, not an established result.

This document is the forward-looking roadmap for that goal. `NEXT_STEPS.md`, `CODING_AGENT_PIVOT.md`, and `COMPOSER_SUPER_PROMPT.md` remain historical context; their portfolio milestones do not determine the new implementation order.

## What usable means

The first useful version should let Jorge:

1. Select a configured model and open a repository without editing harness source code.
2. Ask for a bounded bugfix, small feature, focused refactor, or code explanation.
3. Let the agent inspect and edit code without manually correcting every tool call.
4. See the actual diff, relevant check results, unresolved problems, and resource usage.
5. Interrupt a run, preserve completed work, and resume after a restart.
6. Receive an honest blocked or incomplete result when the model cannot finish.

Initial scope is single-user, local-first, and one active writer per workspace. Python is the first fully verified language because the current tools and fixtures already support it. Add the next language and its project commands when Jorge selects a real repository that needs them.

The harness does not need to make every model good at every task. It needs to identify a useful task envelope for supported models and improve reliability inside it.

## Starting point

### Implementation progress

- **2026-09-25 — First implementation slice completed:** added root `AGENTS.md`;
  extracted shared settings, versioned prompt, coding-tool registry construction,
  memory injection, and configured turn execution into `harness/`. The API and
  live-model smoke evaluations use this setup. Offline scenario contracts retain
  their historical low-level behavior. Reports now distinguish `scripted-contract`
  from `live-model-simulated-tools` and explain their metric limitations.
- **2026-09-25 — Second implementation slice prepared:** single-file ripgrep and
  option-like patterns now return usable matches; bounded `read_file` ranges work
  on large files and include a whole-file hash. `replace_text` applies one exact
  match only when the file still has that hash, returns a diff, and contributes to
  files touched and patch summaries. This covers item 2 in the implementation
  sequence. Ignored-path parity, explicit search truncation metadata, and managed
  command execution remain Milestone 2 work.
- **2026-09-26 — Third implementation slice prepared:** code-tool subprocesses
  now use async managed processes with bounded captured output, per-command
  deadlines, and process-group termination on POSIX when timed out or cancelled.
  A disposable workspace helper copies trusted fixtures and reports added,
  modified, and deleted files, including files created by commands. The copy
  omits `.env` files and passes only a small default command environment. The
  five-task real evaluator now uses this helper. This is not OS-level isolation
  for unfamiliar repository code, and the API still lacks a run cancellation
  endpoint.
- **2026-09-26 — Fourth implementation slice prepared:** five trusted Python
  bugfix fixtures now have independent acceptance tests and checked reference
  solutions. A separate real-task runner uses disposable copies, the application
  runtime and real coding tools, then grades final files. It records traces,
  diffs, checks, usage, and latency, with a same-model minimal-prompt mode. No
  live model comparison has been run; the milestone's model baseline remains open.
- **2026-09-28 — Fifth implementation slice prepared:** application and real-task
  evaluation now construct chat providers from one settings path. A configured
  OpenAI-compatible URL/model/key can be selected beside Ollama and dedicated
  adapters. Local `gemma4:12b` passed the `divide_zero` task once in harness
  mode and once in minimal-prompt mode, on the same starting revision and
  runtime budget. Each used four tools; the agent did not run a check, but the
  external acceptance check passed. These are two smoke observations, not a
  measured improvement. A full five-task, repeated baseline remains open.
- **2026-09-28 — Sixth implementation slice prepared:** configured plan and
  file-count gates now reject disallowed file-tool edits before mutation.
  Verification status follows the latest recognized check after the latest
  file-tool edit; help/version commands cannot certify a task. An editing turn
  can receive one bounded corrective prompt before an incomplete result. The
  response now reports completion and verification status while retaining old
  fields. Scripted tests cover these mechanics; no model-quality gain is claimed.
- **2026-09-28 — Item 6 follow-up prepared:** an exact project check argv can now
  be configured in shared settings, shown to the model, and used to decide
  post-edit verification in both the application and real-task evaluator.
  Unrelated passing commands cannot satisfy that configured check. Without an
  explicit command, the historical recognized-command heuristic remains. This
  is one check command per run, not automatic project detection.
- **2026-09-28 — Item 6 tool-budget slice prepared:** the configured runtime
  caps tool dispatch across a turn and stops before an unchanged identical
  call repeats again. Rejected batched calls receive tool errors, and the
  terminal status and real-task termination label record why execution stopped.
  This prevents further tool execution without claiming a hard wall-time or
  token limit; model-level gains have not been measured.
- **2026-09-28 — Item 6 partial-work reporting prepared:** responses now expose
  ordered verification attempts, configured-check relevance, stale check
  evidence, and tool errors. The CLI and panel surface these with completion
  status and tool-reported edits. This is observed turn evidence, not a final
  filesystem diff or independent acceptance result.
- **2026-09-28 — Item 6 wall-budget slice prepared:** an optional per-turn
  deadline cancels in-flight model requests and blocks new tool dispatch once
  elapsed. Completed tools and partial work remain visible. In-flight tools
  finish normally, so this is a dispatch deadline rather than a strict bound
  on wall latency or process isolation.
- **2026-09-28 — Item 6 output-token slice prepared:** the runtime passes a
  remaining output allowance to providers, stops further work when reported
  usage reaches it, and fails closed if output usage is unavailable. Responses
  show observed prompt/output totals. This is not a strict total model-token
  cap because prompt usage arrives after each request.
- **2026-09-29 — Item 6 changed-file slice prepared:** the shared runtime hashes
  the workspace before and after each turn and reports added, modified, and
  deleted paths, including command side effects. Edits present before the turn
  are excluded unless changed again. Concurrent edits by other processes cannot
  be distinguished, ignored directories are not tracked, and oversized
  workspaces report the result as unavailable. Paths only; diffs, checkpoints,
  and revert remain open.
- **2026-09-29 — Item 6 context/total-token slice prepared:** optional context-window
  and per-turn total-token budgets are checked before every model request with
  an estimated prompt size, cap the requested output, and block tool dispatch
  when no follow-up request could fit. Reported usage stays authoritative, and
  missing usage fails closed. On one local `gemma4:12b` first request, the
  estimate was 2,950 tokens against 1,781 reported. This is an estimate-based
  gate, not a strict cap. Exhausting the context stops the turn without
  compaction.
- **2026-09-29 — Item 7 baseline recorded (Milestone 1 done-when met):** the
  real-task runner now repeats attempts across both prompt modes and writes
  per-mode raw counts with the harness revision. Local `gemma4:12b` completed
  30 real attempts (5 tasks × 2 modes × 3): harness 4/15, minimal 6/15
  accepted. Seventeen attempts ended when output filled the 4,096-token
  Ollama context the harness never configured. Two failed on literal `\n` in
  `replace_text` arguments. See `evals/BASELINES.md`; the prompt difference is
  not significant.
- **2026-09-30 — Item 7 first experiments:** `OLLAMA_NUM_CTX` and
  `OLLAMA_THINK` are now optional provider settings, recorded in real-task
  runtime config, and the runner writes its report after every attempt. With a
  16K context, hidden reasoning ran into request timeouts instead of the context
  limit (run stopped early). With thinking off, one variable from the baseline,
  `gemma4:12b` went from 4/15 to 12/15 in harness mode and from 6/15 to 9/15 in
  minimal mode, at about a third of the latency. Repeats were nearly identical
  at temperature 0, so each cell is close to one sample. The remaining
  failures are `python3` commands rejected by the allowlist, escaped newlines in
  `replace_text` arguments, and one task's reasoning. No attempt ran a passing
  check. See `evals/BASELINES.md`.
- **Next:** item 6's listed mechanics are in place; see the known limitations
  below before relying on them unattended. Item 7 starts with the repeated
  five-task baseline on the selected model. Model profiles and context
  management remain open. Offline checks alone are not evidence of better model
  coding performance.

### Review baseline

The existing foundation is worth keeping:

- A small handwritten ReAct loop, typed tool registry, and provider abstraction.
- Workspace-aware file tools and structured tool traces.
- FastAPI, CLI, and a browser panel sharing the same execution path.
- SQLite memory and offline provider-format tests.
- Real filesystem integration tests alongside scripted scenario tests.

Review baseline: 381 offline tests passed, five live tests were deselected, and Ruff and mypy passed. No live model-quality comparison was performed during that review.

Important limitations found in the current implementation:

| Area | Current behavior | Consequence |
|---|---|---|
| Evaluations | `--live` uses a real model with scripted code-tool results and omits the application's system prompt | It does not measure actual coding success or the production harness |
| Patch scoring | Checks expected filenames in `files_touched` | An incorrect patch can receive full credit |
| Completion | A response without tool calls ends the loop | Premature answers are accepted |
| Policy | Plan, verification, and edit-budget checks run after execution | Problems are flagged after the opportunity to prevent or repair them |
| Verification | Any successful recognized executable can count, even before subsequent edits | `pytest --version` followed by an edit can appear verified |
| Confidence | File reads get a fixed evidence score; the answer is not inspected | An unrelated read can produce confidence 1.0 for an unsupported answer |
| Editing | Whole-file replacement is the only edit operation | Small fixes require unnecessary generation and risk unrelated changes |
| File tools | Single-file ripgrep output is parsed incorrectly; large files are rejected before line slicing | Valid exploration attempts can fail or return misleading results |
| Execution | Registry timeout does not terminate the underlying worker-thread subprocess | Work can continue after the model is told it timed out |
| Context | Full conversation grows without a token budget or compaction | Longer work has no controlled context strategy |
| Memory | Repo conventions are stored by user only | Notes from unrelated repositories mix together |
| Isolation | File-path checks and command allowlists are treated as a sandbox | Repository tests still execute with host-process permissions |

Relevant implementation: `harness/loop.py`, `harness/outcome.py`, `harness/grounding.py`, `tools/code.py`, `tools/registry.py`, `api/server.py`, `evals/run.py`, and `evals/scorers.py`.

## Principles

- **Measure model outcomes.** Preserve deterministic harness tests, but do not use scripted success as evidence of model capability.
- **Make tools easy to use correctly.** Fix misleading errors and brittle interfaces before compensating with more instructions.
- **Put enforceable rules in code.** Prompts explain the workflow; the runtime enforces budgets, edit preconditions, and completion requirements.
- **Use a compact default prompt.** Add instructions or examples only when a measured failure justifies them.
- **Recover specifically and within limits.** Give actionable feedback for the actual failure, with a retry budget and a clear stopping condition.
- **Preserve user work.** Checkpoints, diffs, and rollback must distinguish pre-existing edits from agent changes.
- **Keep the model replaceable.** Neither normal operation nor evaluation should require a premium model as a hidden supervisor.
- **Keep complexity earned.** Semantic search, extra agents, and elaborate planning enter only after simpler approaches have measurable limits.

## Milestone 1 — Establish a real task-success baseline

**Outcome:** a model attempts real repository tasks through the same runtime used by Jorge, and independent checks determine whether it succeeded.

- [x] Extract shared prompt, coding-tool definitions, memory injection, and policy/configuration setup for the API/CLI backend and live-model smoke runner. Model profiles remain part of Milestone 4; the real-task runner will reuse this setup.
- [x] Keep existing scripted scenarios as harness contract tests; label real-model/scripted-tool tests separately from end-to-end coding evaluations.
- [x] Create five small Python bugfix tasks, each with an immutable starting fixture, task request, setup instructions, and externally controlled acceptance checks.
- [x] Give every attempt a fresh disposable repository copy and real read, search, edit, and command tools.
- [x] Verify that the relevant acceptance check fails on the initial fixture and passes for a known reference solution.
- [x] Keep authoritative acceptance checks outside the agent's editable workspace. Changes to visible tests must not redefine success.
- [x] Evaluate the final artifact for behavior and regressions; do not require the model to match a reference patch's text or exact filenames.
- [x] Capture task ID, starting revision, model/endpoint identity, prompt/profile version, tool trace, final diff, check results, termination reason, usage, and latency.
- [x] Add a minimal baseline mode using the same model, task, trusted disposable workspace, and comparable budget.
- [x] Add a configurable OpenAI-compatible endpoint path where supported, retaining dedicated adapters when a backend needs them. Keep credentials out of traces.

The first baseline can use one already-supported provider. Configurable endpoints should unblock Jorge's chosen hosted models before expanding the benchmark; do not build an exhaustive provider catalog first.

**Done when:** one selected model completes real attempts on all five tasks, the final artifacts are independently scored, and baseline versus harness reports can be reproduced. A low success rate is an acceptable baseline; simulated success is not.

**Dependencies:** disposable execution and timeout cleanup from Milestone 2 are required before unattended real-model evaluations. Initial runner construction can use deterministic providers and trusted fixtures.

## Milestone 2 — Make tools and execution dependable

**Outcome:** the agent can explore, edit, and run checks without fighting the harness or losing control of running work.

- [ ] Fix single-file grep output, option-like search patterns, ignored paths, and explicit truncation reporting.
- [ ] Make bounded reads work on large files; handle empty files and invalid ranges predictably.
- [ ] Add exact text replacement with unique-match validation and a stale-content precondition. Return the applied diff or an actionable mismatch error.
- [ ] Retain whole-file writes for new files and deliberate replacements; use atomic writes where possible.
- [ ] Record the initial working-tree state and derive actual agent changes from snapshots/diffs, including changes made by commands and new untracked files.
- [ ] Prevent overlapping writers to the same workspace and serialize turns within a session.
- [x] Replace code-tool worker-thread subprocess execution with managed processes: bounded output capture, explicit timeout, process-group termination on POSIX, and cancellation.
- [ ] Align tool, provider, turn, and client timeouts so a reported timeout has a defined effect.
- [ ] Use disposable execution environments for evaluations. Provide an explicit local workspace mode for trusted projects and an isolated mode before running unfamiliar repository code.
- [ ] Keep model-provider credentials in the orchestrator; expose only explicitly configured environment variables and mounts to repository commands.
- [ ] Configure project check commands explicitly. Detect likely commands as suggestions; do not guess that every project uses pytest.
- [ ] Fix CLI handling of nullable confidence and other response states that are already valid in the API.

**Done when:** regressions cover single-file search, large/empty reads, ambiguous edits, stale edits, command timeouts, cancellation, command-created file changes, and preservation of pre-existing user edits. No command continues after its run is reported cancelled or terminated.

## Milestone 3 — Enforce honest completion and bounded recovery

**Outcome:** the runtime helps a model finish the task instead of only annotating its final answer.

- [ ] Track task mode and state: exploration, editing, verification, completed, blocked, failed, cancelled, or budget exhausted.
- [ ] Enforce write permissions and file budgets before mutation. Where a plan is required, reject a premature edit with a specific instruction rather than flagging it afterward.
- [ ] Do not require planning for every trivial edit or tests for read-only questions.
- [ ] Require configured relevant checks after the latest mutation before marking an editing task verified.
- [ ] Distinguish check invocation, check success, check failure, unavailable checks, and stale verification. Version/help commands do not count as checks.
- [ ] When the model tries to finish too early, return concise feedback describing the missing action and allow a bounded retry.
- [ ] Return targeted feedback for malformed tool arguments, edit mismatches, failed checks, and recoverable provider failures.
- [ ] Detect repeated identical calls and repeated unchanged failures; stop or change strategy instead of spending the full budget on a loop.
- [ ] Respect provider truncation and finish reasons. A token-limited partial answer is not successful completion.
- [ ] Apply explicit total budgets for wall time, model tokens, tool calls, and recovery attempts. Reserve enough budget to report partial work clearly.
- [ ] Replace confidence-as-correctness in the UI with observed evidence and completion status. Preserve existing fields temporarily if needed, but document their limited meaning.
- [ ] Report actual files changed, checks run, outstanding failures, and why execution stopped.

Keep the loop flexible: inspect → edit → verify → repair is a useful default, not a rigid requirement to manufacture activity. Existing unrelated test failures must be reported separately from newly introduced failures; they should not cause endless repair attempts.

**Done when:** tests prove that an unrelated read cannot certify a fix, `pytest --version` cannot certify verification, a later edit invalidates earlier checks, a later failure is not hidden by an earlier pass, and early completion produces bounded corrective feedback.

## Milestone 4 — Tune the prompt and context for weaker models

**Outcome:** each model gets a small, coherent interface and enough relevant context to make the next coding decision.

- [ ] Move the runtime prompt into a versioned module or template used by both application and evaluations.
- [ ] Start with a concise coding contract: inspect relevant files, make focused changes, use actual check results, repair actionable failures, and report uncertainty honestly.
- [ ] Remove support-demo instructions, duplicate memory aliases, and unavailable semantic-search tools from the default coding toolset.
- [ ] Avoid asking the model to generate bookkeeping that the runtime can derive.
- [ ] Load bounded repository instructions and configured commands with a clear precedence policy. Treat retrieved source, command output, and remembered observations as data, not authority to override user instructions.
- [ ] Introduce model profiles for endpoint/model ID, supported tool features, context/output limits, generation settings, tool verbosity, and retry budgets.
- [ ] Count or estimate prompt usage before requests and enforce model-specific context limits.
- [ ] Preserve the user's objective, current diff, relevant file spans, unresolved failures, and recent actions when compacting context.
- [ ] Keep complete raw traces outside the model context so compaction does not destroy debugging evidence.
- [ ] Bound search and command output while retaining a way to retrieve omitted details.
- [ ] Separate repository conventions from global user preferences. Add provenance, correction/deletion, and injection limits for remembered facts.

**Experiment order:** compact prompt → clearer tool errors → smaller toolset → targeted recovery → context management → profile-specific examples. Change one major variable at a time. Start with deterministic state extraction before adding model-generated summaries.

**Done when:** repeated benchmark runs show which changes help the selected models, and a longer task can cross the context-management threshold without losing its objective, diff, or unresolved failures. Retain only demonstrated improvements or changes needed for basic correctness.

## Milestone 5 — Make daily use recoverable and comfortable

**Outcome:** Jorge can use the harness on an actual project without treating every run as a disposable demo.

- [ ] Persist sessions, run state, events, and model/profile identity in SQLite.
- [ ] Provide reliable cancel and resume behavior across CLI, API, and panel. A browser disconnect must have an explicit execution policy.
- [ ] Use run IDs and idempotent submission/reconnection so transport retries cannot duplicate an edit-producing turn.
- [ ] Resume from persisted state only after checking whether the workspace has changed since the checkpoint.
- [ ] Show a concise live view of current activity, recent tool failures, budget consumption, and final status.
- [ ] Show reviewable diffs, test output, and a partial-work summary on failure.
- [ ] Provide checkpoints and an explicit revert-agent-changes action that preserves unrelated user work. Never automatically reset a dirty repository.
- [ ] Make model switching possible without losing the task, while checking tool-history and context compatibility.
- [ ] Remove mandatory support/RAG startup dependencies from coding-only operation.
- [ ] Update onboarding, README, and development instructions to reflect the new purpose and tested model profiles.

**Done when:** Jorge can start a task, interrupt it, restart the server, inspect its diff, and resume or safely discard only the agent's changes. At least one real project workflow works through the CLI and panel without manual API calls.

## Milestone 6 — Validate the useful task envelope

**Outcome:** the project can state where it helps, how much it costs, and where human intervention is still needed.

- [ ] Expand to 20–30 tasks spanning bugfixes, small features, focused refactors, and repository questions.
- [ ] Include failure cases: ambiguous requirements, malformed tool calls, misleading search results, pre-existing failing tests, context pressure, and interruption.
- [ ] Include a held-out task set that is not used to tune prompts.
- [ ] Run repeated comparisons on at least two intended lower-tier models with recorded sampling settings and budgets.
- [ ] Measure success per category and per model, not just an aggregate average.
- [ ] Track human interventions, false completion claims, regressions, tool misuse, token use, elapsed time, and cost when pricing is configured.
- [ ] Record why each failed task failed: model reasoning, context selection, edit mechanics, tool interface, environment/setup, premature completion, or budget exhaustion.
- [ ] Use the harness for ten real tasks on Jorge's repositories and record every manual rescue.

Provisional personal-use release targets, to revisit after the first real baseline:

- At least 8 of 10 agreed bounded real tasks produce acceptable changes without manual patch repair; ordinary initial clarification is allowed and recorded.
- Held-out evaluations show a repeatable success improvement over the same-model baseline under comparable resource limits. Publish raw counts and variation, not just percentages.
- Zero false “verified” states in the adversarial completion regression suite. This validates bookkeeping, not universal semantic correctness.
- Cancellation, restart, and revert tests show no loss of pre-existing user work.
- Successful tasks fit a per-task time/token budget Jorge chooses after observing the baseline. There is no hidden premium-model dependency.

These targets are acceptance criteria, not claims about current performance. If a model cannot meet them, narrow its supported task envelope or report it as unsuitable for unattended edits.

## Evaluation rules

The primary score is **task acceptance on the final repository state**. Supporting metrics explain why a run succeeded or failed.

| Metric | Measurement |
|---|---|
| Task success | Independent acceptance checks plus task-specific review where automation is insufficient |
| Regressions | Previously passing protected checks fail on the final artifact |
| False completion | Agent claims success despite known missing or failing acceptance requirements |
| Intervention | Human corrections needed after the initial task specification |
| Tool reliability | Invalid calls, execution failures, repeated calls, and recovery outcome |
| Efficiency | Input/output tokens, tool calls, wall time, and configured monetary cost |
| Change quality | Unrelated edits, test weakening, and unnecessary churn assessed against the task |

Use equivalent task inputs, starting snapshots, and budget definitions when comparing variants. Include all attempts, including timeouts and environment failures, with separate failure labels. Keep authoritative grading outside the model's control. Do not use the final answer's wording or the number of citations as a proxy for working code.

For coding explanations, use a task-specific factual rubric tied to the repository. Keep those results separate from executable patch success. A model judge is optional assistance, not the sole source of truth.

## First implementation sequence

Each item should be a reviewable change with focused validation. This is the immediate queue, not authorization to implement the entire roadmap at once.

1. **Shared runtime and truthful eval labels:** extract common prompt/tool/config setup; preserve scripted tests and distinguish their purpose.
2. **File-tool fixes and targeted edits:** fix grep/read defects and add exact replacement with stale-edit protection.
3. **Managed execution:** process cancellation, bounded output, disposable workspaces, and actual change tracking.
4. **Five-task real evaluator:** independent acceptance checks, reproducible traces, and same-model baseline mode.
5. **Chosen-model connectivity:** configurable endpoint/profile support sufficient to run Jorge's selected model through the evaluator and application.
6. **Completion and recovery:** post-edit verification, pre-action limits, bounded corrective feedback, and explicit terminal statuses.
7. **Measured prompt/tool experiments:** compare the compact coding profile against the baseline before adding context complexity.
8. **Personal-use workflow:** persistence, safe resume, reviewable diffs, and ten real project tasks.

The earliest useful checkpoint is items 1–5: a reliable way to observe an actual model doing actual coding. Items 6–8 turn that foundation into a practical assistant.

## Deferred until evidence supports them

- Semantic/vector search: add only when task failures show lexical search and bounded file inspection are insufficient.
- Multi-agent orchestration: add only if a measured gain justifies its coordination and token costs.
- Premium-model supervision or fallback: optional and explicitly selected, never required to make the default harness function.
- LLM confidence judges: lower priority than independent tests and honest completion status.
- Fine-tuning, full IDE integration, hosted multi-tenancy, and leaderboard optimization: outside the first personal-use release.
- Large framework migration or architectural rewrite: the existing provider/loop/tool separation is sufficient for this plan.

## Decisions to make from the first baseline

- Which exact local or hosted models Jorge wants to use and what tool protocol they support.
- Which real repositories and task types matter most.
- Acceptable per-task latency, token use, and cost.
- Which project commands and environment setup are needed beyond Python.
- Which model failures are recoverable with tooling and which require narrower tasks or another model.

Keep this roadmap tied to observed failures. The next feature should remove a demonstrated obstacle to Jorge completing a real coding task.

## Known limitations and proposed fixes

Limitations of shipped slices, with the fix we currently expect to use. A fix
enters the queue when a real task or evaluation shows the limitation matters.

### Turn change reporting (item 6, PR #30)

| Limitation | Proposed fix |
|---|---|
| Changes cannot be attributed. A concurrent edit by Jorge or another process during a turn is reported as a turn change. | Add the Milestone 2 single-writer lock per workspace. Label each changed path as a file-tool edit (its hash matches the last tool result) or as a change from a command or another process. |
| Only paths are reported, not diff content. There are no checkpoints and no revert. | In Git repositories, checkpoint the working tree with a temporary index (`git write-tree` without touching Jorge's index or stash) and diff or revert against it. Outside Git, keep bounded copies of changed text files. This is the Milestone 5 checkpoint/revert work. |
| Ignored directories (`.git`, `.venv`, `node_modules`, caches) are not tracked, so dependency installs or Git metadata changes are invisible. | Record a cheap per-directory stat summary for ignored top-level directories and report "ignored directory changed" without listing its contents. |
| Snapshot time counts toward neither the wall-time budget nor `latency_ms`. | Measure snapshot time, report it in the response, and start the wall deadline before the first snapshot. |
| Each turn reads every tracked file twice, up to the limits. Large repositories hit `unavailable`. | Cache `(size, mtime_ns, inode)` beside each hash and rehash only files whose metadata changed. Reuse the previous turn's after-snapshot when metadata is unchanged. |
| Five files on `main` fail `ruff format --check` (pre-existing). | Run a formatting-only PR with no behavior changes. |

### Context and total-token budgets (item 6)

| Limitation | Proposed fix |
|---|---|
| The byte heuristic overestimated one `gemma4:12b` first request by about 66% (2,950 versus 1,781 tokens), so a context limit stops turns early. | Add a per-model-profile bytes-per-token ratio, calibrated from recorded first-request usage in real evaluations. Keep the conservative default for unprofiled models. |
| The harness does not set the model server's context length. A harness limit larger than the server's window still allows silent server-side truncation. | Put the context length in the Milestone 4 model profile, and have the Ollama adapter send it as the request's context option so both sides agree. |
| Exhausting the context stops the turn instead of continuing. | Add Milestone 4 compaction that keeps the objective, current diff, relevant file spans, unresolved failures, and recent actions. Keep raw traces outside the model context. |
| The total budget is estimate-gated. One request can overshoot it when the provider's count exceeds the estimate. The overshoot is reported, not prevented. | Profile calibration narrows the error. A stricter variant can reserve a safety margin proportional to the observed estimate error. |
| Some backends can omit cached prompt tokens from reported usage, so the total budget may undercount work. This was not observed in the one local check. | Record the per-request estimate beside reported usage in real-eval traces, and flag requests where the reported prompt count is far below the estimate. |
