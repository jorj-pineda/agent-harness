# Roadmap: a coding harness for lower-tier models

Updated: 2026-10-12

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
- **2026-10-04 — Item 7 clearer-tool-errors experiment recorded:** exact-edit
  mismatch errors now give specific retry advice, and rejected `python3` calls
  suggest `python`. Thirty local `gemma4:12b` attempts with thinking off and
  matching recorded settings/fixtures yielded harness 11/15 and minimal 9/15
  accepted, versus 12/15 and 9/15 previously. Two attempts recovered with exact
  newline edits and one ran a passing check; task acceptance did not improve.
  Repeats also varied before changed feedback. See `evals/BASELINES.md` and its
  raw report; these results do not establish a causal regression or improvement.
- **2026-10-04 — Item 7 Python pytest alias experiment recorded:**
  `python3 -m pytest` now uses the existing `python` execution path; module
  restrictions and exact configured-check matching remain. Tool descriptions
  advertise the alias. Thirty local attempts with matching recorded settings
  and fixtures yielded harness 8/15 and minimal 9/15 accepted, versus 11/15 and
  9/15. All six submitted alias checks executed and passed, but their artifacts
  failed independent acceptance on empty input. Two later edits made check
  evidence stale. Harness `divide_zero` got blocked on repeated edit mismatches
  in all repeats; empty final answers also received completed status. This is
  basic tool-usability support, not a measured coding-quality gain. See the raw
  report and limitations in `evals/BASELINES.md`.
- **2026-10-05 — Blank-final recovery and comparison recorded:** the shared loop
  now treats empty or whitespace-only final replies as missing completion,
  uses the existing completion-retry allowance, and returns incomplete with an
  explicit fallback when recovery cannot finish. Verification and blank-answer
  recovery share that allowance and retain existing budget/finish-reason gates.
  Thirty matching local attempts yielded harness 10/15 and minimal 8/15
  accepted, versus 8/15 and 9/15. No blank answer received completed status;
  ten attempts reached the iteration limit and six decimal artifacts still
  received completed status despite failed acceptance. This corrects completion
  bookkeeping, without establishing a coding-quality gain. See
  `evals/BASELINES.md` for raw results and limitations.
- **2026-10-07 — Smaller-toolset comparison recorded:** an opt-in shared
  `whole_file` setting removes only `replace_text` and adapts its one prompt
  instruction. Reports record the toolset and exposed schemas. Thirty matching
  local attempts yielded harness 9/15 and minimal 8/15 accepted, versus 10/15
  and 8/15. Harness division improved from 1/3 to 3/3, but harness slug declined
  from 3/3 to 0/3; minimal flags declined and minimal slug improved. All six
  decimal artifacts still passed visible checks and failed independent empty-input
  acceptance. Exact-edit errors disappeared without an aggregate gain; `full`
  remains the default. See `evals/BASELINES.md` for results and limitations.
- **2026-10-07 — Configured-check guidance comparison recorded:** exercised the
  existing shared project-check setting with the whole-file toolset held fixed;
  required verification remained disabled. Thirty matching local attempts yielded
  harness 9/15 and minimal 3/15 accepted, versus 9/15 and 8/15. Passing configured
  checks occurred in 11 harness and seven minimal attempts, but minimal mode
  repeatedly supplied string argv and stopped before editing five fixtures.
  All six decimal artifacts still failed acceptance. No default or runtime
  implementation changed; see `evals/BASELINES.md` for results and limitations.
- **2026-10-07 — String-argv feedback comparison recorded:** malformed command
  strings now receive an array example and configured-check reminder without
  coercion or execution. Existing budgets bound model correction attempts. Thirty
  matching local attempts yielded harness 9/15 and minimal 5/15 accepted, versus
  9/15 and 3/15. Six attempts later dispatched valid arrays, but string-argument
  errors rose from 16 to 23. The two newly accepted dedupe artifacts were correct
  before their first changed feedback, limiting causal attribution. Retain the
  accurate diagnostic as tool usability, not a demonstrated model-quality gain.
  See `evals/BASELINES.md` for results and limitations.
- **2026-10-08 — Normalized response tracing prepared:** the shared loop now
  supports opt-in request/response, recovery-message, and tool-result evidence
  outside model context. Real-task reports collect it with tool-call IDs, batch
  boundaries, finish reasons, per-request estimated/reported usage, and failures
  by exception class. Four local real-tool instrumentation smoke attempts captured
  two blank-final recoveries followed by non-empty completion and one exhausted
  recovery. Three artifacts passed independent acceptance; one still hit the
  iteration limit. This validates evidence capture, not improved coding quality.
  Provider wire payloads/hidden reasoning, durable traces, and crash recovery
  remain outside this slice. See `evals/real_tasks/README.md` and `evals/BASELINES.md`.
- **2026-10-08 — Required-verification comparison recorded:** enabled the existing
  post-edit check requirement, holding other recorded settings and budgets fixed.
  Thirty matching local attempts yielded harness 9/15 and minimal 5/15 accepted,
  unchanged in every task/mode cell. Full traces show one combined blank-answer
  and verification retry followed by a passing check and non-empty completion;
  the artifact was already correct. The other 29 attempts received no
  verification-specific guidance. Thirteen turns reached the iteration limit,
  and all six decimal artifacts still failed empty-input acceptance. No runtime
  default or implementation changed; this measures the existing gate, without
  establishing a coding-quality gain. See `evals/BASELINES.md` and its raw report.
- **2026-10-08 — Early argument-recovery trial measured and retired:** a frozen
  opt-in implementation injected a configured-check argument example immediately
  after string-argv validation failure, sharing the existing recovery allowance.
  Thirty matching local attempts yielded harness 10/15 and minimal 4/15 accepted,
  versus 9/15 and 5/15: aggregate acceptance stayed 14/30. Ten of eleven early
  prompts were followed by valid arrays, but two minimal slug artifacts completed
  despite failed acceptance. The trial did not establish a reliable quality gain
  or a correctness requirement, so its code/settings/tests were reverted. Its
  immutable revision, raw evidence, and isolated replay instructions remain in
  `evals/BASELINES.md`; no new runtime option or default is shipped.
- **2026-10-08 — Personal-use turn diffs prepared:** shared snapshots now retain
  bounded text alongside hashes and return per-file review diffs through the API,
  CLI, and panel, including budget-exhausted turns. The working tree at turn start
  is the baseline; staged and unstaged edits are preserved and Git's index/stash
  are untouched. Binary, symlink, environment-file, unreadable, and over-limit
  content has explicit omission reasons. The CLI also accepts null confidence
  so valid partial results remain visible. This supports item 8's review workflow,
  with 528 offline tests, Ruff/mypy, both 90/90 scripted matrices, and CLI/panel
  checks against a temporary scripted-provider API using real file tools. These
  validate reporting, not model capability; no live comparison or quality gain
  is claimed. Durable checkpoints, revert, attribution, persistence, cancellation,
  and resume remain open.
- **2026-10-08 — Process-local turn admission prepared:** the shared configured
  runtime now rejects overlapping session/workspace turns before snapshots,
  inference, transcript mutation, or tools. Reservations include final review
  generation; unrelated workspace roots can progress concurrently. Owned tool
  execution settles before timeout/cancellation returns, keeping a reservation
  while synchronous writes finish or asynchronous cleanup runs. Busy submissions
  receive a blocked envelope and explicit retry instruction, without being queued.
  This addresses personal-use overlap within one process; cross-process locks,
  external-writer attribution, persistence, cancellation endpoints, transcript
  repair, and resume remain open. Worker cleanup can exceed deadlines; a stuck
  synchronous worker remains busy. Validation passed 73 focused and 547 offline
  tests, Ruff/mypy, and both 90/90 scripted matrices with evidence labels inspected.
  These checks validate coordination, not improved model coding quality; no live
  comparison is claimed.
- **2026-10-08 — Coding-only startup prepared:** the API now imports and opens
  legacy support retrieval resources only with `ENABLE_SUPPORT_TOOLS=true`.
  Default coding startup allocates no embedding client or Chroma collection and
  works with unavailable support storage. A fresh-interpreter regression blocks
  support imports while completing a real file edit through the API. Explicit
  support mode retains real retrieval and read-only SQL tools; missing resources
  from a custom factory fail at startup with component cleanup. Package dependency
  extras, persistence, cancellation, and resume remain outside this slice. Validation
  passed 38 focused and 550 offline tests, Ruff/mypy, and both 90/90 scripted matrices
  with evidence labels inspected. No model behavior or coding-quality improvement
  is claimed.
- **2026-10-09 — Evidence-first turn review prepared:** the panel and CLI now
  lead with completion and observed check status. The panel removes confidence
  success coloring and the broad "verified" badge; the legacy score appears in
  collapsed details explaining that reads can score 1.0 without establishing
  correctness. Source locations are labeled as tool-harvested evidence, and
  unavailable-check errors are visible. API numeric fields and escalation
  mechanics remain compatible. Validation passed 35 focused and 553 offline
  tests, Ruff/mypy, both 90/90 scripted matrices with labels inspected, and
  browser rendering checks across partial, stale, failed, unavailable, missing,
  and passed states. These validate presentation, not model coding quality.
  Persistence, cancellation, resume, and independent acceptance remain open.
- **2026-10-10 — Finalized-session review persistence prepared:** the API now
  archives session creation and finalized runtime turns in a separate SQLite
  store, including transcript, turn records, full final response envelopes,
  configured model IDs, observed completion/check evidence, and bounded diffs.
  Owner-scoped list/detail routes expose committed snapshots after a fresh API
  lifecycle, even without the original workspace. Saved history is read-only
  after restart; both chat transports reject continuation before inference.
  Saves run before admission release and final delivery; storage failure surfaces
  an error and retires the live session. Failed/interrupted execution retains
  only the previous saved snapshot and cannot reuse its unfinished transcript.
  Validation passed 75 focused and 560 offline tests (five live deselected),
  Ruff/mypy, and both 90/90 scripted matrices with evidence labels inspected.
  These validate persistence mechanics, not coding quality; no inference or model
  comparison was run. Interrupted-turn recovery, live-event persistence, profile
  configuration capture, cancellation endpoints, workspace-checked resume,
  checkpoints/revert, retention/deletion, cross-process coordination, and CLI/panel
  history views remain open. This completes only the review-persistence slice
  of item 8; Milestone 5 is not complete.
- **2026-10-10 — Explicit turn cancellation prepared:** a per-turn cancellation
  signal now stops further inference/tool dispatch and settles owned async/sync
  work before returning `cancelled`. The API exposes an owner-scoped cancellation
  request, with 202 meaning requested and 409 for idle/finalizing/read-only state.
  Cancelled results retain partial evidence and bounded diffs, are archived before
  delivery, and retire the session instead of granting continuation. Interrupted
  invocations are marked explicitly; prior checks become stale when an interrupted
  edit/command may have changed files. HTTP/SSE disconnects leave app-owned turns
  running under their budgets; shutdown requests cancellation and waits before
  closing resources. CLI cancellation and Ctrl-C handling share the endpoint;
  the panel adds Cancel and removes stream-error POST resubmission.
  Validation passed focused API/runtime/tool/CLI regressions, 574 offline tests
  (five live deselected), Ruff/mypy, and both 90/90 scripted matrices with labels
  inspected. A real subprocess regression prevents descendant writes after
  cancellation; browser checks exercised real file-tool cancellation through a
  scripted-provider API and a simulated stream disconnect without resubmission.
  The temporary server was stopped. No inference or coding-quality comparison ran.
  Synchronous workers remain uninterruptible and can delay cleanup indefinitely;
  remote inference/billing cannot be guaranteed to stop. Cross-process coordination,
  run IDs/idempotence, reconnection, crash recovery, checkpoints/revert, and safe
  resume remain open. Milestone 5 remains incomplete.
- **2026-10-10 — Durable run identity/idempotent submission prepared:** API
  submissions now receive a saved run ID. Optional per-user request keys on HTTP
  and SSE, required on asynchronous `POST /runs`, bind exact input before work
  starts. Matching retries join the active task or replay its saved result;
  conflicting input returns 409. Owner-scoped run inspection/listing exposes
  lifecycle status and the full final envelope, including bounded review evidence.
  Run result and finalized session archive commit atomically before admission
  release and final delivery. Startup marks abandoned running records interrupted
  without dispatching tools; failed/interrupted keys never start another run.
  Validation passed 62 focused API/storage tests, 587 offline tests (five live
  deselected), Ruff/mypy, and both 90/90 scripted matrices with evidence labels
  inspected. Real file-tool regressions verify one edit across duplicate
  HTTP/SSE/async submissions, restart replay, cancellation replay, busy/conflicting
  submissions, and storage-failure rollback. No inference or model-quality
  comparison ran. Legacy unkeyed calls remain distinct; current clients do not
  retain retry keys or expose run-history/reconnection flows. Tool-event replay,
  partial crash evidence, profile/configuration capture, cross-process execution
  coordination, checkpoints/revert, and workspace-checked resume remain open.
  Milestone 5 remains incomplete.
- **2026-10-10 — Client run review and explicit submission recovery prepared:**
  CLI chat prints/retains a stable submission key before dispatch, blocks new
  messages after an uncertain outcome, and offers read-only `/check` plus explicit
  same-input/key `/retry`. New `runs`/`run` commands inspect server history and
  final evidence without submitting work. The panel retains one exact pending
  request before sending, restores it read-only after reload, and offers explicit
  check/retry plus paged history. History and recovered results never enable
  continuation; forgetting a local reference does not cancel server work.
  An owner-scoped request-key filter finds a run when its ID response was lost.
  Validation passed 48 focused CLI/API tests, 596 offline tests (five live
  deselected), Ruff/mypy, JavaScript syntax, and both 90/90 scripted matrices with
  labels inspected. A real-tool CLI regression loses the response after archive
  commit and recovers the same turn with one edit. Browser checks against a
  temporary scripted-provider API covered real file-tool active retries, reload,
  lost POST response, read-only history after restart, failed/interrupted/cancelled
  presentation, and local-storage failure before dispatch. The server was stopped;
  no inference or coding-quality comparison ran. Browser pending input is stored
  locally; CLI retry input lasts only within its REPL. Cross-tab coordination,
  event replay, automatic reconnection/polling, checkpoints/revert, profile capture,
  and workspace-checked resume remain open. Milestone 5 remains incomplete.
- **2026-10-10 — Durable final-workspace comparison prepared:** finalized API
  turns now save a scoped final fingerprint baseline and root identity atomically
  with run/session evidence, reusing the final runtime snapshot. An owner-scoped
  workspace-review endpoint and CLI command compare current contents after
  restart, naming the archived turn and reporting capped changed-path lists plus
  full counts. Older archives remain not-recorded; disabled/missing tracking,
  changed/replaced roots, read/traversal failures, special files, growing files,
  and snapshot limits do not silently produce unchanged. Process-local admission
  covers owned comparison reads through disconnect/shutdown without modifying
  archives, user files, or Git state. Environment hashes and raw symlink targets
  are omitted from durable review records. Validation passed 153 focused tests,
  610 offline tests (five live deselected), Ruff/mypy, and both 90/90 scripted
  matrices with labels inspected. Restart tests use real file tools and preserve
  pre-existing edits; rollback tests cover baseline/session/run atomicity. No
  inference or model-quality comparison ran. Unchanged is scoped review evidence,
  not resume permission: ignored directories, environment files, file modes, and
  symlink referents remain outside comparison. Non-atomic external-writer races,
  profile/transcript compatibility, partial crash recovery, explicit resume,
  checkpoints/revert, and panel comparison controls remain open. Milestone 5
  remains incomplete.
- **2026-10-10 — Transcript structural integrity prepared:** a shared normalized
  history check now blocks malformed existing tool-call/result history before
  inference or transcript mutation; configured execution also checks before
  prompt refresh, snapshots, and finalization callbacks. Invalid model batches
  (empty/duplicate call IDs or empty tool names) stop as incomplete before any
  batch tool dispatch, without fabricating results or repairing history.
  An owner-scoped saved-transcript review endpoint and CLI command report capped
  message-index/issue-code diagnostics plus counts and archived turn identity
  after restart, without exposing transcript payloads or enabling continuation.
  Validation passed 131 focused and 632 offline tests (five live deselected),
  Ruff/mypy, and both 90/90 scripted matrices with evidence labels inspected.
  Regressions cover preserved archives/user files, keyed replay, external
  cancellation leaving unmatched calls, and reservation release for fresh
  sessions after worker cleanup. These validate integrity mechanics, not model
  coding quality; no inference or comparison ran. Structurally valid history
  does not establish completion, tool-result truth, turn-record alignment,
  provider/profile compatibility, context fit, or unchanged workspace state.
  Profile capture, broader workspace coverage, transcript repair, partial crash
  recovery, explicit guarded resume, checkpoints/revert, and panel review controls
  remain open. Milestone 5 remains incomplete.
- **2026-10-10 — Durable configured execution evidence prepared:** the shared
  runtime now exposes an optional pre-execution snapshot callback inside admission,
  after transcript validation and before prompt refresh/snapshots/inference/tools.
  API runs bind an allowlisted configuration snapshot once in SQLite; storage
  failure prevents execution. Final turn archives copy it atomically with final
  run/session/baseline evidence. Snapshot fields include budgets/policies/review
  limits, provider/configured model, applicable Ollama options, tool names, and
  hashes of base prompt, exposed schemas, configured endpoint and check argv.
  Credentials, raw endpoints/check arguments, data paths, and injected facts are
  excluded from this metadata; sensitive/invalid endpoint and unknown custom
  provider identities remain unknown. Existing transcripts/traces are not redacted.
  API detail/history and CLI run review retain the original snapshot across restart
  and keyed retry, even when current settings differ; older/unadmitted records have
  no invented profile. Validation passed 97 focused and 662 offline tests (five
  live deselected), Ruff/mypy, and both 90/90 scripted matrices with labels inspected.
  Tests cover pre-inference durability, storage rollback/failure, final evidence
  atomicity, abandonment, privacy, legacy records, and real file-tool preservation.
  No inference or coding-quality comparison ran. This records configured evidence,
  not backend attestation or a selectable model-profile system. Matching hashes do
  not establish compatibility: implementation/runtime revisions, server/provider
  defaults, environment routing, command environment, facts, and capabilities are
  outside capture. Automatic restore, compatibility/context gates, broader workspace
  preconditions, explicit resume, crash repair, checkpoints/revert, and panel profile
  controls remain open; saved sessions remain read-only and Milestone 5 incomplete.
- **2026-10-10 — Read-only configuration/context review prepared:** an owner-scoped
  API computation and thin CLI command compare the latest saved execution snapshot
  with the selected current provider/settings/tools. They report changed field
  names, unknown identities, missing profiles, and inconsistent archive identities
  without exposing compared values or replacing original evidence. Context review
  refreshes detached history with current owner facts/check/prompt/tool schemas and
  optional proposed input, reusing the configured runtime's first-request budget
  math. Invalid transcripts/inconsistent archives skip estimation; exhausted
  context/total budgets are blocked estimates, and disabled context limits or
  omitted proposed input are labeled explicitly. Shared API registry setup preserves
  coding-only startup and support-resource separation. Process-local admission
  rejects overlapping owned turns/reviews; no run, inference, tools, facts, files,
  or archive mutations result from review. Validation passed 90 focused and 686
  offline tests (five live deselected), Ruff/mypy, and both 90/90 scripted matrices
  with evidence labels inspected. Restart tests cover matching/drift/unknown,
  exhausted budgets, missing workspace, ownership, preserved user edits/archives,
  and unchanged run history. No model inference or coding-quality comparison ran.
  All reviews explicitly retain read_only=true and resume_supported=false. Matching
  fields and byte estimates do not establish backend/tool compatibility, true model
  context fit, workspace state, task completion, or a lease for later execution.
  Runtime/implementation identity, broader workspace coverage, actual compatibility
  enforcement inside execution admission, explicit continuation, crash repair,
  checkpoints/revert, and panel controls remain open; Milestone 5 is incomplete.
- **2026-10-10 — Saved regular-file permission review prepared:** finalized workspace
  baselines now retain non-excluded regular-file permission bits from the same
  open descriptor used for content hashing, without another filesystem sweep.
  Restart review detects permission-only drift in the existing modified-path
  lists/counts and reports mode coverage explicitly through API and CLI. Older
  baselines remain content-only with not-recorded coverage; skipped/failed reviews
  do not claim mode comparison. Environment metadata remains excluded. Per-turn
  content diffs retain their existing scope. Validation passed 80 focused and 690
  offline tests (five live deselected), Ruff/mypy, and both 90/90 scripted matrices
  with labels and metric limits inspected. Real-file restart regressions preserve
  user content/archives and reject restored continuation; no inference or coding
  quality comparison ran. Directory/symlink modes, ownership, ACLs, extended
  attributes, ignored contents, and symlink referents remain outside comparison.
  Non-atomic external-writer races, explicit guarded continuation, checkpoints/
  revert, crash repair, and panel review controls remain open. Milestone 5 remains
  incomplete; permission evidence does not authorize resume.
- **2026-10-12 — Panel saved-session review prepared:** selecting a saved run now
  exposes explicit workspace, transcript, and configuration/context review actions
  through the existing API endpoints. Turn evidence remains first; review labels
  distinguish the selected run from the latest archived session turn. Results show
  mode coverage/exclusions/path counts, structural diagnostics, changed/unknown
  fields, and estimates with optional proposed input/current provider selection.
  Input used for an estimate remains attached to that result; actual inclusion,
  blocked/skipped/unbounded states, missing records and request failures stay
  explicit. Selecting another run clears results/input and discards late responses.
  Reviews never enable the composer, retry submissions, poll, or restore archives.
  Validation passed 35 focused and 690 offline tests (five live deselected), Ruff,
  mypy, and JavaScript syntax checks. Browser checks against a fresh scripted-
  provider API lifecycle used real file edits, older-run/latest-turn comparisons,
  permission drift, context exhaustion, and preserved run/session evidence. Simulated
  rendering/error checks covered busy/unavailable/legacy, invalid/truncated history,
  drift/unknown/unbounded estimates, missing fields, 403/404/network errors,
  client timeout recovery, safe text rendering, and selection races. The temporary
  server was stopped; no inference or model quality comparison ran. Review inputs are transient; there is no automatic review,
  continuation, checkpoint/revert, partial crash repair or cross-process lock.
  Broader workspace coverage and guarded execution preconditions remain open;
  Milestone 5 is incomplete.
- **2026-10-12 — Ignored top-level metadata reporting prepared:** shared snapshots
  now retain hashes of lstat fields for the fixed default ignored root names,
  without traversing contents, following links, or adding another sweep. Turn
  envelopes and saved final-workspace review expose a separate metadata-only
  added/modified/deleted summary through API, CLI and panel. New baselines save it
  atomically with existing evidence; older baselines remain not-recorded and failed/
  skipped comparisons not-checked. Overall workspace-review changed includes
  metadata drift; tracked-file counts/diffs keep their scope and the path cap is
  shared, preserving tracked-file priority. Regressions demonstrate command-created
  ignored entries, removal/replacement, no content reads/referent inspection,
  environment privacy, capped counts, restart review of complete/partial turns,
  preserved user index/content/archives, legacy coverage and unavailable reads.
  A regression also confirms the blind spot: rewriting an existing child can leave
  directory metadata unchanged. Matching metadata never certifies a subtree,
  dependencies, attribution or permission to resume. Validation passed 97 focused
  and 699 offline tests (five live deselected), Ruff/mypy, JavaScript syntax, and
  both 90/90 scripted matrices with labels/metric limits inspected. Browser checks
  used a fresh scripted-provider API lifecycle with real edits and metadata drift;
  missing/skipped coverage and truncated summaries were also rendered. The temporary
  server was stopped; no inference or model-quality comparison ran. Non-atomic
  external-writer races, deeper ignored contents, guarded continuation, checkpoints/
  revert, crash repair and cross-process locks remain open; Milestone 5 incomplete.
- **2026-10-12 — Workspace review timing and earlier dispatch deadline prepared:**
  configured turns now start the wall clock inside admission after transcript
  guards, before execution-profile capture, prompt refresh and initial snapshot.
  The loop uses that earlier start, so slow setup cannot grant a fresh model/tool
  budget. Initial snapshot expiry prevents inference/edit dispatch. Final snapshots,
  diffs and baseline construction still settle under admission after the deadline,
  preserving evidence rather than reclassifying already completed work. A nullable
  workspace_review_ms records those review phases before persistence/delivery;
  skipped tracking is zero, legacy/low-level/blocked responses unmeasured. API
  archives/replays, CLI and panel retain/display it separately from loop latency;
  setup, storage, transport and standalone reviews are outside this timing field.
  Regressions cover held initial/final reads, pre-action budget enforcement,
  preserved edits/diffs, callback timing, disabled/unavailable tracking, slow profile
  capture, HTTP/SSE restart replay and nullable client presentation. Validation
  passed 127 focused and 711 offline tests (five live deselected), Ruff/mypy,
  JavaScript syntax, and both 90/90 scripted matrices with labels/limits inspected.
  Browser checks used real-file saved results from a fresh scripted-provider API
  lifecycle and covered missing/null/zero/large durations. The temporary server was
  stopped; no inference or quality comparison ran. Historical real-eval deadlines
  excluded initial setup; their latency/budget scope is preserved and documented.
  Tool cleanup/final review/persistence may exceed deadlines, and synchronous
  workers remain uninterruptible. Guarded continuation, checkpoints/revert, crash
  repair and cross-process coordination remain open; Milestone 5 incomplete.
- **2026-10-12 — Durable bounded tool-progress metadata prepared:** admitted API
  runs now initialize a metadata journal atomically with execution-profile binding,
  before inference. Existing shared tool-start/end callbacks commit ordered time,
  allowlisted name, error presence and latency before onward execution/live delivery.
  An owner-scoped endpoint and thin CLI command inspect the journal after restart;
  run/history response contracts remain unchanged. The first 200 observations are
  retained with explicit truncation and a full observed-event count. Arguments,
  outputs, error text, tool-call IDs and messages are excluded; unknown names are
  replaced. Legacy/unadmitted records remain not-recorded. Storage failure stops
  following work and retires the session, without undoing a possible prior edit.
  Validation passed 78 focused and 721 offline tests (five live deselected), Ruff/
  mypy, and both 90/90 scripted matrices with labels/metric limits inspected. A forced
  owned-process exit after a real write leaves durable start/end evidence and the
  preserved user line, then restart marks interrupted without inventing a final
  response or dispatching tools. Other regressions cover HTTP/SSE, keyed replay,
  ownership, empty versus legacy journals, privacy, prefix bounds, profile/journal
  atomicity, terminal/out-of-order writes, and pre/post-edit storage failure.
  No inference or coding-quality comparison ran; the fixture process was stopped.
  Start precedes dispatch gates and cannot prove a mutation; end metadata cannot
  certify acceptance/check success. Full event/output replay, partial transcript/
  diff recovery, crash repair, guarded continuation, checkpoints/revert, retention
  and cross-process coordination remain open; Milestone 5 incomplete.
- **2026-10-12 — Panel tool-progress inspection prepared:** saved-run selection now
  exposes an explicit owner-scoped GET for that run's durable metadata journal,
  separate from latest-session workspace/transcript/configuration review. It shows
  coverage, full counts versus saved prefix, timestamps, ordered tool names and
  error presence/latency without success claims, output/argument rendering or replay.
  Large prefixes are collapsed and scrollable; legacy/empty/missing/truncated data
  remains explicit. Lifecycle is labeled as state when selected, not live status.
  Failed reads replace prior observations; mismatched run IDs are rejected, and
  timeouts release controls. New selections/sessions and explicit submission retries
  clear journal results; delayed responses cannot cross selections. The composer
  stays read-only for saved history. Validation passed 13 focused and 721 offline
  tests (five live deselected), Ruff/mypy and JavaScript syntax. Browser checks used
  a forcibly stopped fixture API process after a real edit, then a fresh lifecycle
  to inspect interrupted and completed journals without new runs or archive changes.
  Simulations covered legacy/empty/missing fields, 200-entry truncation, safe text,
  403/404/network/timeout errors, wrong-run identity, late responses and cleared
  retry/new-session views. Owned fixture/server processes were stopped and the tab
  closed; no inference or coding-quality comparison ran. Evaluation behavior is
  unchanged; scripted matrices were not rerun for this client-only slice. Full
  transcript/output/diff recovery, automatic reconnection, guarded continuation,
  checkpoints/revert and cross-process coordination remain open; Milestone 5 incomplete.
- **2026-10-12 — Atomic whole-file writes prepared:** `write_file` now prepares
  content in a same-directory exclusive temporary file and replaces the destination
  only after writing/flushing and copying existing permission bits. Partial-write,
  permission-copy and rename failures preserve existing user content and clean up
  temporary files. New files retain normal umask-based permissions, special files
  are rejected, and the existing response/tool schema is unchanged. Real-file
  regressions cover failure before publication, executable/private permissions,
  new nested/empty files, size limits, hard-linked aliases and in-root symlink
  resolution. Validation passed 55 focused and 733 offline tests (five live
  deselected), Ruff/mypy, changed-file formatting, and both 90/90 scripted matrices
  with contract labels and metric limits inspected. These validate file publication,
  not model coding quality; no inference or comparison ran. Replacement changes
  inode identity and does not copy ownership/ACLs/extended attributes. External-writer
  races and whole-file stale-content guards remain open; abrupt exit can leave a
  temporary file, and no fsync/power-loss durability, checkpoint, revert or resume
  guarantee is added. This completes the bounded atomic-write item in Milestone 2;
  the remaining personal-use workflow and Milestone 5 remain incomplete.
- **2026-10-12 — Workspace-scoped search and explicit overflow prepared:** native
  ripgrep searches now apply escaped literal workspace exclusions after caller
  globs and validate returned paths through the workspace guard. Python fallback
  traversal prunes ignored directories and checks resolved file paths before
  reading, avoiding out-of-root symlink reads. Both backends retain the existing
  successful match-list contract up to 100 matching lines; observing a 101st line
  raises an explicit incomplete-search error with narrowing advice instead of
  silently returning a prefix. Ripgrep's per-file match cap includes that extra
  observation, while its process output/time limits remain separate. Validation
  passed 81 focused and 747 offline tests (five live deselected), Ruff/mypy,
  changed-file formatting and both 90/90 scripted matrices with contract labels
  and metric limits inspected. Real-file regressions cover nested/custom ignores,
  literal glob characters, positive caller globs, exact/global limits, overflow
  and explicit narrowing; a configured-runtime test retains the tool error and
  subsequent narrower result without changing user files. These validate search
  mechanics, not model quality; no inference or comparison ran. Native hidden-file,
  Git-ignore, binary and regex behavior still differs from fallback coverage;
  fallback file-read cost is not bounded by file size. Path guards do not provide
  process isolation or atomic protection against external path changes. This
  completes the bounded search-fix item in Milestone 2; guarded continuation,
  checkpoints/revert and the remaining personal-use workflow stay open.
- **2026-10-12 — Bounded fallback search reads prepared:** Python search now reads
  at most 2,000,000 bytes per file and 16,000,000 bytes per call, with one extra
  byte to detect overflow, including nonmatching files. Byte overflow or read
  failure raises an explicit incomplete-search error instead of silently omitting
  files or returning earlier matches. Narrowing recovers without changing files;
  native ripgrep behavior and successful match-list contracts remain unchanged.
  Validation passed 52 focused and 752 offline tests (five live deselected),
  Ruff/mypy and both 90/90 scripted matrices with contract labels/limits inspected.
  No inference or model-quality comparison ran. Traversal cost, regex CPU time,
  external path races and hard return deadlines remain outside these byte bounds;
  guarded continuation/checkpoints/revert and Milestone 5 remain incomplete.
- **2026-10-12 — Bounded ranged-read scans prepared:** read_file now caps the
  whole-file hash/line scan at 16,000,000 bytes plus one overflow-detection byte,
  including long lines outside the selected range. In-budget slices retain complete
  hashes/counts and existing response fields; overflow returns an explicit error
  rather than partial evidence. Opened descriptors must be regular files, with
  POSIX no-follow/nonblocking flags guarding final-component link/special-file swaps.
  Validation passed 75 focused and 757 offline tests (five live deselected), Ruff/
  mypy and both 90/90 scripted matrices with contract labels/limits inspected.
  Regressions cover exact/overflow scans, long unselected lines, complete hashes,
  and real symlink/FIFO swaps without reading external content. No inference or
  quality comparison ran. Parent path races, concurrent content edits, filesystem
  stalls and hard return deadlines remain; this is not process isolation, a
  checkpoint or resume permission. Milestone 5 remains incomplete.
- **2026-10-12 — Explicit directory exploration limits prepared:** list_dir and
  tree now reject result overflow instead of returning a silent prefix. Exact caps
  remain valid (200 immediate entries; 300 tree rows including the root), and
  errors recommend narrower paths or shallower depth. Successful lists retain
  their shape; kind=symlink explicitly labels links, which tree treats as leaves
  without deliberately following directory targets or cycles. Validation passed
  47 focused and 765 offline tests (five live deselected), Ruff/mypy, changed-file
  formatting and both 90/90 scripted matrices with contract labels/limits inspected.
  Real-file regressions cover empty/exact/overflow results, excluded entries,
  root counting, external/cyclic/broken links and explicit narrowing with preserved
  user content. No inference or model-quality comparison ran. These are output
  bounds, not bounded sorting/enumeration work or hard return deadlines; pagination,
  external path races, guarded continuation/checkpoints/revert remain open.
  Milestone 5 remains incomplete.
- **2026-10-12 — Explicit durable workspace checkpoints prepared:** an owner-scoped
  POST captures current eligible regular-file bytes/modes into SQLite independently
  of saved turn envelopes. Metadata-only paged list/detail routes survive restart;
  immutable checkpoint/blob inserts commit atomically, without exposing raw contents
  or changing session/run evidence. Capture caps entries at 2,000 and bytes at
  16,000,000, excludes environment/default ignored components, rejects symlinks,
  special files, unreadable/changing files and harness storage within the workspace.
  App-owned capture holds process-local admission through disconnect and storage;
  shutdown settles owned work before closing resources. Validation passed 23 focused
  and 777 offline tests (five live deselected), Ruff/mypy, changed-file formatting
  and both 90/90 scripted matrices with contract labels/limits inspected. Regressions
  cover exact binary/text bytes, modes, exclusions, restart/ownership, bounds,
  immutable IDs, storage rollback, concurrent rejection, per-file changes and
  transport cancellation without abandoning worker cleanup. No inference or model-
  quality comparison ran. Contents are plaintext local source copies; retention/
  deletion/quota, empty-directory/ownership/ACL restoration and cross-process races
  remain open. Capture is not atomic against external writers. Checkpoints are
  explicit current-state captures, not turn-linked before/after evidence; no restore,
  revert or resume endpoint is enabled. Milestone 5 remains incomplete.
- **2026-10-12 — CLI checkpoint capture/inspection prepared:** thin commands now
  explicitly POST one capture, list paged metadata or inspect a saved manifest
  after restart. JSON output contains metadata only; human output retains scope
  and restore_supported=false. Failed/uncertain capture requests report exception
  class and advise history inspection without automatic retry, polling or coding
  submissions. Validation passed 45 focused and 780 offline tests (five live
  deselected), Ruff/mypy and changed-file formatting. A real-file API/CLI restart
  regression preserves later user edits and run history; timeout tests confirm one
  POST and no retry/GET. No inference or quality comparison ran. Evaluation behavior
  is unchanged; scripted matrices were not rerun for this client-only slice.
  Panel controls, turn-linked captures, restore/revert, retention/deletion/quota,
  guarded continuation and the rest of Milestone 5 remain open.
- **2026-10-12 — Opt-in turn-linked content checkpoints prepared:** sessions may
  explicitly enable bounded plaintext captures before/after admitted configured
  turns. Shared runtime callbacks run inside admission before inference and after
  tools/final workspace review, before archive delivery; missing capture support
  blocks enabled execution. Capture/blob/run-reference binding is atomic, ordered,
  immutable and running-run gated. After records identify the paired before point;
  root replacement rejects pairing. Run history preserves both IDs across restart
  and keyed replay, with no repeated capture. CLI opt-in and saved-run review expose
  the behavior through HTTP. Validation passed 73 focused and 791 offline tests
  (five live deselected), Ruff/mypy, changed-file formatting and both 90/90 scripted
  matrices with labels/limits inspected. Real-file HTTP/SSE/CLI regressions cover
  preserved user edits, restart/keyed replay, before failure preventing inference,
  after failure retaining edits/earlier capture with no invented final response,
  transactional rollback, ordered/terminal binding, missing callback and cancel
  during capture without abandoning workers. No inference or quality comparison ran.
  Default capture remains off; initial capture consumes dispatch budget while final
  capture/storage can exceed deadlines and sit outside existing timing fields.
  Exclusions/plaintext/retention and non-atomic external-writer limitations remain;
  these pairs do not prove attribution or acceptance. Restore/revert, guarded
  continuation, panel opt-in and Milestone 5 remain incomplete.
- **2026-10-12 — Journaled guarded single-file reversal prepared:** an explicit
  owner-scoped run action reverses one eligible before/after checkpoint difference
  only when current root identity, target bytes/mode or absence, and named-file
  state match preconditions. POSIX directory descriptors avoid deliberate parent
  link traversal; only the selected file is replaced/unlinked, preserving other
  paths and Git index. A stable keyed journal commits pending state before work;
  matching retries never repeat it, and startup marks unknown unfinished outcomes
  interrupted without mutation. App-owned actions hold admission through disconnect
  and storage; applied/uncertain actions retire the source session while historical
  archives/run/check evidence remain unchanged. Thin CLI action/status commands
  expose explicit submission and read-only inspection without automatic retry.
  Validation passed 67 focused and 815 offline tests (five live deselected), Ruff/
  mypy, changed-file formatting and both 90/90 scripted matrices with labels/limits
  inspected. Real-file tests cover text/binary/empty/added/deleted content, modes,
  preserved pre-existing/later/unrelated user edits and index, root replacement,
  parent links, preparation races, ownership, corrupt saved data, storage failures,
  keyed restart replay, interruption, disconnect/busy coordination and CLI recovery.
  No inference or model-quality comparison ran. This is observed-change reversal,
  not attribution or acceptance; saved checks must be rerun. External writers can
  race final checks, replacement metadata/inode relationships are not restored,
  parents must exist and SQLite/filesystem outcome commits are not atomic together.
  Whole-checkpoint/multi-file atomic restore, reconciliation, cross-process locks,
  panel controls and guarded continuation remain open; Milestone 5 incomplete.
- **2026-10-12 — Explicit guarded continuation API/CLI:** saved history can now be
  forked into a new checkpoint-enabled session with a pinned latest turn, exact
  idempotent input/key and required acknowledgment of uncaptured state. Source
  evidence stays unchanged. Fork seed/run identity commit atomically before work.
  Each admitted fork turn freshly matches scoped bytes/modes/root identity, valid
  aligned history, known captured configuration and bounded estimated context;
  Ollama context configuration must cover the harness limit. Missing/changed/unknown
  evidence, stale source history, busy roots and newer failed/interrupted/running
  related or archived same-root work block before inference/tools. Matching keyed
  continuation replay survives restart without execution; blocked/failed forks are
  retired. Ordinary saved sessions remain read-only. CLI submits exactly once and
  exposes keyed inspection without polling or automatic retry.
  Validation passed 59 focused and 835 offline tests (five live deselected), Ruff,
  mypy, changed-file formatting and both 90/90 scripted matrices; contract labels and
  metric limits were inspected. Tests cover real writes/restart/fork/replay, preserved
  source evidence and user edits, repeated live-turn checks, configuration/context/
  transcript/checkpoint rejection, ownership, explicit acknowledgment, unknown work,
  admission and atomic storage failure, plus CLI dispatch/error/replay. An outdated
  CLI text assertion was updated after the first full gate detected it; the rerun
  passed. No inference or model-quality comparison ran. Ignored files, empty directory
  topology, index, ownership/ACL/xattrs, unrecorded runtime/backend/dependency state and
  external writers remain outside guarantees; context fit is heuristic. Failed
  unrelated sessions without archives cannot be associated with a workspace. No
  automatic crash repair, cross-process coordination or general compatibility proof.
  Panel continuation controls, full reconciliation and Milestone 5 remain incomplete.
  Jorge selected Duodoro for real-project validation; inspected its clean checkout at
  `5324cabba4ad3c8ad2e1cd1480218cf087d9b741`, instructions and isolated Node/Vitest
  package setup without changing it. Candidate tasks are robust invalid-input duration
  formatting, integer timer-setting normalization, and optional Sunday-first calendar
  layout. Freeze external acceptance and disposable starting copies before inference;
  these candidates are not completed tasks or model-quality evidence.
- **Next:** item 6's listed mechanics are in place; see the known limitations
  below before relying on them unattended. Item 7 has a repeated baseline and
  nine completed comparisons. Argument feedback did not reliably break repeated
  errors; normalized traces show that completion-time verification recovery rarely
  fires in this task envelope, and the earlier argument cue was not retained after
  mixed outcomes. The personal-use workflow now has bounded turn-diff review,
  finalized-session archives, explicit cancellation, durable run identity and keyed
  submission, client run-history/recovery controls, durable scoped workspace review,
  transcript integrity guards/review, durable configured execution evidence,
  read-only configuration/context review, panel session-review controls, bounded
  durable tool-progress metadata, and process-local turn admission; failed-check recovery, model profiles, context
  management, and the rest of the personal-use workflow remain open. Offline checks
  alone are not evidence of better model coding performance.

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

- [x] Fix single-file grep output, option-like search patterns, ignored paths, and explicit truncation reporting.
- [ ] Make bounded reads work on large files; handle empty files and invalid ranges predictably.
- [ ] Add exact text replacement with unique-match validation and a stale-content precondition. Return the applied diff or an actionable mismatch error.
- [x] Retain whole-file writes for new files and deliberate replacements; use atomic writes where possible.
- [ ] Record the initial working-tree state and derive actual agent changes from snapshots/diffs, including changes made by commands and new untracked files.
- [x] Reject overlapping workspace and session turns within one configured-runtime process. Cross-process/external-writer coordination remains open.
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
- [x] Replace confidence-as-correctness in the UI with observed evidence and completion status. Preserve existing fields temporarily if needed, but document their limited meaning.
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
- [x] Remove mandatory support/RAG imports and resource initialization from coding-only startup. Chroma installation extras remain separate work.
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
| Process-local turn admission prevents overlapping configured-runtime turns, but changes from editors, other harness processes, or detached commands still cannot be attributed. | Add cross-process coordination before multiple API workers. Label changed paths using file-tool hash evidence where available; other changes still need explicit source uncertainty. |
| Bounded per-turn text diffs are available, but capture/output limits omit content. Explicit bounded checkpoints and opt-in turn-linked captures now exist; only explicit single-file reversal is supported; whole-workspace/atomic multi-file revert remains open. | Add conflict checks and an explicit bounded revert workflow from durable before/after content pairs, preserving later user edits and Git state. Excluded contents and non-atomic external writers remain outside capture; in-memory diff text is review evidence only. |
| Default ignored top-level entries now have metadata-only reporting, but child rewrites and deeper edits can leave that metadata unchanged. | Keep scope explicit; require selected dependency/Git checks or broader bounded capture before using ignored state as an execution precondition. Metadata equality cannot certify unchanged contents. |
| Initial setup/snapshot now consumes the dispatch budget; final review and storage can still exceed it. Application loop latency and workspace review duration omit setup/storage/transport. | Keep timing scopes explicit. A hard end-to-end bound requires safely interruptible work and a deliberate policy for preserving final evidence; do not abandon workers or imply a hard return deadline. |
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
