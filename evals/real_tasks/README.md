# Trusted coding task development corpus

These small Python coding tasks form a development corpus, not a held-out set. Each task
has a starting fixture with a failing visible test, a reference solution, and
authoritative acceptance tests outside the agent's editable copy. The runner
checks that the original fails and the reference passes before every model run.
It grades the final files, regardless of the agent's answer or the visible tests.

Authoritative pytest uses an owned bootstrap launched with Python `-I -B`. It
imports installed pytest before adding workspace imports, pins an external empty
pytest configuration, disables conftest/plugin autoload and the cache provider,
and rejects an acceptance directory inside the editable workspace. Workspace
`pytest.py`/packages cannot select the grader. Ordinary visible/model-invoked checks
retain project behavior and can therefore report a pass that acceptance rejects.
This corrects a reproduced offline false acceptance from an editable `pytest.py`;
it does not establish that historical model attempts encountered that issue.

New result rows record `acceptance_policy: owned-pytest-v1`. Old absence is unknown;
historical results are not rewritten or regraded, and their original grading
boundary remains part of their limitations. Compare policy versions as well as
task revisions/settings. This controls grader startup/configuration, not hostile
implementation execution: imported task code still runs on the host with its
permissions. Use trusted fixtures; the bootstrap is not process isolation.

Validate the corpus without making a model request:

```sh
uv run python -m evals.real_run --validate-only
```

Run one configured model against the task manifest, then repeat with the same model
in minimal-prompt mode. Use separate report paths:

```sh
uv run python -m evals.real_run --provider ollama --mode harness --report /tmp/real-harness.json
uv run python -m evals.real_run --provider ollama --mode minimal --report /tmp/real-minimal.json
```

`--task divide_zero` limits either run to one task. Provider model IDs and
endpoints come from the normal application settings. Set `OLLAMA_MODEL` to the
exact installed tag (for example, `gemma4:12b`) for both the app and evaluator.
An OpenAI-compatible endpoint can instead use `OPENAI_COMPATIBLE_BASE_URL`,
`OPENAI_COMPATIBLE_MODEL`, and optionally `OPENAI_COMPATIBLE_API_KEY`, then
`--provider openai_compatible`. Anthropic and official OpenAI remain available
when configured; hosted calls may incur charges. The JSON report records the
starting revision hash, provider/model, prompt version, runtime configuration,
tool trace, final diff, independent check
output, termination, usage, and latency. A single run is a baseline observation,
not evidence of a repeatable model improvement.

The manifest contains nine tasks: the five original bugfixes, `paginate_items`
(feature), `extract_label_normalizer` (refactor) and `positive_total_existing_failure`
(bugfix with an unrelated failing check), plus `retry_after_seconds` (feature).
Pagination specifies one-based
pages, validation (including bool),
empty/past-end behavior, input/element preservation and the existing helper contract.
The acceptance suite is outside the editable fixture. A weakened visible test cannot
replace those checks. Reference validation establishes a solvable fixture, not that
any model can solve it. New tasks are public development tasks and must not be used
as held-out evidence after tuning on them.

### Division acceptance correction (2026-10-29)

The division criterion now includes non-integral positive/negative quotients,
fractional operands, zero numerators and signed numeric zero divisors. The old
integer-only nonzero cases admitted floor division and integer truncation, which
violate the unchanged request for real division. Disposable mutant regressions
reproduce that false acceptance and require rejection under the corrected checks;
the original reference still passes. Requests, fixtures, visible tests and
references are unchanged.

This changes the development criterion bytes for future evaluations. Closed trials,
their archived runtimes/criteria, source-review conditions and published results
remain pinned to their original versions; they are not regraded or relabeled.
Compare exact criterion revisions before comparing scores. This fixes a measured
grading blind spot, not model behavior or the release no-go.

The refactor task extracts a shared `strip().lower()` helper from two callers.
Acceptance checks Unicode/error/duplicate/iteration behavior and replaces the
module helper to verify that both callers use the requested seam at call time.
This tests the explicit refactor contract without source similarity or AST matching;
adding an unused helper can pass visible checks while failing external acceptance.
It is one narrow refactor task, not a general assessment of design quality.

`retry_after_seconds` exercises a feature across parser and client interfaces. The
reference updates both modules; authoritative acceptance checks behavior at both
interfaces instead of counting changed files. Numeric boundaries, zero/leading
zeros, invalid/large inputs, canonical-header precedence, fallback identity and
header preservation are explicit requirements. Parser-only/client-only reference
changes fail acceptance. This deliberately supports two header spellings and
numeric seconds only, not general HTTP Retry-After/date compliance or provider
retry behavior.

`positive_total_existing_failure` requires a focused fix while preserving two
legacy files byte-for-byte. Its broad `pytest -q` intentionally retains an unrelated
failure after a correct patch; `pytest -q test_visible.py` is the passing target
check. Corpus validation checks focused visible tests and external acceptance,
not universal broad-suite success. External acceptance checks both behavior and
protected bytes: repairing the unrelated setting or weakening its test is rejected
even when the broad suite turns green. Acceptance can therefore pass while the
observed broad check is failed. Results keep those facts separate; no automatic
check attribution or answer-honesty scoring is added. Operators requiring a check
must explicitly choose a relevant command, with identical settings across compared
modes. This task does not override configured verification policy.

New result rows record the manifest's `category`; summaries include raw accepted/
attempt counts per category and mode, alongside unchanged per-task and aggregate
counts. These labels describe task intent, not a model-derived score. Supported
labels are `bugfix`, `feature`, `refactor` and `unclassified`. The original five
manifest entries are explicitly bugfixes. Missing custom/project metadata stays
unclassified; it is never guessed from a task name or answer. Old reports remain
unchanged, and old summaries without categories remain renderable. Compare equivalent
task sets, revisions and budgets; expanding the corpus does not improve old results.

A failed, timed-out or cancelled in-flight evaluation request makes aggregate
prompt/completion usage `null`: the backend may have consumed unreported tokens.
Earlier known usage remains in response traces; the summary excludes unknown
full totals from its median and counts them as unreported attempts. Zero remains
valid when the harness prevented every request or the provider actually reported
zero. This does not attest to backend billing. Historical reports remain unchanged;
earlier failure rows labeled zero need their original evidence/limitations.

The disposable directory prevents edits to the source fixture, but it is not a
security sandbox. Python tests and agent commands can access the host and network.
Use only these trusted fixtures until an isolated execution mode exists. The
minimal mode changes only the system prompt; it retains the same application
runtime, tools, settings, and task copy for a same-model comparison.
To compare verification behavior, set `PROJECT_CHECK_ARGV` and
`REQUIRE_VERIFICATION_BEFORE_FINISH` identically for both runs. The exact argv
is recorded in each report's runtime configuration.


`CODING_TOOLSET=full` is the default. The opt-in `whole_file` experiment omits
only `replace_text`, with a matching shared harness edit instruction; minimal
mode keeps its original prompt. Application and real evaluations use the same
selection. Reports record the toolset and exposed schemas. Keep settings and
budgets identical when comparing toolsets, and use unique report paths. The
initial comparison did not establish a reliable gain; see `evals/BASELINES.md`.

New reports identify `trace_format: normalized-turn-v2` (older v1 reports remain unchanged) and include an ordered
`turn_trace` per attempt, alongside the existing `tool_trace`. The shared loop
collects this only when a caller supplies a trace list; application API payloads
and model context are unchanged. Each record carries a zero-based iteration:

- `request`: message count, estimated prompt tokens (not a tokenizer count), and
  requested output allowance (`null` means no harness cap). The allowance also
  respects `runtime_config.max_request_output_tokens` when enabled, separately
  from the cumulative turn output budget. Old runtimes had no per-request cap. The first request
  includes a deep copy of the initial normalized messages, including system
  instructions; subsequent requests do not duplicate history. Tool schemas
  remain in `runtime_config.tool_specs`. New records also include a metadata-only
  `footprint`: message/tool-schema heuristic tokens, the adjustment from the
  previous reported prompt anchor, and the ten largest message payloads by
  heuristic UTF-8 bytes (ties use ascending message index). The message estimate
  includes overhead; message + schema + anchor adjustment equals the gate estimate.
  `prompt_estimate_bytes_per_token` records the configured ratio (default 3);
  older footprints used the fixed ratio 3. New `runtime_config` records it too,
  along with `context_limit_policy` and `effective_max_context_tokens`; configured
  `max_context_tokens` remains separate. `configured-min-v1` caps Ollama by its
  explicit requested window even if the harness cap is zero. Historical reports
  retain their original runtime behavior; backend compliance is not attested.
  Payload size is not provider tokenization or wire size. Rankings contain message
  index, role and tool name, excluding raw content, arguments, paths and call IDs.
  Older request records parse with `footprint=null`; no footprint is invented.
- `budget_stop`: stage is `pre_request` for a prevented request or `post_response`
  for a token stop after a response, before tool dispatch/final completion. Old
  records default to `pre_request`. Records include status/reason, observed usage,
  configured limits and `requested_tool_count` (zero for pre-request stops).
  Post-response footprints describe current normalized history including the
  assistant reply; that view is not a request that was sent. The post-response
  gate itself uses reported usage and its conservative follow-up lower bound,
  not necessarily the displayed history estimate. Requested calls still receive
  existing not-executed tool records. Wall-time and other terminal conditions
  remain outside this record's coverage; final status/reason stay authoritative.
- `response`: full normalized content, requested tool calls with IDs and batch
  boundaries, finish reason, reported usage, model, and provider latency. Captured
  before completion/budget gates, including whitespace replies and calls that a
  truncation/filter/error finish reason prevents from executing.
- `recovery`: the exact runtime-injected user message.
- `tool_result`: the exact model-visible tool message, linked by tool-call ID,
  including dispatch rejections and results from the final batch even if no
  further model request occurs. Structured results/errors/latencies remain in
  the legacy `tool_trace` in dispatch order.
- `request_failure`: exception class only, including timeout and cancellation;
  provider exception text is excluded. An interrupted process can lose the
  current attempt because reports are still written after each attempt.

For ordinary requests, reconstruct the next transcript from `initial_messages`,
then append assistant messages from responses and the `recovery`/`tool_result`
messages in order. Suppress assistant tool calls for `length`, `content_filter`,
and `error`, as the runtime does. A response received after the wall deadline is
retained as evidence but need not enter model context; the terminal status and
reason remain authoritative. Runtime fallback answers are in `answer`, not
provider response records.

This is complete **normalized loop evidence**, not provider wire traffic or
hidden reasoning. Raw provider payloads, transport headers, and exception text
are deliberately excluded. Content already visible to the model is retained
without redaction; use trusted fixtures and control access to reports. Collection
is in memory, adds copying/serialization overhead, and is bounded by configured
turn limits rather than a separate trace byte cap. Persistence, crash recovery,
and context compaction remain separate work. Historical reports omit this field
and cannot be retroactively used to count every blank recovery or model reply.

### Dispatch deadline and latency scope (2026-10-12)

The shared configured runtime now starts `max_turn_wall_seconds` after admission
and transcript guards, before configuration capture, prompt refresh and the first
workspace snapshot. Those stages consume the same allowance as model/tool work;
the clock is not restarted when entering the loop. Final review and owned work
still settle even after the dispatch deadline. This is not a strict return bound.

Historical evaluations before this slice used a deadline beginning inside the
loop, excluding initial setup/snapshot time. Keep that distinction when comparing
recorded budgets and revisions. This change was validated offline; no model-quality
comparison or inference was run for it. The real runner's existing `latency_ms`
measures the whole configured call, including snapshots/diffs, and remains unchanged.
Application envelopes retain loop-only `latency_ms` and now expose separate nullable
`workspace_review_ms`; neither application field represents end-to-end HTTP latency.


Footprints are opt-in trace evidence. API response fields, model context, token
limits, recovery and acceptance scoring are unchanged. Metadata excludes payloads;
other trace entries still contain sensitive source and are not credential-redacted.
Trace computation consumes processing time under existing dispatch deadlines.
No compaction, profile calibration or model-quality improvement is established.

### Normalized request identity (2026-10-20)

New opt-in `request` records include nullable `identity` with format
`normalized-chat-request-v1`: SHA-256 digests for the complete rendered normalized
message list, exposed tool schemas, and their combination with the actual
`max_tokens` allowance. The loop computes these after request gates and before
calling the provider, on every attempted request, including one that later fails.
Prevented requests still have no request record. Legacy or non-JSON inputs have
unknown identity (`null`); no digest is invented from stringified unsupported data.

Canonical JSON sorts object keys, uses compact separators and UTF-8 without ASCII
escaping, and rejects non-finite numbers. Arrays retain order. Message digests
include all normalized `ChatMessage` fields, including roles, text, tool arguments,
call IDs and tool names; schema digests include all `ToolSpec` fields. The combined
digest covers an object with `format`, `messages`, `tools`, and `max_tokens` keys.
Paths, newlines and call IDs are never removed or normalized. Equal base-prompt
profile hashes are insufficient: workspace paths, memories, check instructions
and prior messages can change the rendered request. The Duodoro feedback cohort
demonstrated this distinction; its old records and outcomes remain unchanged.

Compare fingerprints before interpreting a paired experiment, and inspect any
difference instead of silently stripping arm/path information. This is observation,
not an automatic comparison admission gate or a fix for workspace-path confounding.
No new comparison is frozen or run by this change. Matching fingerprints identify
only these normalized inputs; they exclude provider/model identity, temperature,
provider-specific serialization/options, server defaults and hidden state. Preserve
the separate runtime/model/configuration evidence. They do not prove deterministic
sampling, request delivery, backend equivalence or quality improvement. Hashes do
not provide secret redaction or access control; raw traces remain sensitive.
Configured profiles do not gain these fields. New API runs now retain a separate
compact budget journal, described below; older API archives do not acquire it.
Client comparisons still need equivalent-input preparation before a future freeze;
the journal records actual attempts and is not paired preflight enforcement.

### Durable API budget evidence (2026-10-20)

New admitted runs initialize `RunRecord.budget_evidence` with an empty journal in
the execution-profile transaction. The shared loop synchronously observes numeric
request/usage/token-gate evidence without allocating a full trace. The API commits
each observation before onward inference/tools; a save failure stops execution
with 503 and retires the live session. Earlier edits or unreported usage can still
exist. The normal run detail/list routes and CLI `run --json` expose this evidence
under their existing owner convention; final TurnResponse fields are unchanged.
The panel and formatted CLI summaries do not yet render this journal.

The retained prefix is capped at **100 observations**, with `observed_events` and
`events_truncated` indicating overflow. A null journal means legacy/not admitted,
not zero work. An empty initialized journal means no observation was committed yet,
not proof of backend inactivity. A missing response/failure after a request may
mean process death or unsaved evidence. Startup retains the prefix while marking
the run interrupted; keyed retries never append or dispatch again.

- `request`: iteration, message count, normalized identity, output allowance and
  heuristic/anchor/combined prompt estimates. It precedes the provider attempt;
  it proves neither delivery nor remote execution. Null output allowance means
  harness-uncapped only on a request; other event kinds do not record an allowance.
- `response`: that response's provider-reported prompt/output usage, plus cumulative
  known totals after recording it. Missing components stay null, including when
  a configured usage guard then stops the turn. This is not billing attestation.
- `request_failure`: request did not return a normal response. Per-request and
  cumulative usage here are unknown; prior response observations retain the known
  prefix. No exception text, arbitrary exception class or provider payload is saved.
- `pre_request_stop` / `post_response_stop`: the token gate prevented a request or
  stopped after a response. Estimates describe the transcript at that observation;
  a post-response transcript includes the assistant but not pending tool results.
  Compare the separate saved profile's limits and final completion reason. Wall,
  cancellation, tool and iteration stops are not additional token-stop events.

No message bodies, paths, tool names/IDs/arguments/results, largest-message rows or
raw errors enter the journal. Fingerprints are hashes, not credential redaction or
access control. This does not make existing full run/transcript evidence secret.
Collection and SQLite commits consume the existing dispatch deadline; cleanup and
final review can still exceed it. The shared evaluator's full opt-in traces remain
separate, with the same runtime/budget logic. This adds no token policy, context
reduction, resume permission, automatic check, comparison or model-quality claim.

Named startup profiles use the same `Settings` binding as the API. The evaluator's
explicit provider must match the selected profile; no fallback/selection occurs.
`runtime_config.model_profile` records only the selected name/provider/model and
canonical definition hash (null means no selected profile), alongside resolved
budgets/options. Historical reports are not rewritten. Profiles do not attest the
served model revision, tooling capabilities, tokenizer fit or coding quality.

Summary `accepted` counts refer to final artifacts passing independent acceptance.
New summaries separately record `completed_and_accepted` per mode, task and category:
only accepted artifacts with `completion_status="completed"` count. An accepted
patch that exhausted its budget stays accepted but is not a completed workflow.
This metric does not require a model-run passing check, nor assess answer claims;
inspect `verification_status`, raw checks and final answers separately. Human output
labels artifact acceptance and shows completed-and-accepted totals. Rendering an old
summary without the new field says `unknown`, never inferring it from acceptance.
Historical reports and their original summaries are not rewritten.

New real-task rows include `acceptance_output_truncated`: true if either grader
stream exceeded its 12,000-byte capture cap, or if concatenated stdout/stderr
exceeded the existing 12,000-character report prefix. False means neither capture
nor report clipping occurred; it does not certify task correctness or complete
process logging. Old/custom rows default to null (coverage unknown). Exit-code
acceptance is unchanged. A clipped prefix can omit failure names or pytest totals;
inspect this flag before treating the diagnostic text as a full explanation.
Historical reports retain their original output and coverage limitations.

The retry-header acceptance suite uses short explicit parameter IDs, including
long leading-zero/oversized inputs and non-ASCII digits. This keeps its known
feature failure names and pytest totals within the unchanged grader output cap.
The values, assertions and expected behavior are unchanged; the acceptance file
hash changes because pytest labels are part of its source. Historical evaluations
remain pinned to their original revision/criterion hashes and are not rewritten.
Offline regressions apply both archived feature diffs unchanged to fresh copies,
retain their four failures/24 passes and check that all four failure IDs fit in
unclipped output. This is artifact replay, not new inference or improved acceptance.
Other noisy failures can still exceed the cap; inspect output coverage metadata.
