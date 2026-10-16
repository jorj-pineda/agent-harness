# Five trusted coding tasks

These small Python bugfixes are the first real-tool evaluation corpus. Each task
has a starting fixture with a failing visible test, a reference fix, and
authoritative acceptance tests outside the agent's editable copy. The runner
checks that the original fails and the reference passes before every model run.
It grades the final files, regardless of the agent's answer or the visible tests.

Validate the corpus without making a model request:

```sh
uv run python -m evals.real_run --validate-only
```

Run one configured model against all five tasks, then repeat with the same model
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
  requested output allowance (`null` means no harness cap). The first request
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
