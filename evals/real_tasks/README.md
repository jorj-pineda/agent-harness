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

New reports identify `trace_format: normalized-turn-v1` and include an ordered
`turn_trace` per attempt, alongside the existing `tool_trace`. The shared loop
collects this only when a caller supplies a trace list; application API payloads
and model context are unchanged. Each record carries a zero-based iteration:

- `request`: message count, estimated prompt tokens (not a tokenizer count), and
  requested output allowance (`null` means no harness cap). The first request
  includes a deep copy of the initial normalized messages, including system
  instructions; subsequent requests do not duplicate history. Tool schemas
  remain in `runtime_config.tool_specs`.
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
