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

`--task divide_zero` limits either run to one task. Provider model IDs and the
Ollama endpoint come from the normal application settings. Anthropic and OpenAI
are also available when their API keys are configured; those calls may incur
charges. The JSON report records the starting revision hash, provider/model,
prompt version, runtime configuration, tool trace, final diff, independent check
output, termination, usage, and latency. A single run is a baseline observation,
not evidence of a repeatable model improvement.

The disposable directory prevents edits to the source fixture, but it is not a
security sandbox. Python tests and agent commands can access the host and network.
Use only these trusted fixtures until an isolated execution mode exists. The
minimal mode changes only the system prompt; it retains the same application
runtime, tools, settings, and task copy for a same-model comparison.
