# Bounded Ollama transport comparison: preparation

**Not an execution freeze; no inference ran in this slice.** The latest
[diagnostic cohort](PROVIDER_DIAGNOSTIC_LIVE_RESULTS.md) failed T05/T09 near the
configured 60-second read timeout before edits. HTTP duration was observed; backend
cause and in-flight usage remained unknown. Investigate the specific hypothesis
that non-streaming output can leave the HTTP connection inactive long enough to
hit its read timeout while a streamed response might deliver chunks sooner.
This is a hypothesis, not a diagnosis or established remedy.

## Experiment boundary

`transport_probe.ProbeTransport` is evaluation-only and is not reachable through
application settings. Both arms keep the production Ollama adapter's message/tool/
option serialization and final normalization. The wrapper re-encodes both bodies
identically except `stream`: false for buffered control, true for treatment. Both
arms use the same 8 MiB **decoded body byte** bound. Stream assembly additionally
caps nonblank frames at 32,768 and complete tool calls at 64. Record any boundary
stop; those experimental parsing constraints are part of treatment, not production
policy or proof of a pure one-variable model effect.

The streamed treatment accumulates content and complete function-call objects as
Ollama documents. It rejects malformed/fragmented calls, explicit errors, invalid
UTF-8, missing/duplicate terminal records and data following terminal output.
It waits for terminal `done: true` **and HTTP body completion** before returning
anything to the production normalizer. No partial tools execute. Only the terminal
record supplies usage/finish metadata; missing counts remain unknown. Cancellation
closes the owned response; this is not a remote-generation termination guarantee.
No prompt, default, recovery rule or governed action limit changes.

Each request supplies an external bounded numeric observation: mode, duration,
first decoded body-chunk arrival, largest received-chunk gap (including initial
wait), chunk/decoded-byte/frame counts, terminal presence and fixed outcome.
These are client/transport observations, not wire-packet/token timings, backend
queue/generation evidence, billing or model-quality scores. A failed tail has no
last received chunk, so its final silent interval is not a measured inter-chunk
gap. Content, function arguments, exception strings and URLs are excluded. Raw
stream frames remain transient in memory, not durable trace artifacts. Byte bounds
are not CPU/RSS limits or process isolation. The observer is called only after
attempted response cleanup; failed evidence recording must stop the invocation, not silently
permit tools.

The production API profile cannot attest this evaluation-only transport override.
A future freeze must record each actual mode, code hash and observer recipe
separately; never relabel the configured profile as backend attestation.
The experimental transport's duration includes buffering, stream parsing/assembly
and cleanup. Because that work runs inside HTTP post, the existing adapter's call
duration includes it too in this experiment; do not describe these numbers as
parsing-excluded network time or compare them as pure inference. Preserve that
scope override alongside the unchanged API field, not as a rewritten historical
measurement or changed production description.

## Proposed fixed comparison

Prepare exactly four original **T09 tag_filter** CLI attempts, ordered buffered-1,
streamed-1, streamed-2, buffered-2. This selected public task had no edits before
its observed timeout. Do not substitute tasks or rerun failed cells. All four use
fresh disposable starting bytes/modes at the same literal workspace path, separate
stores and unseeded memory. Before each attempt, explicitly stage only that owned
common path from its immutable prepared fixture; never touch Duodoro's checkout,
Git index or an unchecked workspace. Record each staging/root identity and preserve
all final artifacts before the next staging action.

Pin Duodoro `5324cabba4ad3c8ad2e1cd1480218cf087d9b741`, original exact request,
allowed path, independent acceptance/reference and visible test, local Gemma
identity, compact/whole_file, thinking off and temperature zero. Keep each attempt's
approved 300-second dispatch, 64K total/8,192 output/24-tool/12-iteration limits,
16,384 context, 2,048 request output, 256 minimum output, 60-second provider timeout
and one completion recovery. Keep required configured verification/captures and
600-second CLI response wait. Four attempts reserve at most 20 dispatch minutes;
review/cleanup/storage can exceed that. No paid/new model, dependency installation,
Docker, retry, manual rescue or extra turn. The same timeout number has a deliberate
different transport effect: it bounds read inactivity per chunk, not whole-request
elapsed time. Streaming can run longer than 60 seconds; the shared 300-second
turn deadline remains in force. Do not call this equal hard per-request time.

## Gates before inference

1. Independent code/protocol review and merge. Actual fresh inputs/runtime G0 and
   a separate reviewed execution freeze remain required. This document does not
   authorize inference or alter the historical cohort/freeze.
2. Validate initial-fail/reference-pass acceptance and full preservation baselines.
   Capture each first normalized input and wire-body hash without inference, at
   the common literal root with exact CLI create/capture settings. All four
   normalized identities must match; decoded first wire bodies may differ only
   in the intended stream flag. Reject drift before forwarding. Later generated
   histories/tool-call IDs may diverge and must not be invented as identical.
3. Exercise actual CLI submission, real writes/checks, failure/partial-stream
   rejection and saved review after restart through the exact comparison factory
   and transport. Use synthetic providers/local HTTP only for preflight and label
   them correctly. Exercise default resource allocation/cleanup without inference
   too. Do not repeat the earlier separate-factory preflight mistake.
4. Freeze runtime/tree/lock, source/criteria/reference/settings/executables, actual
   roots/stores, model/environment, all observer/launcher/audit recipes, per-cell
   modes and staging/first-input guards. Run focused/full offline/static gates and
   both scripted matrices at the exact runtime; independently review/merge the
   actual manifest before launch. Drift blocks without silently refreshing inputs.

## Reporting and retention

Report all four scheduled outcomes, including setup/stream parsing/byte-bound
failures and unknown usage. Keep artifact acceptance, workflow completion/checks,
resource evidence and transport observations separate. Grade/regrade unchanged
archived artifacts externally before staging the next cell. Check frozen hashes,
protected bytes/modes, one run per cell, no resubmission, saved-run equality after
review-only restart, complete observation journals and explicit missing evidence.
Stop owned APIs and unload evaluation Gemma after each attempt; preserve services.

Publish raw counts by arm, activation/first-chunk evidence, every intervention and
limits. A streaming success alone does not establish causal coding improvement,
prove why historical calls timed out or justify a production default. Two repeats
on one public task are not held-out/two-model evidence. Retain no-go/G2–G5 and all
original outcomes. Shipping transport changes, raising limits or more inference
requires a separate evidence-based decision; do not automatically retain the
experiment as a product feature.

## Protocol references and offline evidence

The official [Ollama streaming description](https://docs.ollama.com/api/streaming)
defines NDJSON and the stream flag; its [tool-calling examples](https://docs.ollama.com/capabilities/tool-calling)
accumulate content and complete calls before tool use. [HTTPX timeout documentation](https://www.python-httpx.org/advanced/timeouts/)
defines read inactivity per chunk. These describe protocols, not this model's
observed behavior. No SDK/package installation was needed.

Tests cover production-adapter serialization/normalization, Unicode byte splits,
complete tool assembly, terminal-only/missing usage, bounded parsing, malformed/
incomplete/error output, HTTP failures, partial timeout, cancellation/closure, cleanup/evidence-recording failure and
real-file API preservation before terminal completion. A real loopback synthetic
server demonstrates inactivity semantics under the same 0.5-second timeout:
buffered output waits 1.2 seconds and times out; regular chunks allow a 1.2-second
response to complete. This is a scripted HTTP contract, not Ollama inference,
a historical-request replay or a coding-quality result. An initial API test used
a provider name not registered by its fixture; correcting that input exercised
the intended partial-stream rejection before the passing gate.

Final validation: 22 new transport regressions; 62 focused passed with two live
tests deselected. The final full gate passed 1437 passed, 5 deselected, 1 warning in 74.55s (0:01:14). Ruff, formatting and
mypy passed. Both scripted matrices passed 90/90 escalation decisions with contract
labels and metric limits inspected. The earlier gate caught a Ruff cleanup-style
issue after pytest passed; it was corrected before this complete passing gate.
These checks do not establish real-backend activation or task-quality improvement.
