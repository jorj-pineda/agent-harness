# Post-plan input diagnostic: preparation

**Not an execution freeze; no inference ran in this slice.** The completed
[transport comparison](TRANSPORT_COMPARISON_RESULTS.md) retained four fifth-call
read timeouts after a completed `emit_plan`, with zero received body chunks and
no edits. Streaming was activated but did not remove that failure. Its failed
inputs were fingerprinted, not stored; empty failed session archives cannot
reconstruct them. Do not retry or modify that cohort.

## Hypothesis and missing evidence

Inspect whether the request immediately following the plan contains an unexpected
message conversion, tool-result expansion or configured-option difference at the
production provider boundary. Retained request estimates are about 6,908–6,914
prompt tokens with 2,048 output allowed under a 16,384 context cap, not a measured
backend context overflow. File size, the preceding plan and zero received
chunks do not prove the model proposed a write or identify generation/queue cause.

The next experiment collects exact **inputs before dispatch**, so even a failed
call can be inspected offline. It does not change the prompt/tool interface,
provider request/response normalization, timeout, retries or expected artifact.
No diagnosis or runtime remedy is established by adding this recorder.

## Evaluation-only recorder

`request_capture.RequestCapture`, `CapturedProvider` and `CapturedTransport` wrap
one sequential production provider/HTTP transport pair. They are not selectable
through application settings. Both layers save inputs before delegating unchanged
arguments/the same HTTP request object. The transport does not read or alter
responses, stream output, add retries or synthesize model responses/usage.

- Each normalized record retains exact messages/tool-call IDs, tools (including
  None versus an explicit empty list), temperature, output allowance and the
  existing normalized request identity. Unsupported/non-JSON values stop instead
  of being coerced to text. Each wire record retains exact JSON body bytes, with
  no headers, endpoint/URL, timeout extensions, exception strings or responses.
- Use a new canonical absolute directory under the owned diagnostic root, created
  with mode 0700; exclusive files use mode 0600 and are flushed/fsynced before
  forwarding. Existing directories/files are rejected, never resumed/overwritten.
  Inputs are **plaintext and may contain sensitive model-visible content**. This is
  not credential redaction, process isolation, an external-writer lock or a
  power-loss/crash-atomic two-file transaction. Keep raw inputs local to the trusted
  disposable task. Do not publish message/tool-result contents or credentials.
- Fixed ceilings: twelve requests, 1 MiB per file and 8 MiB combined stored bytes
  by default; permitted test limits can only lower the count/overall ceilings.
  Byte bounds apply after serialization, not to CPU/RSS or temporary allocation.
  They are diagnostic constraints; capture/bound/storage failure stops the attempt
  and must be retained separately from a model/backend failure.
- A returned provider response requires one paired wire record. Missing/duplicate
  pairs or swallowed recording failures cannot return proposed tools. Overlapping
  calls are rejected. A capture preceding forwarding is evidence of prepared
  input, **not proof of backend receipt or execution**. Correlate it with saved
  request/failure observations and actual source HTTP activity.

Normalized capture I/O precedes the adapter's HTTP post; wire capture I/O runs
inside it. Both consume the existing turn allowance and can affect timing. The
adapter observation in this experiment includes wire recording overhead; it is
not pure inference or a historical timing replay. Synchronous write/fsync cannot
be forcibly interrupted by the async deadline and may delay cancellation. Response cleanup remains owned
by the production HTTP client, and close failures propagate normally.

## Proposed single attempt

Prepare exactly one fresh original **T09 tag_filter** CLI turn, local Gemma, using
production **buffered** HTTP. There is no treatment/control quality comparison.
Pin Duodoro `5324cabba4ad3c8ad2e1cd1480218cf087d9b741`, original exact request,
allowed path, visible setup, external criteria and reference. Keep compact/
whole_file, temperature zero, thinking off, 300-second dispatch, 64K total/8,192
output tokens, 24 tools, 12 model iterations, 16,384 context, 2,048 request output,
256 minimum output, one completion recovery, 60-second provider read timeout,
required configured verification/captures and 600-second CLI response inactivity.
No extra turn, scripted prefix, manual rescue, model/limit switch or direct replay
of a captured request. An unexpectedly successful run is retained too; the fifth
call or timeout is an observation, not a condition for replacing the attempt.

## Gates before inference and offline audit

1. Independently review/merge recorder and protocol. Prepare a new full disposable
   workspace with absent session/memory/input directories; preserve old cohorts,
   Duodoro checkout/index/refs and pre-existing work. Validate original initial/
   reference acceptance. Capture the new first identity without inference.
2. Exercise actual CLI submission, real tools/checks, a synthetic later timeout,
   exact normalized/wire pairing and saved review after restart through the same
   production-allocation source factory. Review must allocate no input recorder
   and dispatch no inference. Include bound/storage failure without forwarding.
3. Freeze actual runtime/tree/lock/sources/criteria/reference/requests/settings,
   inputs/profiles/first guard, executable/distribution/model identities and local
   launcher/audit recipes. Run exact G0 and both scripted matrices, then obtain
   independent review/merge of the actual freeze before the single submission.
4. After settlement, stop the owned API/unload Gemma, archive and externally grade/
   regrade the unchanged artifact, restart for GET-only saved review and stop.
   Retain every outcome; unknown delivery permits printed-key inspection only.
5. Offline, match each normalized identity to saved budget request events; pair
   exact wire bodies by sequence and compare them to the production adapter's
   serialization. Inspect roles, tool-result/name continuity, configured model/
   options, request bytes/heuristic footprint and changes around the plan boundary.
   Record matched/missing evidence, never invent missing payloads or backend counts.
   Original tool arguments/result contents stay local; publish numeric/hash/safe
   structural findings with explicit heuristic and timing limits.

No API/model call may replay the captured failed input as an extra diagnostic.
If the fifth request is absent or no failure occurs, report that outcome; do not
force it with replacement attempts or a scripted prefix. Any further causal
experiment needs separate scope and comparable resource reservations.

## Evidence meaning

Regressions use synthetic HTTP and real tools: failure retains the fifth input,
count/byte/storage stops prevent forwarding, forwarding preserves identities/
arguments/options, duplicate/missing evidence prevents response delivery, and
cancellation delegates cleanup. A scripted API write followed by timeout preserves
pre-existing/agent edits, input records and identical saved JSON through restart.
These are recording contracts, not model quality or a timeout diagnosis. Actual
client preflight/freeze and execution remain separate slices. Keep original
cohorts, no-go and G2–G5 unchanged; no production feature/default or release approval.

Final implementation validation: 16 new recording regressions; 60 focused tests
and 1,453 offline tests passed (five live deselected in the full gate), plus Ruff,
formatting and mypy. Both scripted matrices passed 90/90 with contract labels and
metric limits inspected. The existing Chroma deprecation warning remains. An
initial static check caught an unused test binding/formatting after focused tests
passed; those were corrected before the complete passing gate. These validate
recording mechanics, not the proposed hypothesis or model quality.
