# Bounded provider-timeout comparison: protocol

**Preparation only; not an execution freeze. No inference ran in this slice.**
The [offline analysis](PROVIDER_WAIT_ANALYSIS.md) motivates a specific hypothesis:
the independent 60-second provider timeout may prevent useful completion even
when the agreed 300-second task dispatch allowance has not been consumed. The
latest retained input showed no unexpected post-plan expansion/option drift;
backend cause remains unknown. This is a fresh configuration comparison, not a
retry/replacement of the completed cohorts or a replay of captured failed input.

## Fixed comparison

Prepare exactly four original **T09 tag_filter** actual CLI attempts, ordered
**C1:60, W1:180, W2:180, C2:60** seconds for `request_timeout_seconds`.
Production buffered HTTP is used throughout. Use existing validated settings;
change no provider/loop/prompt/tool code. The scalar changes connect/read/write/
pool inactivity timeouts, not only read. It is not a hard request deadline.

The candidate can wait longer **inside the same five-minute task budget**. Keep
300-second dispatch, 64K total/8,192 output tokens, 24 tools, 12 model iterations,
16,384 context, 2,048 request output/256 minimum, one completion recovery, compact/
whole_file, temperature zero, thinking off, required configured verification and
checkpoints. CLI response inactivity stays 600. Four attempts reserve at most
twenty dispatch minutes; owned cleanup/review/storage can exceed dispatch. No
production default change, wall-budget increase, paid/new model, dependency
install, Docker action or Ollama restart/debug-setting change.

Pin local `gemma4:12b`, Duodoro `5324cabba4ad3c8ad2e1cd1480218cf087d9b741`, original
exact T09 request, allowed `client/src/lib/focusTags.ts`, visible setup and independent
criteria/reference. Fresh full starting bytes/modes use the **same literal owned
workspace path** across cells; each has separate unseeded memory/session/input
stores. Before replacing that common path, require the prior attempt's unchanged
full archive, grade/regrade and restart review, plus current full-tree/root-identity
agreement. Later or unexplained edits block staging. Never stage/reset Duodoro's
working checkout, index or refs.

First normalized identities, temperature/output allowance and exact first wire
bodies must match across all four cells; provider timeout is not a body field.
API profiles should differ only in their recorded timeout scalar, after preserving
all other configured fields. State directories are deliberately isolated, not a
model treatment. Later generated messages/tool-call IDs may diverge. Require the
first guards before forwarding, rather than treating parity as a post-hoc label.

## Observation and pre-inference gates

1. Independently review/merge this protocol/analysis. Prepare new actual roots,
   empty stores, original initial-fail/reference-pass validation and full starting
   manifests. Do not reuse old failed stores, workspaces or private input files.
2. Use the reviewed evaluation-only input recorder equally in both arms: twelve
   requests, 1 MiB per file, 8 MiB combined, fresh private plaintext files, no raw
   payload publication. A bound/storage/recording failure is retained as its own
   outcome, not bypassed or replaced. Recording presence is not backend receipt;
   wire I/O contributes to adapter post timing and sync fsync may delay cancellation.
3. Exercise actual CLI read/write/check and synthetic later failure, paired input
   audit, default source allocation and guarded GET-only restart through the
   exact source factory for **both** settings. Confirm actual HTTP client phase
   timeouts 60/180 without inference, unchanged body/first identities, one-run
   delivery-key handling, cleanup and negative guards. No scripted live prefix.
4. Pin actual runtime/tree/lock/sources, roots/identities/prepared bytes, criteria/
   reference/request/settings/profiles, managed environment/model, all observer/
   launcher/staging/audit recipes and evidence. Run focused/full/static exact G0
   plus both scripted matrices. Independently review/merge the exact execution
   freeze and recheck unchanged launch inputs before any model call.

## Execution and scoring

Run C1, W1, W2, C2 once each. Submit the original request through the actual CLI
with checkpoints; retain its printed key. Unknown delivery permits GET inspection
only, never a new POST. No extra turn, rescue, patch repair, cancellation/fork/
reversal, raw-input model replay, timeout escalation after a failure or replacement
cell. A different path, early completion, wall stop or absence of a fifth request
is a retained outcome, not a reason to force the expected failure.

After each settled run, stop the owned API completely and unload evaluation Gemma,
keeping the existing service. Archive and independently grade/regrade unchanged
artifacts. Restart in guarded review-only mode, compare actual CLI saved JSON and
ensure GET-only/no-inference activity, then stop. Check before the next owned
common-path staging; preserve each complete artifact and all records separately.

Keep artifact acceptance, completed-and-accepted, observed checks, failure reason,
known/unknown usage, changed paths and configured timeout separate. Report all
four outcomes, including setup/observer/byte-bound failures. Preserve incomplete
analysis when inputs are missing, without inventing payloads or backend counts.
Compare available normalized/wire identities and safe structure offline through
MockTransport only. Do not forward a recorded failed body to a real backend.

The primary test is whether 180 yields acceptable completed artifacts within the
unchanged task envelope when 60 does not, with actual post durations indicating
what happened. A passing artifact without completed workflow stays a separate
count; unknown in-flight usage cannot be certified within token budgets. Stops
under the unchanged task deadline remain visible. No automatic implementation or
configuration promotion follows even if this selected comparison favors 180.

Two repeats per arm on one public task are not held-out/two-model evidence or a
supported unattended envelope. First-input equality does not guarantee identical
later trajectories, backend execution, tokenization or billed work. Report raw
counts and limitations, not a causal/general quality guarantee. Original cohorts,
8/10 release target, no-go and G2–G5 remain unchanged. These three slices are
protocol/offline evidence, reviewed actual freeze, and retained comparison results.
