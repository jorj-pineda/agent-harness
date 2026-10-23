# Provider-timeout comparison results

The four frozen original-T09 CLI attempts are complete. **60 seconds: 0/2 accepted
artifacts and 0/2 completed-and-accepted. 180 seconds: 2/2 accepted artifacts and
2/2 completed-and-accepted.** The two longer-wait runs passed the configured check
and independent acceptance within the unchanged five-minute task allowance.
This supports the selected-task timeout hypothesis; it does not establish backend
cause, general model quality or a production default change.

The [protocol](PROVIDER_WAIT_PROTOCOL.md) and [execution freeze](PROVIDER_WAIT_FREEZE.md)
remain unchanged. PR #193 was independently reviewed and merged before inference.
Actual runtime was `6721cc9997a00a473cee39508aba8da169eb2758`; source/preflight/recipe
hashes still match. [Structured results](../../results/2026-10-23-duodoro-provider-wait-comparison.json)
retain all four outcomes, independent grades, safe input analysis, budget/progress
records, cleanup receipts and the separate read-only post-execution audit.

| Cell | Provider timeout | Saved outcome | Requests / responses | Independent artifact | Completion / check |
|---|---:|---|---:|---|---|
| C1 | 60 s | failed: read timeout | 5 / 4 | rejected, no edits | no final response |
| W1 | 180 s | finished | 7 / 7 | accepted | completed / passed |
| W2 | 180 s | finished | 7 / 7 | accepted | completed / passed |
| C2 | 60 s | failed: read timeout | 5 / 4 | rejected, no edits | no final response |

## What the longer wait changed

Both controls failed on call five after observed adapter HTTP-post times of
**60,053.321 ms** and **60,023.842 ms**. W1 and W2 returned from call five after
**95,628.052 ms** and **95,871.552 ms**, respectively, and continued to completion.
Their final response latencies were **210,363.246 ms** and **202,794.495 ms**;
workspace snapshot/diff overhead and cleanup are separate from those latencies.
Both successful artifacts changed only `client/src/lib/focusTags.ts`; protected
contents and modes were preserved. Independent grading and regrading agreed.

The only configured comparison variable was the provider timeout scalar, which
changes connect/read/write/pool together. Both arms retained production buffered
HTTP, local `gemma4:12b`, original task/acceptance, 300-second dispatch, 64K total/
8,192 output tokens, 24 tools, 12 iterations, 16,384 context and 2,048 request output,
compact/whole_file, temperature zero, thinking off and required verification.
No production defaults or task limits changed. First normalized inputs and exact
wire bytes matched across all cells; later generated trajectories may differ.
The observed association is not a controlled guarantee about backend causality.

## Usage and observation limits

W1 reported **34,124 prompt / 1,771 output** tokens; W2 reported **34,158 / 1,678**.
These completed-response totals are within the configured token allowances. C1's
known returned prefix was **13,528 / 199** and C2's **13,528 / 205**; their in-flight
usage and total usage remain **unknown**. No completion/check result is invented
for failed runs without a final response.

All **24 normalized/wire pairs** match saved request identities, allowances and
message counts, and production serialization reconstructed offline with
MockTransport. No additional backend request replay occurred. Tool-reply ID/name
continuity passed. Private input storage was 164,891 / 279,511 / 279,707 / 164,963
bytes by cell, within the equal recording bounds. All file hashes and owner-only
directory/file permissions match; no clipping or observer stop was recorded.

Input capture proves prepared inputs, not backend receipt or generation progress.
The recorder does not measure failed response bodies, first-byte/chunk timing or
backend logs. Adapter post durations include wire recording I/O, not pure inference;
synchronous storage and owned-process cleanup can delay cancellation/return.
The provider scalar is an inactivity timeout, not a hard request deadline.

## Preservation and cleanup

Each cell had exactly one saved run and CLI submission with its retained key.
There was no replacement, retry turn, repair, rescue, reversal or raw-input replay.
All four complete artifacts were archived, independently graded/regraded and kept
unchanged. Guarded GET-only restarts returned identical saved JSON without new
inference. Common-root staging required prior archive/review/audit and full current
root identity/tree agreement; all four input audits were complete.

The read-only final audit verified all 92 retained outcome hashes, frozen source/
recipe/evidence hashes, profiles differing only in the timeout scalar, archives,
acceptance/regrades, private inputs and saved reviews. All eight owned APIs exited
with full shutdown logs and released port 18760. Gemma is unloaded; the existing
Ollama service remains. Duodoro head/status/index/refs are unchanged. No dependency
installation, paid/new model, Docker action or Duodoro commit/push occurred.

Local raw artifacts and private plaintext inputs remain at:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-provider-wait-ph07ytu3`

No execution correction occurred. A preliminary read-only reporting query used
`tool_name` instead of the progress event's `tool` key and stopped; the final audit
uses the actual schema. No stored evidence, launcher or model attempt was changed.
The new audit recipe and receipt are separately hashed alongside the raw results;
they do not rewrite the committed unrun freeze.

## Disposition

This bounded comparison is complete. It shows that the longer existing provider
setting allowed two accepted completed workflows on this selected public task
within the agreed task envelope. It does not establish a general timeout fix,
backend cause, held-out/two-model evidence or a supported unattended envelope.
No automatic application promotion or further inference follows. The original
cohorts, 8/10 release target, no-go and G2–G5 remain unchanged. Any adoption or new
validation needs its own explicit scope and independent acceptance.

This docs/data report uses the frozen G0 and new retained-outcome audit. Offline
unit tests and the scripted preflight remain mechanics evidence, not coding-quality
measurements; they are not counted as model successes.
