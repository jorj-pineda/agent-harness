# Invalid-hash feedback execution freeze

**Prepared, not executed.** This [immutable manifest](hash-feedback-execution-freeze.json)
implements the reviewed [four-cell protocol](HASH_FEEDBACK_PROTOCOL.md). It pins
actual new roots, both clean published runtimes, first inputs and the local
preparation/execution/audit recipes. Independent exact-head review and merge of
this freeze, then unchanged launch checks, remain required before inference.
This is the second PR of the protocol/freeze/results batch, not a coding result.

## Actual execution boundary

| Order | Cell | Arm | Clean runtime |
|---:|---|---|---|
| 1 | C1 | Prior diagnostic | e92f6a11043e6d7025de13e75fd0e01818c4f696 |
| 2 | T1 | Complete-hash diagnostic | e416ce5b4afd9fc643d3a9613cb1618cb1d2a94a |
| 3 | T2 | Complete-hash diagnostic | e416ce5b4afd9fc643d3a9613cb1618cb1d2a94a |
| 4 | C2 | Prior diagnostic | e92f6a11043e6d7025de13e75fd0e01818c4f696 |

Bundle root:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-hash-feedback-live-6grynhsy`

The common literal workspace is root/workspace. Each root/cells/C1,T1,T2,C2 has
its own prepared full fixture, settings and original request. Scheduled state and
inputs directories and root/outcomes are absent. Arm checkouts are root/runtime-C
and root/runtime-T. API/CLI processes receive the selected checkout via explicit
PYTHONPATH/cwd; construction verifies actual API, code-tool, Ollama and CLI module
paths/hashes. Separate-process first captures and all actual preflight constructions
exercise that selection. No runtime selector or production rollback was added.

Only the 12-line hash diagnostic differs in production code. The manifest pins
276 control and 277 treatment Python/config/asset source hashes, both Git commit/
tree identities and lock hashes. Added tests/docs are not model inputs. All four
scheduled first normalized identities, exact wire hashes and execution profiles
are identical. This proves prepared first-input parity, not identical later
histories, backend receipt/tokenization or a causal effect.

Original T02 request, source revision, acceptance/reference/setup hashes and modes
are unchanged. Every fresh scheduled fixture fails initial visible/external checks
and its reference passes both. Only server/payloadParsers.js may change; independent
acceptance still checks requested flooring plus preserved coercion/default/clamp/
Infinity/ID/mode/other-parser behavior and full eligible scope/modes. Visible checks
are narrower than acceptance and do not cover full Vitest/Socket.IO/server behavior.

## Configuration and environment

Keep 300 dispatch seconds, 64,000 total reported tokens, 8,192 output tokens,
24 tools and 12 iterations per cell. Four cells reserve 1,200 seconds, 256,000 total
reported tokens, 32,768 output tokens, 96 tools and 48 iterations. Request output
2,048/minimum256, context16,384, ratio3, temperature0, think=false, compact/full,
required original pytest adapter, captures on, plan/root injection/support/inspection
off and provider180 remain exactly as the protocol. File budget2 and repeated-call/
completion-retry limits remain fixed. No budget transfer, extra phase or warm-up.

The actual CLI owner is duodoro-hash-feedback-cli, API origin
http://127.0.0.1:18761 and CLI read-inactivity timeout600. It does not extend dispatch
time. Settings bind selected paths before allocation; malformed capture/runtime/
scalar metadata rejects before providers/stores are allocated.

The manifest pins local gemma4:12b digest, Ollama version, unloaded state, Python
and managed Node24.22.0 executable bytes, CLI entrypoint bytes/module resolution,
command PATH and installed distribution inventory. Node's digest is
b468fb9c67427c9d3aec07ee1105e63020923a631a9a2a00beb71a7ba3768bfd.
This does not establish Node22/full application compatibility or future package
compatibility. No new model, paid call, dependency installation or Docker action ran.

## Scripted actual-client preflight

Sixteen isolated first captures use fake transports, including the four scheduled
cells and separate normal/budget/missing-usage preflight roots. None allocates a
scheduled cell's state/input store. Within each root, all four arm inputs/wires/
profiles match. Raw bodies stay local, with directories0700/files0600 and unchanged
12-request/1MiB-per-file/8MiB-total recorder bounds.

Twelve actual CLI flows use the selected production allocation/factory with only
an explicit MockTransport override. Both arms exercise real read → malformed hash
→ explicit corrected hash → configured check → final response in the normal root.
The controls expose their original generic pattern failure; treatment exposes its
complete-hash instruction in actual subsequent model history. Four reference
artifacts pass visible checks and independent grade/regrade. The scripted provider
chooses the correction, so these are mechanics checks, not model recovery.

Four separate budget flows return synthetic reported usage above the total cap,
preventing the corrected edit before dispatch and preserving not_run checks.
They test an overrun stop, not exact T02 follow-up-estimate replay or a hard backend
cap. Four missing-usage flows retain incomplete/not_run and unknown aggregate
usage without edits. All twelve outcomes' 44 normalized/wire pairs match saved
request observations and offline production MockTransport serialization.

Each flow then restarts once with the default source factory and once with a guarded
GET-only review factory. Actual CLI GET returns identical saved JSON with no new
input files or model calls. All **36 owned API processes exited**, their shutdown
logs and released port were checked. The sixteen in-process capture lifecycles also
closed. The existing Ollama service remains; Gemma is unloaded. No browser was used.

## Guard, audit and staging checks

- Sixteen normalized and twelve wire mismatches block before forwarding; matching
  forwarding preserves the request object. Four review guards reject inference.
- Thirty-two missing/malformed identity, wire, scalar or runtime cases reject before
  resource allocation; four positive selected-arm factory validations still pass.
- Ten copied-evidence staging cases preserve current bytes/root and create no next
  outcome: changed current tree/root, existing next state/inputs, missing prior wire,
  changed review proof, incomplete input analysis, empty review/input proofs and
  missing prior audit.
- Missing paired input makes a copied audit incomplete and blocks next staging.
  Complete copied evidence stages once. Copied aggregate checks count all four
  scripted successes, while missing inputs/incomplete audits block aggregation.
  Original preflight outcomes and inputs remain unchanged.

One initial copied-evidence probe retained its original staging inode while the
workspace copy had a new inode. The root guard correctly rejected before the
intended missing-input check. The original recipe and copies are retained; a fresh
final probe updates both copied root proofs and passes. No actual client flow,
model input, scheduled artifact or inference was rerun/corrected for this probe.
The manifest pins that correction separately.

The bundle pins **20 local recipes and 1,799 preparation-evidence hashes**, including
private captures, actual CLI results/logs, grade/regrade/input analyses, runtime/
environment receipts, negative probes and exact G0 reports. These are local
plaintext evidence with host permissions, not process isolation or secure storage.
Do not rerun preparation/capture/preflight/finalization helpers into existing paths.

## Exact-runtime G0 and execution gates

Control passed **119 focused and 1,453 offline** tests; treatment passed **131
focused and 1,465 offline**, with five live tests deselected in each full suite.
Both passed Ruff, formatting and mypy. Both exact runtimes passed coding/support
scripted matrices **90/90 each**, at unique bundle-local paths; scripted-contract
labels and metric limits were inspected. The treatment includes eight new hash-unit
parameter cases and four API integration cases; no tests were excluded to make
control pass. The existing Chroma deprecation warning remains. Green mechanics
do not establish model quality.

The local manifest is byte-identical to the committed freeze, and the full frozen
launch check passed without dispatch. After independent review/merge, the operator
may invoke only one named execution.py cell at a time. It rechecks immutable
sources/environment/model/criteria/recipes/inputs and initial root/tree; existing
state/inputs/outcomes block replay. Subsequent cells require complete prior input,
archive/full-tree, grade/regrade, saved/restarted JSON and cleanup evidence. Current
workspace equality is checked before replacing only this owned disposable root.
Uncertainty leaves remaining rows unrun; it never authorizes workspace repair.

The separate results PR must retain all four denominator rows and report actual
malformed-hash activation, rereads, corrected proposals, applied edits, pre-feedback
normalized/wire divergence, checks, independent acceptance and unknown usage.
Immediate valid editing without feedback is nonactivation. First-input parity
alone cannot attribute a later success to the diagnostic.

Original cohorts, the 8/10 target, no-go/feature freeze, production defaults and
G2–G5 remain unchanged. No Duodoro working-checkout edit, repaired model patch,
extra coding turn, quality gain, new unattended envelope or release approval is
established by this unrun preparation.
