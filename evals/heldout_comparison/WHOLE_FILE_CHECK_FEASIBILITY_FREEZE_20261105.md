# Fresh whole-file fixed-check development feasibility freeze

Jorge requested continued cohesive PRs after the closed fixed-check attempt.
The interface and explicit evaluation binding are merged in #373 and #374.
This proposal selects one fresh whole-file fixed-check attempt. The actual parent
may issue its bound receipt only after exact-head independent review, completed
local gates, both CI checks and merge, followed by fresh launch checks. No actual
parent receipt, candidate admission or generation has occurred.

## One planned attempt

Follow the [whole-file fixed-check protocol](WHOLE_FILE_CHECK_FEASIBILITY_PROTOCOL.md). Only fresh
original residency ordinal 2 is planned: ho_iso_week, harness, repeat 1, keep-alive
300 seconds. Other captured fixtures are offline preservation inputs, never
planned live attempts. These are known retired-v1 development tasks, not a
matched or held-out comparison. All prior attempts remain closed.

Use local Ollama 0.33.2/gemma4:12b with digest
`4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`.
Select direct/whole_file_check, omitting replace_text, run_command and emit_plan.
Use write_file for full-file edits and run_project_check({}) for the configured
check. The configured command and other tools remain unchanged. Prompt framing
and version identify this interface;
this is not a claim that prompt bytes match the older no_plan run.

Preserve original task and independent acceptance, context 16384, thinking off,
temperature zero, 150 dispatch seconds, 32K total/4096 output tokens, 12 tools,
6 iterations, request cap 2048/min 256 and other original settings. Keep-alive 300
is not the dispatch budget. Setup, grading, preservation and cleanup remain
outside dispatch time. No warmup, control, extra attempt, manual repair,
substitution, resubmission or budget transfer. No downloads/installs, Docker,
hosted/paid calls, or user Ollama service/configuration/identity management.

## Selected runtime and preparation

Retained root:
`/Users/jorge/.local/share/agent-harness/trials/whole-file-check-feasibility-r7em1cuq`.
Clean detached runtime: `ad7175dbd2fd57a41929d9184127b24464350ffe`, containing
merged #374. Source checkout creation under umask 022 inside a private 700 root
preserves sealed catalog modes; artifacts use umask 077. All commands use the
existing external `/Users/jorge/Projects/agent-harness/.venv/bin/python` with -B,
explicit retained PYTHONPATH and recorded PATH. No source-local environment or
resolver is allowed. Selected dependency files and package versions do not prove
complete import/shared-library closure or host-state identity.

Fresh candidate and rehearsal each use independently verified APFS model copies
and exact-path staged fixtures. The read-only image source remains
a prior owned heldout-v1-image-t8rosvgt bundle; no old workspace or receipt is
reused. Each bundle captures sixteen first inputs through the production Ollama
adapter behind mandatory MockTransport, archives consumed fixtures, then stages
fresh ones at their original roots. Only ordinal 2 may become live work; the
other fifteen remain unused. Manifest-selected direct/whole_file_check feeds the
same validated Settings at capture, preflight and execution. Mismatch refuses;
selection changes manifest bytes and invalidates old approvals.

The existing single-cell executor retains preventive budgets, shared runtime,
original controls/grading, first-input/native-response/usage/ownership guards,
response observations and settlement. Results persist before cleanup; only clean
settlement records the ledger cell. Failure retains evidence and stops. No loop
or adapter is duplicated. Filesystem and resource observations remain bounded
and non-atomic, not process isolation or a continuing lease.

The runtime also records the received-HTTP refusal category, native request timing
and lifecycle checkpoints, with their documented scopes. These records improve
observation only; they do not change limits, establish cause, guarantee a return
deadline or infer historical timing. Native/lifecycle clocks have distinct origins;
UTC can jump and monotonic suspend behavior is platform dependent.

## Mechanical evidence

Both fresh bundles prepared successfully. One mandatory-mock whole-file fixed-check workflow
completed and passed original acceptance through shared execution, a real owned
metadata lifecycle and independent cell image. Offered schemas omit exact replacement, raw-command
and plan tools; the scripted oracle selected run_project_check with no arguments
and actual verification passed. Only rehearsal ordinal 2 was admitted. This is
reference-script mechanics evidence, not model behavior or a working model patch.

Twenty-two refusal probes passed, including legacy fixed-check/no-plan/full toolset mismatch,
profile/scope and pin drift, invalid receipts, withdrawal, occupied endpoint and
caller-failure cleanup. A GET-only observation recorded not_loaded with selection
300. All three owned metadata groups settled cleanly without force and were
observed absent; image receipts and independent inodes verify. No generation was
forwarded or actual receipt issued.

Retained G0 passed 3,087 offline tests and the working-tree gate passed 3,101,
five live deselected and one dependency warning each. Both passed Ruff, format,
mypy and both scripted 90-result matrices, with labels and metric limits
inspected. All 106 operator-recipe contract tests passed. The original validation
runs were interrupted by the server restart and are not counted as passes; their
partial logs and a restart record are preserved and indexed separately.

The copied launcher's check-only passed; missing execution arguments and default
mode refused as expected without candidate admission or generation. The machine specification binds candidate, recipe, G0, captured
settings/first inputs and 32 evidence files. Exact-head independent review,
both CI checks, merge, actual parent receipt and fresh launch gates remain
required; this proposal does not itself authorize generation.

## Parent-controlled execution and reporting

The machine recipe requires explicit mutually exclusive --check-only or --execute.
Check-only inspects clean runtime/interpreter, candidate/withdrawal, artifact pins,
exact G0 and free owned port, then returns without admission or generation.
Execute additionally requires the actual parent-selected receipt/hash and exact
reviewed/merged commits, checking the two-parent merge, runtime ancestry and
reviewed Git spec/recipe/G0/manifest bindings. Admission precedes transport
construction; retries are zero and environment proxy inheritance is disabled.
Local Git/JSON consistency cannot authenticate the parent, caller, review, CI,
argument provenance or authority. Only the actual trusted parent may issue its
receipt after verifying authorization, exact-head review, all local/CI gates and
merge, then fresh source/artifact/resource checks. Only owned 11447 may be managed;
preserve user 11434 and configuration/identity. Historical absence is not a lease.

Any terminal outcome closes this attempt. Preserve unknown submission/usage,
provider/identity/runtime/environment/criteria/preservation failures and failed or
forced settlement stops. Report original acceptance separately from completion,
actual edits/checks/refusals, known/unknown usage, scoped timings and loaded points.
HTTP errors are not model completions. Loaded snapshots prove neither continuous
residency nor cache reuse. Publish logs/results even if loading fails or the
artifact is rejected. No automatic default adoption, general quality, causal,
equivalence, held-out claim or no-go/8-of-10/G2–G5 closure follows.
