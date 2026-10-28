# Token recovery: two pairs observed, then an operator-caused stop

**2026-10-27 — Incomplete cohort; no policy ranking.** The
[execution freeze](TOKEN_RECOVERY_EXECUTION_FREEZE.md) merged in
[PR #257](https://github.com/jorj-pineda/agent-harness/pull/257) at
`58861c76e655b1e9e4d38ca37e563313774191e8`, after exact-head Sol approval and both
named CI checks on `1613f7bb6d4d74edd221b4e42fc5d97772f5be73`. Parent receipts and
fresh launch gates passed before one execution of the fixed operator at archived
runtime `65f6072711c7a12b1b6c5d34fabb1b9ba48cdb5f`. The
[sanitized rows and evidence hashes](../../results/2026-10-27-gemma-token-recovery.json)
retain every planned source and arm.

The development assistant then ran Ruff on the reporting audit script with the
retained runtime as its working directory. Ruff created `.ruff_cache` there.
The exact runtime-tree gate detected five added paths before S3 could start a
backend, API or model request and stopped the remaining cohort. This was an
avoidable development-tooling mistake, not a model failure or demonstrated
product regression. Every previously pinned runtime entry is unchanged; the
runtime directory nevertheless **fails exact equality**. Cache bytes, stop marker,
traceback and nonzero exit are preserved. Nothing was deleted to pass the gate,
and no source, arm or cohort was resumed, retried or replaced.

## Observed rows

| Source | Source outcome | First followup | Second followup |
|---|---|---|---|
| T02-S1 | Native 150-second wall stop; unchanged/rejected; usage unknown | C1: token gate, unchanged/rejected | P1: token gate, unchanged/rejected |
| T02-S2 | Native 150-second wall stop; unchanged/rejected; usage unknown | P2: token gate, unchanged/rejected | C2: token gate, unchanged/rejected |
| T02-S3 | Staged, then runtime gate refused before dispatch; initial artifact rejected | P3 unrun | C3 unrun |
| T02-S4 | Unrun: cohort stop | C4 unrun | P4 unrun |

Two sources and four followups ran: six actual model phases and sixteen provider
requests, with fourteen terminal responses. Both sources finalized with the exact
native wall-budget reason; their APIs/backends settled before recovery. Each
returned source prefix reports 1,964 prompt/22 output tokens. The cancelled second
requests leave aggregate source usage unknown. The protocol explicitly permits
these finalized wall stops; no provider read-timeout exception was used.

Among the **two observed pairs**, each policy has zero accepted followup artifacts,
zero completed-and-accepted workflows and zero rescues of an incorrect source.
Both policies still have four planned workflows, with two followups unrun each.
This is not a completed 0/4-versus-0/4 comparison, evidence of policy equivalence,
or a recovery-benefit result. The staged S3 artifact is an unchanged baseline,
not another model attempt or model-quality failure.

All four followups stop at native total-token gates. C's third response proposes
`emit_plan`, which is recorded as not executed after the stop; P reaches the
next-request estimate gate after its third response. No model edit or configured
check execution is observed. All seven audited source/actual artifacts—the six actual phases
and nondispatched S3 baseline—receive two original independent rejects plus a
matching reconstruction regrade, exit codes `[1, 1, 1]`. Eligible files/modes and
non-root directories remain initial; acceptance criteria and references are intact.

## Inputs, omitted history and costs

The two actual source first inputs match their frozen captures. Every arm's first
normalized/wire input matches its separate native preview, and all sixteen requests
have paired captures whose normalized identities match the saved events. All observed wire
transports use the fixed 180-second timeout in all four HTTP phases.

| Arm | First prompt estimate | First wire bytes | Reported prompt/output tokens | Saved run seconds |
|---|---:|---:|---:|---:|
| C1 | 9,009 | 27,569 | 23,880 / 115 | 115.632 |
| P1 | 5,683 | 17,482 | 20,051 / 67 | 100.055 |
| P2 | 5,683 | 17,482 | 20,072 / 67 | 99.033 |
| C2 | 9,009 | 27,569 | 23,880 / 115 | 114.654 |

These are arm-only observations. Saved run time includes setup/storage, not just
dispatch; bytes are not provider tokens. Smaller first inputs and these four
arm costs do not establish workflow efficiency or quality improvement.

Each source reads AGENTS.md lines 1–265 before its pending request is cancelled.
C retains that raw tool history and reads the parser's 174 lines in its own arm.
P retains the original user request and fixed parser lines 1–80, while omitting
three nonuser messages, including the raw AGENTS tool reply. Its arms read the
parser's 174 lines and parser test's 167 lines, without a fresh AGENTS read.
The source inspection obligation is not automatically waived by projection.
Recorded read coverage, retained user text and selected bytes are not evidence
of comprehension, instruction compliance or semantic preservation. Both projected
first drafts estimate 5,683 tokens versus 9,009 for refreshed full history, with
unchanged 2,048 output allowance; neither produces an edit or accepted result.

Source cost is charged to each policy but counted once physically. Because both
sources have unknown tails, aggregate policy-workflow and physical token totals
remain unknown. The physical **known prefix**, 91,811 prompt/408 output tokens,
is not the total. No workflow has demonstrated known budget success. Saved-run
elapsed sums are C source-charged 530.668 seconds, P source-charged 499.469 seconds,
and physical 729.755 seconds; summing policy totals double-counts sources. The
whole operator takes 1,315.423 seconds including gates, startup, cleanup, grading
and other overhead. None of these is a hard end-to-end dispatch allowance.

## Preservation and audit

The post-stop audit reconstructs and independently regrades each saved artifact,
preserving file and directory modes. Closed SQLite checks independently verify
source run/archive identity, checkpoint records/blobs and retained source facts
inside arm/preview/restart clones. Immutable parents still equal closed original
source stores. GET-only restarts preserve actual stores and forward no inference.
The nondispatched S3 directory contains only its scored artifact and descriptor;
there is no CLI intent, store, API or backend allocation.

All 22 owned API/backend lifecycles exit zero with absent registered groups and
free ports 18843/11447. API shutdown receipts record normal return, closing state
and zero unfinished tasks. Existing Ollama on 11434 remains untouched; backend cleanup verifies unchanged
identity-file metadata across each owned lifecycle. Duodoro head/status/index/refs are unchanged.
Paths, allowlists, ignored-content metadata and registered groups remain limited
ownership checks, not OS isolation or proof about arbitrary detached workers.

The audit deliberately **does not report a passed full post-stop runtime gate**.
It records the five cache additions and confirms that exact equality still refuses.
Every original runtime entry, all 21 frozen recipes, all 28,748 preparation files,
model/source-model hashes, environment and Duodoro fingerprints still match.
The public result binds 8,083 outcome/audit evidence files, including the audit
script itself and retained cache bytes. Summaries are publication outputs outside
their own hash map. Raw bodies and stores remain private at
`/Users/jorge/.local/share/agent-harness/trials/token-recovery-live-6_09n6up`.

Preparation G0, scripted rehearsals and Sol review remain mechanics evidence;
they do not establish model quality or certify later environmental invariance.
The next operator should keep development tooling outside an immutable runtime
and disable tool caches. That is a prevention lesson, not permission to weaken
this gate or rerun these rows. This cohort is closed at its mandatory stop.
Further inference needs a new justified authorization and freeze. Original
cohorts, defaults, no-go, the 8-of-10 target and broad G2–G5 remain unchanged.
