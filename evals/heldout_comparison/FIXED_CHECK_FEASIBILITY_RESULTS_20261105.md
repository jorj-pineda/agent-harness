# Fixed-check feasibility: check invoked, replacement edits refused

One fresh direct/no_plan_check ISO-week attempt invoked the configured project
check, then attempted two replacement edits with extra JSON escaping. Both edits
were refused; the original fixture files remained unchanged. Six complete model
responses exhausted the six-iteration cap. Original external acceptance and an
independent copied-workspace regrade reject the artifact: **0/1 accepted and
0/1 completed-and-accepted**. The attempt is closed without retry or manual repair.

The fixed-check interface was exercised by this model, but this single known
retired-v1 development attempt establishes no causal improvement, held-out gain,
default-adoption recommendation or release readiness. The raw catalog still says
held_out; that historical identity does not make this tuning task held-out evidence.

## Authority and configuration

[PR #371](https://github.com/jorj-pineda/agent-harness/pull/371) merged exact reviewed
head `4c2d40c9f00ffaef960f167a42eb4a01a14ecda0` at
`6eb449ef5fe43818cc77fe4a39c44f8a8fc878f4`. Independent Luna reviews completed the
code and private-source scopes without blockers; a separate executed verifier
checked 29 selected evidence hashes. Retained G0 passed 3,055 offline tests and
final working gates passed 3,069, with 92 recipe contracts, static checks and both
90-result scripted matrices. Both CI checks passed. Fresh hash verification and
check-only passed before the actual parent issued the receipt under Jorge's
next-eight-PR authorization. Private provenance records these checks; affirmative
JSON and local Git consistency alone do not authenticate parent authority.

The [freeze](FIXED_CHECK_FEASIBILITY_FREEZE_20261105.md) and
[protocol](FIXED_CHECK_FEASIBILITY_PROTOCOL.md) selected only ordinal 2:
ho_iso_week / harness / repeat 1, direct/no_plan_check, keep-alive 300 seconds.
The other 15 captured fixtures were never planned live attempts and stayed unused.
The clean retained runtime is `d3f67e0f6a57308fb27c97f4e4293bf58375ad56`;
prompt version is `coding-v1-direct-v1-fixed-check-v1`.
Ollama 0.33.2 / gemma4:12b used the pinned image digest
`4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`.

Original criteria and budgets stayed fixed: 150 dispatch seconds, 32,000 total
and 4,096 output tokens, 12 tools, 6 iterations, request cap 2,048/minimum 256,
context 16,384, temperature zero and thinking off. Keep-alive 300 is separate
from the dispatch budget. No warmup, control, second attempt, resubmission,
download/install, Docker, hosted/paid inference or user-service management occurred.

## Observed workflow

The six tool calls were tree, two file reads, run_project_check with empty
arguments, and two replace_text attempts. The configured visible pytest check ran
and failed with exit 1. No raw-command or planning tool was offered or invoked.
Both replacement calls supplied literal backslash-n and escaped quote sequences
where the file had line breaks and quotes; exact matching found zero occurrences.
The tool explained the mismatch and preserved the arguments without rewriting or
applying them. The second call repeated the same failed replacement. The run then
reached its iteration cap without a final answer or a successful post-edit check.
This identifies a remaining edit-argument obstacle; it does not prove a proposed
repair would help, and the proposed new text was never an accepted patch.

Six complete native responses report **12,478 prompt and 440 completion tokens**
(12,918 total), with six loaded point snapshots. There is no cancelled request or
unknown aggregate usage. Loaded snapshots do not establish continuous residency,
cache reuse, authenticated model identity or a residency effect.

The sole ledger cell is recorded and the cohort finished; the command exited 0.
Those bookkeeping states do not mean coding success: the runtime termination is
iteration_limit, completion status budget_exhausted and verification failed.
There are zero unrun planned cells and 15 verified unused capture fixtures.

The command ran from 2026-11-05 08:06:18.848159 UTC to 08:08:35.127690 UTC,
**136.279531 seconds** outer time. Recorded dispatch latency is **87.606977 seconds**.
Owned lifecycle checkpoints record metadata readiness at 6.276691 seconds,
body return at 94.791824 and final verification at 99.567582 from that lifecycle's
own start. These are distinct scopes, not additive durations or an end-to-end
hard deadline. No matched timing contrast or causal speedup is inferred.

## Preservation and closure

The post-run audit verifies clean runtime/catalog seals, selected runtime and
dependency pins, archived capture and exact first wire bytes. Original fixture
bytes and modes stayed unchanged. The submitted inventory gained five pytest
cache files, separate from the result's empty tracked-change lists. All 15 unused
fixtures verify and remain unadmitted. The independent original-criteria regrade
on a disposable copy returned 1; it is not another model attempt.

Owned cleanup records settled/passed true, forced false, returncode 0 and no
errors. The audit observed the owned process group absent and port 11447 bindable;
a subsequent read-only check found no listeners on 11434 or 11447. These are
bounded historical observations, not continuing leases or process isolation.
No-go, 8-of-10 and G2–G5 remain open. No further inference is authorized by these
results, and prior attempts stay closed.

Retained root:
`/Users/jorge/.local/share/agent-harness/trials/fixed-check-feasibility-lcs6urx1`.
All 46 indexed receipt, command, native, backend and audit files match selected
hashes. WORK_LOG.md and READINESS_HANDOFF.md retain the execution and closure.
This results change is documentation-only; no fresh full-suite run is claimed.
Preparation's scripted accepted row remains mechanics evidence.

| Retained evidence | SHA-256 |
|---|---|
| `results-evidence-index.json` | `63266b735794fa7d844c1679076b0f1f8e432e4e2ef0fa0fdd7d35e7fb028fcf` |
| `execution-result.json` | `845a0287f3d67bc0b80f3f8e22efbb561efe6a4e86d198de95325b78af921dd4` |
| `candidate/run/report.json` | `8503c9f5acd2799f6303c8b0846c31317ed5d8f50e5a53da24eb744e168451a2` |
| `results-audit.json` | `fa36ce45ac06ac0de68a842c6014858f868a06d57c285f9d1982a355f7ea3067` |
| `regrade-02.txt` | `edf845eb27f07ecf6cf05780be0185122d19e37609a8fe0b111c9d110d1a6c97` |
| `audit-results.py` | `2dc6bea4c4fa49befa43de8f0829be9cf4f6cbfe1e11894e612067833aa285f6` |
