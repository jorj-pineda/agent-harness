# Whole-file fixed-check feasibility: edited and checked, artifact rejected

One fresh direct/whole_file_check ISO-week attempt made a real edit, passed the
configured visible check and returned a final answer. Independent original
acceptance rejected the artifact: **0/1 accepted and 0/1 completed-and-accepted**.
The patch fixes ISO week-year boundaries but omits the explicitly requested
four-digit year padding: year 1 returns `1-W01`, not `0001-W01`. A separate
copied-workspace original-criteria regrade returned 1. The attempt is closed
without retry or manual repair.

This known retired-v1 development run establishes that this model exercised the
whole-file edit/check interface in this attempt. It does not establish a causal
improvement, held-out gain, default-adoption recommendation or release readiness.
The raw catalog retains its historical held_out identity; it is tuning material.

## Authority and configuration

[PR #375](https://github.com/jorj-pineda/agent-harness/pull/375) merged reviewed head
`fa58fb556e53fc89894e57e96c58d536d0374440` at
`b4a92344f36bc0b8b551c65962193f65a2815ffd`. Complete independent Luna reviews
covered the recipe/helper contracts, documentation, executed 32-hash verification
and private preparation/rehearsal/refusal/observation/settlement sources. Earlier
partial reviews were not counted as clearance. Retained G0 passed 3,087 offline
tests and the working gate passed 3,101; 106 recipe contracts, static checks and
both 90-result scripted matrices passed. Interrupted gate logs remain separate.
Both CI checks passed, then fresh hash verification and check-only passed before
the actual parent issued its bound receipt under Jorge's continued-development
authorization. JSON assertions and Git consistency alone do not authenticate
parent authority; private provenance records the actual prerequisite checks.

The [freeze](WHOLE_FILE_CHECK_FEASIBILITY_FREEZE_20261105.md) and
[protocol](WHOLE_FILE_CHECK_FEASIBILITY_PROTOCOL.md) select only ordinal 2:
ho_iso_week / harness / repeat 1, direct/whole_file_check, keep-alive 300 seconds.
The other 15 captured fixtures stayed unused and unadmitted, never planned live
attempts. Prior cohorts remain closed. Retained runtime is
`ad7175dbd2fd57a41929d9184127b24464350ffe`, prompt version
`coding-v1-direct-whole-file-v1-fixed-check-v1`. Ollama 0.33.2 / gemma4:12b uses
image digest `4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`.

Original task, criteria and limits remained fixed: 150 dispatch seconds,
32,000 total/4,096 output tokens, 12 tools, 6 iterations, request cap 2,048/minimum
256, context 16,384, temperature zero and thinking off. Keep-alive 300 is separate
from the dispatch limit. No warmup, control, extra attempt, resubmission, repair,
download/install, Docker, hosted/paid call or user-service management occurred.

## Observed workflow and acceptance

Five tool calls listed the workspace, read implementation and visible test,
wrote week_key.py, and ran run_project_check({}). No exact replacement, raw
command or plan tool was offered or invoked. The check returned exit 0; the
sixth complete response supplied a final answer. All six responses report known
usage: **10,406 prompt and 383 completion tokens**, 10,789 total. Six post-response
point observations recorded loaded; these do not prove continuous residency,
cache reuse or authenticated model identity.

The result records completion_status completed and verification_status passed
because the configured visible check succeeded after the edit. Separately,
termination is acceptance_failed and artifact acceptance is false. The external
check reports **7 passed, 1 failed**, rejecting four-digit year formatting.
The final answer accurately reports the ISO-year change and observed visible
check; this report does not turn that check into proof of all requirements.
No extra model request, prompt repair or automatic acceptance weakening followed.

The ledger has one recorded cell and a finished cohort; the command exited 0.
These bookkeeping states do not imply artifact success. There are zero unrun
planned cells and 15 verified unused fixtures. This closes the selected workflow,
not a completed matched comparison or the original release gates.

The command ran from 2026-11-07 19:50:57.617922 UTC to 19:53:04.605063 UTC,
**126.987141 seconds** outer time; recorded dispatch latency is **78.951958 seconds**.
Owned lifecycle checkpoints record metadata readiness at 6.276975 seconds,
body return at 86.138495 and final verification at 91.085005 from their own start.
These scopes have different origins and are not additive or a hard end-to-end
deadline. No timing contrast, residency effect or causal speedup is inferred.

## Preservation and closure

The post-run audit checks clean retained runtime/catalog seals, selected runtime
and dependency pins, archived capture and exact first wire bytes. The only changed
original fixture is week_key.py; the visible test stayed unchanged. Four pytest
cache files were added separately from the tracked modified-file report. All 15
unused fixtures verify. Independent original-criteria regrade on a disposable copy
returned 1; it is not another model attempt.

Cleanup records settled/passed true, forced false, returncode 0 and no errors.
The audit observed the owned process group absent and port 11447 bindable; a
post-exit read-only listener check found no listeners on 11434 or 11447. These are
bounded historical observations, not ongoing leases or process isolation.
No-go, 8-of-10 and G2–G5 stay open. No further inference is authorized by this
receipt or report, and no default adoption or general quality claim follows.

Retained root:
`/Users/jorge/.local/share/agent-harness/trials/whole-file-check-feasibility-r7em1cuq`.
All 46 indexed receipt, command, native, backend and audit files match their
selected hashes. WORK_LOG.md and READINESS_HANDOFF.md retain execution and closure.
This is a documentation-only results change; no fresh full-suite run is claimed.
Preparation's accepted scripted row remains mechanics evidence.

| Retained evidence | SHA-256 |
|---|---|
| `results-evidence-index.json` | `b91e2807ab48f55bbd20380f8eb6e03858b10b8cee77007ddd8e12c1af0122a8` |
| `execution-result.json` | `0b7345012ac88af047be3b8d63dd3fb4af763fa7e6022d2c6ae7a8e259563411` |
| `candidate/run/report.json` | `cd19cc0f4ddc435398443d65a93482d13fa7a23d903f4c331154ea820dd5b88b` |
| `results-audit.json` | `dd93a94aac5f0fe5d66f59eb2412e000b9bc4e37b8993d765a9ad3cc2d5a3964` |
| `regrade-02.txt` | `2ee7945a57957cd72cbc7782f09f56bf74e4e1483d9f7720d585d2b106996c2f` |
| `audit-results.py` | `567d60f953624dc1441c4d6cd886dc25808972d6d07ce5345e7c79cc32a450b9` |
