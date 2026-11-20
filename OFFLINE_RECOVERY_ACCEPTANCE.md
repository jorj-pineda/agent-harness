# Bounded offline executable recovery sequence

Selected 2026-11-19 from the [workflow evidence map](WORKFLOW_ACCEPTANCE.md).
This specifies scripted-provider mechanics on disposable workspaces, not G2
closure, independent real-model coding acceptance or execution approval for any
frozen comparison. No backend or paid inference is involved.

## Why this case

The existing [combined CLI recovery test](tests/test_cli_recovery_workflow.py)
uses Typer's in-process runner and patches HTTP through a TestClient. Separate
executable tests exercise cancellation, restart, continuation and reversal, but
do not establish their full composition from a cancelled partial edit through
an actual configured check and reverse-order restoration. Use the existing
[owned loopback fixture](tests/cli_process_support.py), installed CLI and shared
application runtime; do not add a new runtime or model supervisor.

## Required sequence and observations

1. Create a temporary Git workspace with an existing user line, a mode640 target,
   unrelated work and a configured pytest check. Stage initial user files and
   retain the index bytes. Script one partial file-tool edit, then hold the next
   provider request.
2. Submit with the executable CLI and checkpoints enabled. After observing the
   edit and pending request, explicitly cancel through another CLI process.
   Observe the saved terminal cancellation, `not_run` verification, unknown total
   usage and both checkpoint references. A cancellation acknowledgment is not
   settlement. Own and clean up every subprocess.
3. Stop the API, restart against the same stores and inspect the source with
   executable workspace/transcript/configuration review. Review must make no
   provider request and must not change source evidence or user work.
4. Explicitly continue the selected source turn with a stable request key and
   uncaptured-state acknowledgment. Script a final edit, an actual configured
   pytest call, then a finish response. Require saved completed/passed status,
   exit0 check evidence, the expected artifact and a separate fork. Keep the
   cancelled source archive/run unchanged. This passing fixture check is not an
   independent model-quality score.
5. Restart without provider replies. Reverse the fork's selected file first,
   then the source's, with distinct stable keys. Observe each applied journal and
   intermediate bytes, restoring only that file. Preserve later unrelated work,
   original target mode, configured check bytes and the staged Git index.
6. Restart again and look up saved journals, runs and source archive. Require
   exact record equality and zero provider calls. Single-file reversal is not
   atomic multi-file restore or process isolation.

## Adverse sequence: review is not a lease

After step3's successful review, change the target's bytes without changing
settings or source records. Submit the same explicit continuation. Require a
saved blocked result before checkpoint binding or provider dispatch. Explicit
source reversal must reject the changed target too. Another restart and keyed
lookup/replay must retain those outcomes and preserve later user bytes/mode,
unrelated work and the original index. No automatic retry or manual patch rescue
is allowed. Scope excludes concurrent writers during the mutation itself.

## Validation and reporting

Implement and run the successful sequence first, then add the adverse sequence
as a separate reviewable slice. Record focused subprocess results and normal
full offline/static gates. If a case fails, retain the failing observation and
fix only the demonstrated cause. Do not infer that a new test proves a runtime
bug existed. No runtime defect is asserted by this specification.

The historical [live CLI](evals/projects/duodoro/RECOVERY_RESULTS.md) and
[live panel](evals/projects/duodoro/PANEL_RECOVERY_RESULTS.md) results retain their
original limitations. Current rendered-panel and actual-model acceptance remain
separate. G2 still needs agreed candidate-specific workflow acceptance; G3's
8/10 original target, G4's repeated/held-out/two-model measurements and G5 remain
open. This sequence authorizes no generation or release.

## Successful executable sequence — observed 2026-11-19

The [process test](tests/test_cli_process_recovery_sequence.py) now exercises
steps1–6 with actual CLI/API subprocesses and the configured pytest command.
Its focused run passed one case in 9.25s: cancelled source with unknown totals,
completed/passed fork, two applied reversals and exact saved records after restart.
Provider calls are two for the cancelled source, three for the fork and zero
for review/reversal/final inspection. Target bytes/mode, later unrelated work,
visible-check bytes and staged index are preserved. No runtime change was needed.
The adverse stale-review sequence is recorded separately below. PRs #454–#456
merged after required CI. The successful sequence local gate passed 4,030 tests
at its tested revision; no model-quality or G2 closure follows.

## Stale-review sequence — observed 2026-11-19

The same executable test now has a separate adverse branch. After unchanged
workspace, valid transcript and matching configuration reviews, a user byte edit
makes continuation block before checkpoint binding or provider dispatch.
Current workspace evidence may be captured during validation.
Source reversal also rejects. After another restart, the continuation key returns the
same blocked record and reversal lookup/replay returns the same rejected journal;
later user bytes and a newly chosen mode600 survive with unrelated work and Git
index unchanged. Source records remain exact and no extra run is created.
Both successful/adverse focused cases passed in 17.59s. Ruff/format/mypy and diff
checks passed; the local gate passed 4,031 tests (five live deselected), and both
required checks passed on `0581ce0` before merge `df961b5`. These passing guards
required no runtime fix and establish no actual-model quality or G2 closure.

## Failed configured check — observed 2026-11-19

The executable sequence now has a third case: its fork writes value2 against the
unchanged value1 pytest criterion, executes the actual check, then attempts to
finish at the selected three-iteration cap. Saved completion is `incomplete`,
verification is `failed`, check exit is1 and the reason records the unverified
edit. The canned answer's claim of a passed check does not override those fields.
After restart, JSON equality and human `incomplete`/`failed` review retain the
failure, while fork/source reversal restores the actual files and preserves user
mode, source evidence, unrelated work and staged index. Source and fork use the
same settings within each case. All three focused cases passed in 26.10s; static
gates pass. Full offline/CI success remains required before merge. No runtime
fix, inference or model-quality/G2 closure follows. The stale-check case remains
separate future work.
