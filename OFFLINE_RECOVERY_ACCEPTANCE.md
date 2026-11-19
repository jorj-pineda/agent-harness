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
saved blocked result before checkpoint capture or provider dispatch. Explicit
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
