# Copied fresh-artifact acceptance regrades

`regrade_fresh_run` selects private manifest/report bytes by independently supplied
hashes and validates the fixed twelve-cell ledger/result binding. Each recorded
result needs its independently selected artifact snapshot digest. All three task
criteria are hashed separately; every recorded row must identify its own unchanged
original acceptance policy and criterion before allocation. Weak/stronger visible
tests do not redefine independent acceptance.

The existing interpreter entrypoint, selected runtime/dependency files and command
environment are reobserved before and around grading. Fresh task selection and
original criteria remain pinned. These observations and hashes do not authenticate
writers, a complete dependency closure or actual execution authority. Grader
revision is recorded separately from the original execution runtime.

Every present result grades only a bounded copied artifact (100 nonignored files,
1 MB), preserving source bytes/modes and the original report. Copies exclude the
existing cache/environment patterns, reject symlinks and verify correspondence
before and after copying. Independent acceptance runs on the copy through the shared
grader. Imported trusted code still has host permissions: this is not process
isolation. The caller owns authentic selection of artifact snapshot pins.

Original acceptance, completion and ledger state stay distinct from copied grades.
Started results can be graded without certifying settlement; missing slots remain
`no_result` with null outcomes. Mismatches are retained rather than relabeling the
original report. Failure or cancellation keeps earlier per-cell evidence and a
failure marker, never a success summary, reset or retry. Full output binds all
three criterion hashes, source report label and per-cell copy/check results.

This API launches no provider and issues no approval. Scripted regrade controls
measure mechanics only; benefit stays `not_assessed`, with no recovery, held-out,
default or release conclusion. A concrete frozen candidate, passing candidate
gates, actual approval and fresh launch checks still precede real comparison.


## Retained scripted exercise

Clean grader `2095c6b570c7e18e3165d7d7a571a65a593e35f6` regraded twelve retained
scripted artifacts from the earlier admitted mock bundle. Independently selected
snapshots first matched each original reference and selected visible bytes/modes.
All twelve copied grades exited zero and matched recorded exits, with the original
report/source snapshots preserved. The [proof](copied-regrade-proof-20261116.json)
binds manifest, report, recipe, artifact snapshots, three distinct criteria and
copied summary bytes. Execution runtime and grader revision remain separate.
These are reference artifacts, not measured model improvement or actual approval.
Full offline validation and CI remain required before merge.
