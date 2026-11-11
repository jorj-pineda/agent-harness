# Pinned copied-artifact feedback acceptance regrades

`regrade_feedback_run` is an internal offline API for trusted retained evaluation
artifacts. It requires selected manifest/report hashes, independently selected
artifact snapshot hashes for exactly the recorded result ordinals, and a separate
new output directory. It makes no model/provider/backend request and issues no
approval. Existing run reports and inference paths are unchanged.

The trusted caller must select expected artifact hashes from independent retained
evidence, not derive them from unchecked current files and call that provenance.
Pins bind scoped bytes/modes, not authors, authority or historic finality. The API
checks fixed ledger/result correspondence and preserved cell ordinals. It requires
matching original external criterion/grader bytes, the exact selected existing
Python entrypoint, selected runtime/dependency observations and the retained
command environment. Canonical evaluation-owned paths exclude Git working trees;
output must be separate from all managed input trees.

For each result, copying is bounded to 100 nonignored files and 1 MB. Symlinks,
special files, drift and over-budget artifacts refuse before grading. Copies omit
caches and environment files. File hashes and modes must match the selected pin
and source/copy checkpoints. Acceptance uses the shared isolated-Python bootstrap
and original external tests, with the existing 30-second subprocess/output bounds
and cancellation management. Editable visible tests never redefine acceptance.
These are not hard end-to-end limits for the entire operation.

Imported artifact code still has host permissions. Only trusted artifacts belong
here; disposable copies and path checks are not process isolation. Source snapshot
equality records bounded observations, not a continuous writer-attribution lease.
Ignored files are outside that observation; this does not certify the whole host
or establish complete import/shared-library closure.

Original passed/completion/acceptance-exit fields remain alongside each regrade.
A started cell with a copied grading result still has unknown settlement; no-result
slots stay unknown and are not failed attempts. Regrade disagreement does not
rewrite the original report. Benefit assessment remains `not_assessed`: independent
regrades alone do not establish interventions, regressions, completion or quality.

Each available cell record is saved before advancing. A final summary is written
only after all selected work/checks succeed. Failure or cancellation keeps partial
records and a failure marker, retaining the primary exception if recording also
fails. Occupied outputs refuse; there is no reset, retry, resume or model resubmission.

## Evidence

Sixteen focused tests cover pinned copies, started/unrun reporting, weakened visible
tests with independent rejection, hash/ordinal/criterion/environment/interpreter/
path refusals, symlink and size/file-count bounds, retained external writes,
cancellation and partial earlier records after a later failure. Synthetic host
inventory seams are explicit in tests; real grading runs on temporary copies.

A separate check used all eight scripted artifacts from the retained rehearsal.
Expected hashes came from the previously selected evidence index. All copied
grades passed, source/report checkpoint checks matched, and original records stayed
unchanged. The [proof](copied-regrade-proof-20261111.json) binds the source manifest,
report, selected artifact hashes, grader revision and new summary. Private results
remain in the named output directory. An initial invocation through a different
Python entrypoint correctly refused before output allocation; the successful run
used the exact selected absolute interpreter.

These are scripted-reference mechanics, not new model attempts or measured gains.
The frozen live candidate remains unadmitted. Actual approval/fresh launch checks,
real comparison evidence and intervention/regression review remain before any
promising-public-follow-up statement. Original targets and release no-go remain.
