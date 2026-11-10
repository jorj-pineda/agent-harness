# Preserve mock inputs and stage fresh exact-path fixtures

`evals.public_feedback_staging.stage_feedback_capture` accepts a selected mock
capture and index hash, verifies it [read-only](ARCHIVE.md), and reserves a distinct
empty archive directory. Both paths must be canonical evaluation-owned locations
outside Git/source inputs and must not overlap. Occupied archive paths refuse.

The entire consumed capture directory is renamed to `archive/captured` without
rewriting its files or index. This preserves selected evidence, final mock artifacts
and unselected caches. The archived bytes are checked again using the original
capture root as their historical path; historical executable/dependency files
are not reobserved by this operation.

The original root is then recreated with eight fresh selected fixtures at the
exact absolute paths named in the first inputs. Implementation and visible-test
bytes/modes are verified for each arm, and no cell is admitted. The private
`staged.json` receipt is written exclusively and last, after rechecking the archive
and fresh cells. It binds the plan, selected index and archive path and explicitly
retains `inference_authorized: false`.

`verify_staged_feedback` checks that receipt, private root/receipt modes, exact
inventories, the preserved capture and all eight fresh unadmitted cells. A consumed
cell, edited fixture, changed archive or extra container entry refuses. This is a
freshness preflight; it is not an ongoing execution guard for already admitted cells.

Failures preserve the archive and any partial staging without rollback, deletion
or automatic reset. A partially staged root cannot be reused as a capture; retrying
there refuses. Rename across unsupported filesystems can fail with partial archive
reservation retained. The sequence is bounded and non-atomic: it does not lock out
external writers, provide process isolation, authenticate approval or grant a lease.

## Direct proof and regressions

Eleven focused regressions pass: complete file/mode preservation and exact fresh
paths, no replay into a different archive, pre-rename overlap/occupied/drift refusal,
post-rename preparation failure retention, and staged receipt/mode/admission/test/
archive/inventory drift. Tests use temporary synthetic captures only; their fixture
restoration is not a product reset mechanism or a retry of a model cohort.

A separate clean-source run at `a46587523ffc0934f3cddfcc7468f80dbecee1df`
prepared a **new** mandatory-mock capture, then staged it. The earlier retained
rehearsal was untouched. All **218 files and their modes**, including caches, matched
the archive afterward; eight exact-path fixtures verified fresh and unadmitted.
The [proof summary](staging-proof-20261110.json) binds the preserved index and
staging receipt. The recipe and private artifacts remain at:

`/Users/jorge/.local/share/agent-harness/trials/public-feedback-mock-i6ghsgt5`

Preparation used 32 canned mock requests; staging made zero provider requests.
Real HTTP transport construction was forbidden by the proof driver. This is
mechanics evidence, not model performance, an approved live candidate or an owned
backend rehearsal. Dependency selection remains the earlier bounded source-file
selection, not complete import/shared-library closure.

## Remaining execution boundary

Use a separately prepared input capture for each proposed candidate; preserving
an old rehearsal does not authorize its promotion or reopen a closed cohort.
The next slice is an explicit execution manifest, read-only live preflight and
single-use trusted admission. Owned serial execution, retained preparation/full
owned rehearsal, concrete reviewed freeze, actual approval and fresh launch gates
remain before inference. Original criteria/defaults, release no-go and G2–G5 are
unchanged.
