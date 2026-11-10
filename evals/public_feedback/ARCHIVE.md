# Read-only feedback capture verification

`evals.public_feedback_archive.verify_feedback_capture` checks a selected rehearsal
against an explicitly supplied index SHA-256. It constructs no provider/backend,
allocates no output and changes no capture bytes. This is the prerequisite for
preserving the consumed capture and staging fresh candidate fixtures; those actions
are not implemented by this verifier.

The checker validates the strict eight-cell index and fixed ordinal/arm/repeat
order, exact known file maps, private file/container modes, root/native/container
inventories and selected hashes. Unknown path-map entries refuse before extraction.
Captured receipts must match their selected cells; initial/final visible tests must
match the selected arm. The two archived final source files must match their private
captured contents. Workspace caches and other configured ignored paths are outside
that final-source content comparison; it is not a full filesystem or metadata audit.

Normalized first requests are checked for their own identity, original task,
historical workspace path and selected tool namespace. Their production-native
serialization must agree with normalized messages/tools/options and captured
Settings. Runtime configuration, command environment and preservation records must
agree with that selection. The aggregate and individual results must match and retain
scripted completed/accepted states; complete native response evidence and reported
token sums must agree. These checks do not turn canned usage into real inference.

Settings/environment and executable/dependency inventory files are verified as
captured bytes and schemas. **Historical executable/dependency paths are not
reobserved.** Current original public task/plan selection is still checked. Live
preflight will need separate current host/resource observations; archival verification
is not a lease, an authenticated approval, or proof that a model produced the files.

If checking a relocated archive, supply its original absolute capture root using
`original_directory`. The old root need not contain this capture now; the first
request and cell metadata must still name exactly the recorded paths. No path
rewriting, input rebinding, admission or retry occurs.

## Validation

Twenty-six focused regressions pass, covering read-only behavior with provider and
real transport construction forbidden, missing/changed/private files, exact path
maps, wrong selected hashes, cell order, historical roots, task/native identity,
configuration/receipt/test changes, report/usage disagreement and final-source
mutation. Rebinding a test index hash cannot bypass its semantic checks. A separate
test changes a historical dependency and confirms archival checking does not read it.

The retained [native mock rehearsal](mock-rehearsal-20261110.json) also passes
verification at its original selected index hash: eight cells and 160 selected
files. This was a read-only check, not a new model attempt or artifact regrade.
No original cohort, independent acceptance criterion or application default changes.

Next: preserve captures and stage fresh exact-path fixtures, then bind an explicit
execution manifest/preflight/single-use admission. Owned execution, retained
preparation/full owned rehearsal, concrete reviewed freeze, actual approval and
fresh launch gates remain before inference. Release no-go and G2–G5 remain unchanged.
