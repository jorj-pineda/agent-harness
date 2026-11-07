# Named public-pilot workspaces

`prepare_public_cell(ordinal, directory)` creates an exclusive, named copy for
one of the six reviewed public cells. It validates the pinned plan and all task
inputs before allocation, requires a canonical location outside Git workspaces
and source inputs, checks copied fixture bytes/modes, rechecks the selection,
and writes a private receipt last. An occupied destination refuses. Partial
failures retain their directory for inspection; nothing is reset or deleted.

`verify_public_cell` checks the selected ordinal, plan/task/repeat receipt,
container inventory/private mode, and original fixture contents and modes.
Unknown files, changed bytes/modes, symlink entries and receipt mismatches refuse.
`admit_public_cell` verifies before taking the existing workspace baseline and
creating an exclusive admission marker, then rechecks source and fixture before
yielding the existing DisposableWorkspace type. Failure after the marker consumes
admission. Success, caller error and partial edits all remain on disk; there is
no retry, resume, automatic revert or cleanup of submitted artifacts.

Preparation/admission helpers create no provider or service and issue no parent
approval. Shared run_task can forward real provider requests when a trusted caller
selects the public context and supplies a real provider; these workspace guards
do not authenticate that caller or receipt, or establish launch authority. The caller still owns authority, resource lifecycle and
ongoing source/criteria/request guards. The public context is an explicit opt-in path; the existing prepared_directory
path remains registered-held-out only, and ordinary disposable execution is unchanged. The public receipt is a
distinct format and is not a held-out seal or a launch receipt.

Checks are bounded, non-atomic observations, not locks or process isolation.
Filesystem changes by outside writers can race checks. Snapshot limits and
exclusions retain their documented scope; original fixture equality before
admission does not certify arbitrary later command behavior. These scripted
contracts establish preservation/admission mechanics, not model quality or
completion of the pilot's freeze, approval and fresh-launch requirements.

## Explicit shared-evaluator admission

`real_run.run_task(..., public_cell=PublicCellWorkspace(...))` now selects this
context explicitly. The ordinary disposable path and the existing held-out-only
prepared_directory path retain their behavior. Supplying both refuses. Before
admission, the public guard requires the canonical selected Task, harness mode,
cell repeat and the pinned policy's budget/settings values with direct/whole-file
fixed-check selection. Unknown or modified task identities refuse before a marker
or provider request. There is no public live CLI or approval shortcut.

The guard checks original source/criteria/reference pins and selected policy before
each provider request and around independent grading. An observed failure latches
closed even if state is restored. Public drift records an explicit failed result;
grading is skipped when the pre-grade check fails. A post-grade failure preserves
an available grader exit/output but refuses to certify acceptance. Result metadata
records the public plan/cell/checkpoints and failed boundary, not a held-out seal.
Legacy rows gain no public evidence field.

These are point checks, not continuous monitoring or process isolation. The guard
compares the selected policy fields, not a complete host/provider/Settings freeze.
Actual launch still needs captured complete settings, runtime/resource guards,
owned provider composition, exact first inputs, rehearsal and the separate reviewed
freeze/parent receipt/fresh gates. Trusted callers retain launch authority and
provider lifetime responsibility; merely constructing a PublicCellWorkspace is
not permission to generate or execute unfamiliar code.
