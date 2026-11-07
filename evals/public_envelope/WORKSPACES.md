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

This is evaluation-owned workspace machinery, not a live execution interface.
It does not issue parent approval, construct providers/services or authorize
repository execution. The caller still owns authority, resource lifecycle and
ongoing source/criteria/request guards. `real_run.run_task` is not wired to this
helper yet: its existing prepared_directory path remains registered-held-out
only, and ordinary disposable execution is unchanged. The public receipt is a
distinct format and is not a held-out seal or a launch receipt.

Checks are bounded, non-atomic observations, not locks or process isolation.
Filesystem changes by outside writers can race checks. Snapshot limits and
exclusions retain their documented scope; original fixture equality before
admission does not certify arbitrary later command behavior. These scripted
contracts establish preservation/admission mechanics, not model quality or
completion of the pilot's freeze, approval and fresh-launch requirements.
