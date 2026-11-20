# Client and recovery acceptance evidence

Offline review at `384b33c`, 2026-11-19. This maps G2 in the
[release checklist](RELEASE_CHECKLIST.md); it does not replace the roadmap's
criteria, close G2 or authorize inference. Evidence below is inspected source
and retained reports, not a fresh execution of every test or historical browser
flow on this revision.

## Evidence map

| Workflow requirement | Existing evidence | What remains unproven here |
|---|---|---|
| CLI submission, saved review and restart | [Executable/API restart test](tests/test_cli_process_restart.py), [guarded continuation](tests/test_cli_process_continuation.py) | Scripted responses exercise real processes; they do not establish coding quality or arbitrary host-crash durability. |
| Explicit cancellation and owned cleanup | [Actual SIGINT](tests/test_cli_process_sigint.py), [serve shutdown](tests/test_cli_process_shutdown.py), [command descendants](tests/test_cli_process_command_cancel.py) | Owned process-group cases do not prove termination of arbitrary detached children, remote inference or billing. |
| Disconnect and timeout recovery | [SIGTERM/disconnect](tests/test_cli_process_disconnect.py), [keyed timeout recovery](tests/test_cli_process_timeout.py) | Scripted real-process cases cover selected settlement/recovery paths, not every transport failure. |
| Workspace admission and crashed API | [Competing CLI roots](tests/test_cli_process_contention.py), [SIGKILL recovery](tests/test_cli_process_crash.py) | Admission is within one API process. SIGKILL evidence is not power-loss durability; interrupted evidence is not a completed run. |
| Guarded CLI continuation | [Restart, content drift and keyed replay](tests/test_cli_process_continuation.py), [API guard failures](tests/api/test_continuation.py) | Permission-only refusal is now exercised by the offline follow-up below; broader candidate/model acceptance remains open. |
| Guarded CLI reversal | [Restart, stale bytes, reverse order and journal replay](tests/test_cli_process_reversal.py), [API failure contracts](tests/api/test_file_revert.py) | Permission-only stale targets are now covered by the offline follow-up below. Single-file restoration is not atomic multi-file restore. |
| Panel recovery controls | [Actual scripts with a simulated DOM and API-derived records](tests/test_panel_guarded_recovery.py), [draft controls](tests/test_panel_recovery_draft.py) | Node/VM execution is not a rendered browser, layout/accessibility check or full browser-to-server network proof. |
| Actual-model CLI/panel recovery | Historical [CLI sequence](evals/projects/duodoro/RECOVERY_RESULTS.md) and [rendered-panel sequence](evals/projects/duodoro/PANEL_RECOVERY_RESULTS.md) | Selected one-task observations at their original runtimes are not a current repeated workflow acceptance run. They retain unknown usage and other documented limits. |

The scripted process fixtures run the installed CLI, a real loopback API and
shared runtime against temporary workspaces/stores. Only their provider responses
are canned. Keep this distinct from both in-process API tests and actual-model
coding evidence. Neither successful scripted responses nor a configured check
substitutes for independent acceptance of a real coding artifact.

## Completed offline acceptance sequence

The [bounded executable sequence](OFFLINE_RECOVERY_ACCEPTANCE.md) specifies a
cancelled partial source edit, restart/review, guarded continuation with a real
configured check, reverse-order restoration and durable read-only inspection.
A separate adverse case changes user bytes after successful review to demonstrate
that review is not a lease. The successful full composition now has an
[executable process case](tests/test_cli_process_recovery_sequence.py), including
stale-review refusal and durable blocked/rejected replay. The earlier combined CLI test uses
an in-process HTTP bridge. Keep scripted mechanics separate from actual-model/
browser acceptance.

## Offline follow-up — 2026-11-19

The two executable test modules now each include no drift, content drift and
mode-only drift. All six focused cases passed. The mode-only cases keep bytes
identical before submission, then assert the saved blocked/rejected outcome and
the user's mode. Continuation makes zero provider calls and binds no new before
checkpoint; reversal preserves the staged index and source records, then replays
the rejection after another restart without overwriting later user bytes/mode.
No runtime defect was found and no application behavior changed. Ruff, formatting
and mypy passed. PRs #451–#453 merged after required CI; the final local gate
passed 4,029 tests at the original tested revision.
These are scripted-provider, real-process observations, not model-quality or
full G2 acceptance evidence.

## Remaining release work

G2 still needs an agreed candidate-specific CLI/panel acceptance record covering
the required sequence, interventions, failures and preservation of pre-existing
work, reconciling historical and current evidence rather than simply summing PRs.
Fresh actual-model workflow execution needs separate approval. G3's original
8/10 target, G4's repeated/held-out/two-model measurements and G5's distribution
and Jorge's release decision remain open. The fresh twelve-cell development
comparison remains unadmitted under the offline selection.

## Offline verification cases

The composed sequence now tests passed/failed configured checks, a blocked
stale-workspace submission and a post-check edit causing stale verification.
All four focused cases pass. Keep independent acceptance and actual-model
evidence separate.

- **Failed check:** script an incorrect fork artifact, run the configured pytest
  command, then attempt to finish. Bound the iteration allowance so finish returns
  a saved `incomplete` result with verification `failed` and a nonzero check exit.
  This executable case now passes. The answer text must not substitute for the
  structured outcome. Restart review
  must retain it; guarded reversal must still preserve source and user state.
- **Stale check:** script a correct artifact and passing configured check, then
  write a different artifact and attempt to finish without rechecking. Require
  `incomplete`/`stale` while retaining the earlier exit0 check as historical evidence.
  This executable case now passes. Restart must not promote the run to
  completed/passed. Reverse the actual final
  artifact, not an imagined checked artifact.

Use scripted providers, the existing real CLI/API fixture and disposable stores.
Keep source/fork settings identical within each sequence. Preserve configured
check bytes, source records, file mode, unrelated work and staged Git index.
These are missing executable composition cases, not asserted runtime defects.
No generation, paid call, browser run or release approval follows.

## Panel submission over HTTP — 2026-11-20

The [HTTP contract](tests/test_panel_http_recovery.py) now runs actual app,
recovery and submission JavaScript with native Node fetch against an owned
loopback API process. One explicit submit POST is bound to the already-retained
body; after API restart a separate script process restores the same local intent,
sends nothing at load and makes one GET only on explicit Check. The actual
configured pytest completes in the shared runtime and the saved identity is
validated by the production client code. Source records, file/check bytes, mode,
unrelated user work and staged index remain preserved.

The source and reviewed draft are seeded in process; selection is injected at the
existing reviewed-candidate seam. DOM and storage are simulated, so this is not
rendered-browser, real browser-storage, full draft-control or browser network
policy acceptance. Three focused HTTP/existing panel contracts pass with static
gates; full offline/CI remains required before merge. Separate lost-response and
cancellation cases are recorded below. No target inference,
model-quality result or G2–G5 closure follows.

## Panel response delivery loss — 2026-11-20

The HTTP fixture now has confirmed and lost-response modes. Lost delivery is
injected after native fetch receives the server acknowledgment: the wrapper
consumes that response and throws before the client sees it. This is not a real
socket-disconnect or browser-network test. The production script keeps its exact
intent, reports an unconfirmed outcome and sends no automatic second request.
After API/script restart an explicit Retry posts the identical retained body,
recovers the existing completed run and leaves exactly source+fork records with
zero provider calls. A further explicit Check makes one GET and preserves intent,
source/user state and Git index. Four focused HTTP/existing panel contracts and
static checks pass; full offline/CI remains required before merge. Cancellation
is recorded separately below; browser/G2/model-quality limits remain.

## Panel cancellation over HTTP — 2026-11-20

The third HTTP case holds the recovery provider after a real partial file edit.
A new script process restores intent without sending a request, explicitly checks
the running fork, then posts cancellation to that validated fork's session. The
production UI reports acknowledgment with settlement still uncertain and refuses
a repeated cancel click. The test separately observes saved cancelled/not-run
state, no check attempts and unknown aggregate token usage. Two mock provider
requests are recorded; there is no generation backend.

After API/script restart, one explicit Check GET validates and displays the saved
cancelled outcome with zero provider calls. Retained intent, source records,
partial target bytes/mode, later unrelated user edits, configured test bytes and
staged index remain preserved. Five focused HTTP/existing panel contracts and
static gates pass; combined local offline/CI success remains required before
merge. Source/draft seeding and DOM/storage remain simulated. This closes these
bounded HTTP cases only, not rendered-browser, model quality or G2–G5 acceptance.


## Panel draft and approval over HTTP

The [panel HTTP tests](tests/test_panel_http_recovery.py) now drive source selection,
explicit Load, proposal and saved-span entry, Preview and Submit through the
production scripts. Submission carries the hashes of the draft returned by the
loopback API. Editing an approved proposal clears its draft and acknowledgements.
An attempted submit sends nothing until a fresh preview and new acknowledgements
enable submission for the revised message.

The fixture seeds its source in process and runs the panel scripts with simulated
DOM and storage. Recovery uses a scripted provider. Seven focused tests passed.
Rendered browser and candidate-specific release acceptance remain open.
