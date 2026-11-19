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

## Next offline acceptance sequence

The [bounded executable sequence](OFFLINE_RECOVERY_ACCEPTANCE.md) specifies a
cancelled partial source edit, restart/review, guarded continuation with a real
configured check, reverse-order restoration and durable read-only inspection.
A separate adverse case changes user bytes after successful review to demonstrate
that review is not a lease. Existing individual process cases do not establish
this full composition; the combined CLI test currently uses an in-process HTTP
bridge. Keep scripted mechanics separate from actual-model/browser acceptance.

## Offline follow-up — 2026-11-19

The two executable test modules now each include no drift, content drift and
mode-only drift. All six focused cases passed. The mode-only cases keep bytes
identical before submission, then assert the saved blocked/rejected outcome and
the user's mode. Continuation makes zero provider calls and captures no new before
checkpoint; reversal preserves the staged index and source records, then replays
the rejection after another restart without overwriting later user bytes/mode.
No runtime defect was found and no application behavior changed. Ruff, formatting
and mypy passed; full offline/PR CI completion remains required before merge.
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
