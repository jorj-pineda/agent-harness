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
| Guarded CLI continuation | [Restart, content drift and keyed replay](tests/test_cli_process_continuation.py), [API guard failures](tests/api/test_continuation.py) | Existing executable cases do not change only a captured file's permissions between source and continuation. |
| Guarded CLI reversal | [Restart, stale bytes, reverse order and journal replay](tests/test_cli_process_reversal.py), [API failure contracts](tests/api/test_file_revert.py) | Existing executable cases do not exercise permission-only stale targets. Single-file restoration is not atomic multi-file restore. |
| Panel recovery controls | [Actual scripts with a simulated DOM and API-derived records](tests/test_panel_guarded_recovery.py), [draft controls](tests/test_panel_recovery_draft.py) | Node/VM execution is not a rendered browser, layout/accessibility check or full browser-to-server network proof. |
| Actual-model CLI/panel recovery | Historical [CLI sequence](evals/projects/duodoro/RECOVERY_RESULTS.md) and [rendered-panel sequence](evals/projects/duodoro/PANEL_RECOVERY_RESULTS.md) | Selected one-task observations at their original runtimes are not a current repeated workflow acceptance run. They retain unknown usage and other documented limits. |

The scripted process fixtures run the installed CLI, a real loopback API and
shared runtime against temporary workspaces/stores. Only their provider responses
are canned. Keep this distinct from both in-process API tests and actual-model
coding evidence. Neither successful scripted responses nor a configured check
substitutes for independent acceptance of a real coding artifact.

## Next offline acceptance case

Prioritize **permission-only stale-state refusal after restart** for executable
CLI continuation and reversal. The concrete failure to guard against is restoring
an old mode, or dispatching a continuation, after the user changes a file's mode
without changing its bytes. Current guards explicitly compare captured modes;
existing process tests exercise byte drift and unchanged modes instead.

Extend those existing process cases rather than add another provider or workflow:

1. Record a scripted source edit and its checkpoints; stop the owned API.
2. Change only the selected file's mode, retaining identical bytes.
3. Restart over the same stores and submit the explicit continuation or reversal.
4. Require the saved continuation to be blocked before provider dispatch and
   checkpoint capture, or the reversal journal to be rejected. A successful CLI
   exit/HTTP acknowledgment alone is insufficient.
5. Preserve user bytes/mode, unrelated work and source records; for reversal,
   preserve the already-staged Git index and prove journal replay after another
   restart cannot mutate subsequent user work.

This is a bounded preservation gap in executable evidence, not an observed runtime
bug. Fix runtime behavior only if the case demonstrates a defect. A pass closes
these cases only. Scope excludes cross-process admission, detached processes,
atomic multi-file restore, model comparisons and new browser acceptance.

## Remaining release work

G2 still needs an agreed candidate-specific CLI/panel acceptance record covering
the required sequence, interventions, failures and preservation of pre-existing
work, reconciling historical and current evidence rather than simply summing PRs.
Fresh actual-model workflow execution needs separate approval. G3's original
8/10 target, G4's repeated/held-out/two-model measurements and G5's distribution
and Jorge's release decision remain open. The fresh twelve-cell development
comparison remains unadmitted under the offline selection.
