# Paired recovery client preflight

**Offline scripted contracts, 2026-10-27.** This is PR 3 of the
[five-PR local protocol](PROJECTED_RECOVERY_PROTOCOL.md), after the
[owned staging primitives](RECOVERY_STAGING.md). It constructs no inference
provider. The fixed T02/T04 plan and every live source/arm row remain unrun.

The new portable fixture starts the real application in owned local API child
processes with an explicitly substituted scripted provider. Actual Typer CLI
commands send HTTP over loopback, using the plan's shared settings, half-budgets,
check argv and exact followup proposal. Prepared fixture code, acceptance and
references are independent of the user's Duodoro checkout. `gemma4:12b` is a
configured profile identity here, not an observed backend: the responses and usage
are scripted. No Ollama, hosted provider, model download or support resource starts.
Normalized first inputs are observed; SDK/wire and model/environment validation
remain for the separate exact execution freeze.

## Process and preservation boundary

`recovery_lifecycle.OwnedAPIs` is an append-only set of explicitly registered owned
API children and loopback ports. Register each live child created with
`start_new_session=True` before using it. Every settlement check requires all
registered children to exit zero, no surviving member of their process groups,
and successful probes of their loopback ports. A new active arm invalidates an
earlier source settlement check; callers pass the same aggregate to staging,
scoring and closed-store cloning. No process is stopped by this checker, and a
port probe sends no HTTP request or data to an unknown service.

The fixture uses normal application lifespan shutdown, then records closed SQLite
connections, no unfinished application tasks and its scripted first inputs. The
parent stops only its exact owned child and verifies the group/port gate before
grading or replacing files. A timeout forces cleanup of that known owned group,
fails the preflight and grants no staging permission. Test logs/configuration/
selection/approval/trace files are private. HTTP readiness verifies the fixture's
owned PID so another listener cannot masquerade as the intended server.

This checker supports POSIX children and IPv4 loopback only. It cannot discover
unregistered processes or arbitrary workers detached into other groups, prove
ignored-content equality, exclude external writers, reserve a port, or create an
execution lease. Clean application shutdown must still settle its owned tools.
An actual managed pytest cancellation test waits for the terminal turn and confirms
that its separately grouped command PID is gone. `/cancel` acknowledges a request;
its acknowledgment alone is not that cleanup receipt. The reviewed live launcher
must provide a normal exit-zero shutdown path; a signal return code alone is not
accepted as proof of clean shutdown. Actual backend lifecycle remains a fresh gate.

## Observed contracts

| Contract | Offline evidence |
|---|---|
| Same saved source for C and P | Actual CLI source writes a partial file and workspace fact; closed session/fact images clone independently with original identities/checkpoint blobs. Both arm orders run at the same root/inode. |
| Preserved source and sibling | Source run/archive JSON and original/parent store byte hashes stay unchanged. Each settled artifact gets independent grade/regrade before sibling staging. |
| Native C versus projected P | C receives saved tool history; P's actual first messages, schemas and allowance match the exact reviewed draft pins. Both retain the same final proposal and refreshed workspace fact. |
| Different context admission | A real large tool reply makes native C record a blocked outcome before any provider call. The reduced P input fits and executes its scripted patch/check. The blocked C row is retained. |
| Failed checks and missing usage | Failure remains incomplete/failed; unknown output usage prevents the scripted file write and stays unknown. Artifact grading is separate from workflow/check status. |
| Cancellation | In-flight scripted inference preserves unknown usage; separately, cancellation of a real managed pytest waits for terminal cleanup and reaping before proceeding. |
| Provider or acknowledgment failure | Provider failure has no invented final. A deliberately lost POST acknowledgment creates exactly one keyed run; observation uses original-key GET only, with no resubmission or replacement key. |
| Restart and later work | Read-only CLI lookup after a new API process returns the saved record with zero scripted calls. Later workspace edits remain and refuse staging against the earlier scored artifact. |
| Cleanup refusal | Active children, surviving group members, abnormal exits, occupied ports and a later registered API all refuse settlement. No unknown listener is closed or signaled. |

The focused gate also reruns the existing shared admission and API/client contracts:

- `tests/test_recovery_admission.py`: source/record/archive/checkpoint/current-root/
  mode/configuration/context/newer-work gates and exact first-input refusal.
- `tests/api/test_recovery_execution.py`: source/workspace/fact/configuration drift
  after seed, pre-provider first-input drift, unknown usage, cancellation and failed
  execution/restart evidence with no redispatch.
- `tests/api/test_continuation.py` and `test_turn_admission.py`: full-history guards,
  related work, busy roots, invalid transcript/configuration and atomic storage refusal.
- `tests/test_cli_guarded_recovery.py` and `test_cli_recovery_workflow.py`: bounded
  approval/acknowledgment checks, durable identity validation, ambiguous transport,
  restart, actual checks and later-edit preservation.
- `tests/evals/test_recovery_staging.py`: artifact/criterion/reference tampering,
  unknown work, symlinks, special files, modes, directory changes, capture bounds,
  store/WAL/sidecar/corruption refusal and prior-task protection.

These are intentionally constructed development fixtures, not model-quality or
end-user task results. Scripted completion proves no measured recovery gain, token
savings, instruction compliance, backend capability, live budget compliance or
release readiness. Existing historical cohorts, defaults, no-go/8-of-10 and broad
G2–G5 decisions remain unchanged. Next: independently reviewed exact execution
freeze, parent approval receipt and fresh per-phase gates; only then conditional
authorized local Gemma observations/results, without substitutes if a gate fails.
