# Bounded live CLI recovery probe — execution freeze

**Prepared, not run.** This is task 2 of Jorge's authorized three-task batch.
Review and merge this protocol and [manifest](recovery-freeze.json) before task 3
executes the one scheduled workflow. No inference ran during preparation. The
frozen runtime is `dacf072c0a09546fdc69d2407091ab049ee1b520` (PR #165 merge).
The manifest SHA-256 is
`95bed2f20e47cc5b518ffa0857db55a12d43f172a639fd0e68af92a1574aa620`.

## Purpose and request boundary

The original trial cancelled before an edit; its planned reversal had no eligible
patch/checkpoint pair. The later whole-file duration observations produced two
completed accepted artifacts, but exercised neither partial-edit recovery nor
reversal. The new [combined regression](../../../tests/test_cli_recovery_workflow.py)
is scripted evidence. This probe asks whether the actual CLI and local Gemma can
preserve and recover work after an observed edit, with no manual patch repair.

Use a fresh full Duodoro archive at `5324cabba4ad3c8ad2e1cd1480218cf087d9b741`.
Keep the original T01 behavior, allowed file `client/src/lib/format.ts`, visible
check, independent external acceptance and reference unchanged. The request has
one explicit variant: replace **“Inspect AGENTS.md and the existing format tests
first.”** with **“Inspect client/src/lib/format.ts and
client/src/lib/format.test.ts first.”** The full exact request is in the manifest;
this is not an original-T01 repeat or a controlled quality comparison. The operator
read the pinned repository instructions; the file remains intact and available to
the model. No read is forbidden, no tool result is truncated and no application
prompt changes. The existing compact/whole-file preset is selected explicitly;
full remains the production default. Whole-file writes retain their stale-content
and large-rewrite limitations. Do not add coding hints during execution.

The new source/test-focused input is a deliberately narrower workflow probe, not
evidence that omitting instruction inspection improves task quality. It retains
the same negative/non-finite formatter requirements, finite/fractional/tag behavior,
single-file scope, protected tests/work, visible check and no-install/service/Docker
limits. It does not run Next/Vitest/typechecking/full application integration.

## Fixed allowance and environment

Exactly **one workflow**, at most two operator-submitted model turns. Each phase
reserves the same settings before execution and after restart:

| Setting | Each phase | Total reserved |
|---|---:|---:|
| Dispatch seconds | 150 | 300 |
| Reported total tokens, estimate gated | 32,000 | 64,000 |
| Reported output tokens | 4,096 | 8,192 |
| Governed tool calls | 12 | 24 |
| Model iterations | 6 | 12 |

Per-request output is 2,048, minimum output 256, explicit context 16,384, provider
request timeout 60 seconds, completion recovery one per phase, temperature zero
and thinking off. Require the exact visible check
`["python","-m","pytest","-q","test_visible.py"]`; plan gate off, root injection
and support off, checkpoints on. Use unseeded isolated fact/session storage and
user `duodoro-recovery-cli`. No automatic retry, third phase, budget transfer or
larger fallback. The operator enforces phase count; runtime limits are per turn,
not a persisted cross-turn budget controller.

Use local `gemma4:12b` with the manifest's digest and Ollama 0.33.2, pinned Python
3.14.5, Node 24.22.0, executables, environment inventory, lock and command PATH.
Begin with no loaded model/inference. Preserve the existing Ollama service; unload
only evaluation-owned Gemma between phases and afterward. No paid calls, new
models/repositories, dependency installation or Docker. Duodoro's working checkout,
index, branches and remotes are untouched; preparation only used `git archive`.

The canonical workspace is
`~/.local/share/agent-harness/trials/duodoro-recovery-cru8vt1w/workspace`.
Keep its root identity throughout both phases and reversal; no copy/substitution
to make a guard pass. Settings/storage remain identical across restarts. Validate
manifest/recipes/settings/criteria/reference/baseline, clean pinned runtime and
executable/environment/model hashes before dispatch. Scheduled storage/outcomes
must be absent. Missing/changed preconditions pause execution, not authorize a new
cohort. Evidence and source captures contain plaintext; this is trusted host-local
execution, not process isolation or a lock against external writers.

## Client procedure and failure paths

1. Submit the frozen request once through actual `chat --checkpoints
   --response-timeout 360`. Retain its printed session/key; one CLI keyed lookup
   identifies the run. The longer wait changes client read inactivity only.
2. Observe the allowed target hash without editing it. On its first change, inspect
   saved tool progress through the CLI until a successful `write_file` end is
   present; then request CLI cancellation **once**. Do not pause the real provider,
   delay its callbacks or rewrite its choices to manufacture a cancellation window.
   If work ends before cancellation, no edit appears, or cancellation arrives too
   late, retain that missed workflow. No replacement attempt. Unknown/external
   changes or unsettled ownership pause onward work.
3. Wait for owned settlement and export the original run/captures/transcript. Stop
   the owned API, verify process/listener exit and unload Gemma. Restart the same
   API/settings/storage/root. Only a cancelled source with an observed target edit
   may proceed to the reserved second phase; otherwise record phase 2 not run.
4. Use CLI workspace/transcript/configuration reviews. Pin the saved latest turn
   printed by workspace review, exact continuation input and stable key. Submit
   one `continue-session` with explicit uncaptured-state acknowledgment:
   **“Continue the same requested formatter change, preserving existing work;
   report actual checks and limits.”** A blocked fork consumes the planned phase;
   do not bypass it using a standard session or edit the source evidence. Read-only
   CLI run lookups observe settlement; they do not resubmit work. Source run/archive
   evidence must remain unchanged after the fork.
5. Stop/unload owned resources and independently grade the final files before any
   preservation action. Retain the entire scored artifact and raw grade, including
   failures. Protected criteria/tests/modes and source/reference bytes are unchanged.
6. Append one declared later line only to the setup-owned
   `preexisting-user-work.txt`, outside model-visible execution. Through the CLI,
   reverse eligible target differences from fork then source, in order, each with
   its own stable key. Inspect capture metadata first; missing pairs or no target
   difference are ineligible, not successes. Preserve any rejection/uncertain
   outcome and stop chained reversal; never reset Git, repair target bytes or relax
   guards. Applied operations must return target bytes/mode to their respective
   before captures while preserving later sentinel work and all other paths.
7. After another API restart, use read-only CLI run and reversal-status inspection.
   No new coding/reversal submission during this review. Retain historical checks
   as observations, not claims that the reversed artifact still passes. Stop owned
   API/listener/model and archive the entire final workspace without overwriting
   the scored artifact, original preparation or previous results.

The first source request uses the retained evaluation-only normalized identity
assertion before provider forwarding. It rejects drift, never rewrites/retries.
The second-phase first input depends on real saved history and cannot be frozen in
advance; the ordinary continuation workspace/profile/transcript/context guards
apply before inference. Phase 2 uses the actual application provider/shared runtime
without an injected pause or synthetic answer. No direct HTTP workaround replaces
a missing client control. Read-only SQLite exports are audit evidence, not a route
to select/repair continuation state; the CLI supplies the pinned turn.

## Evidence and disposition

Retain every scheduled outcome: source edit/cancel timing, completion/check state,
fork admission/result, independent grade, each reversal and restart review. Record
the phase count, numeric request journals/identities, reported usage and unknown
tails, request/tool counts, transcript/diffs, criteria/scope preservation, exact
recipes/log hashes, interventions and cleanup. A changed target plus saved write
end is an observation, not proof against concurrent writers or task acceptance.

Cancellation can make in-flight usage unknown. Keep the known prefix and unknown
total across both phases; **do not certify known within-budget token success** when
either phase is unknown. Estimates can undercount; dispatch is not a hard return
deadline. Setup/cleanup/final review/storage and operator observation add time.

Report workflow mechanics, artifact acceptance, completed-and-accepted status,
observed checks and known/unknown allowance evidence separately. A source can
already have an accepted patch when cancelled before check/answer completion;
that is still interrupted workflow evidence, not proof of repairing a bad patch.
Do not score prose as patch correctness or hide inaccurate answer illustrations.

One selected public request variant and one workflow do not establish repeatable
quality improvement, general recovery, browser recovery, cross-process locking,
held-out performance, ten-task success or release readiness. Original trial and
comparison results remain unchanged. G2's broader client/ten-task acceptance and
G3–G5 remain open/unmet even if this specific sequence passes.

## Preparation evidence

- Fresh initial visible/external checks fail; unchanged reference passes both.
  Full baseline, original criterion and reference hashes match; live state absent.
- A scripted provider captures the exact first source input/profile through the
  real API, with real read-only workspace/checkpoint capture and keyed replay in
  separate temporary storage. It does not instantiate/call Ollama. Four deliberate
  message/schema/allowance/temperature mismatches forward nothing; matching
  forwarding is unchanged. These are mechanics checks, not a live CLI attempt.
- The pinned runtime passes 41 focused and 1,363 offline tests (five live
  deselected), Ruff, formatting (260 files), mypy (57 source files) and both scripted
  90/90 escalation matrices; contract labels/metric limits inspected. The existing
  Chroma deprecation warning remains. These gates do not measure model quality.
- One initial scripted capture omitted required `finish_reason`; that failed
  preparation root is retained separately. It ran no inference and consumed no
  scheduled live row. The corrected fresh preparation preserves this correction.

Local fixtures, settings, profiles/messages, recipes, logs and hashes remain at
the manifest's root; this manifest is not a portable source/environment bundle.
All execution steps remain unrun until this PR is reviewed and merged.
