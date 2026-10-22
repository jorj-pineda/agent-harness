# Live panel recovery validation — protocol

**Protocol prepared; browser workflow and inference not run.** The
[machine-readable plan](panel-recovery-plan.json) records exact requests, limits,
criteria/reference hashes and inspected implementation hashes. It is deliberately
not an execution freeze: fresh fixture/storage/root identity, browser owner,
initial normalized request, execution recipes and environment still need to be
captured and independently reviewed before dispatch. No missing evidence is zero
usage or a passed preflight. This PR completes protocol work only.

## Question and fixed scope

Can the actual panel perform edit → explicit cancel → API restart → saved review
→ guarded fork/check → eligible single-file reversal without manual API calls?
The [CLI observation](RECOVERY_RESULTS.md) passed that exercised sequence but did
not validate browser controls, SSE delivery or browser-retained request identity.
This is one additional workflow, not a CLI-versus-panel quality comparison.

Keep the CLI probe's exact source/test-focused formatter request and continuation
message from the plan, original Duodoro source `5324cab`, single allowed target
`client/src/lib/format.ts`, visible check, external acceptance and reference.
The request remains a declared variant of original T01; full instructions/source
stay available and no read is forbidden or truncated. Use existing compact/
whole-file tools, not a new prompt/default. Whole-file stale-content/large-rewrite
limitations remain. No coding hints, patch repair, replacement workflow or model
switch after submission. Duodoro's working checkout/index/branches remain untouched.

The proposed application runtime is published `cd11e4c` (PR #167 merge). Source
hashes in the plan identify the inspected panel/server/guard implementation;
changes require renewed protocol review. Preserve original trial/comparison counts,
release no-go and G2–G5. No new feature batch or release approval is implied.

## Five-minute total allowance

Exactly one workflow; at most **two identical reserved phases**:

| Limit | Each phase | Total reserved |
|---|---:|---:|
| Dispatch seconds | 150 | 300 |
| Reported total tokens, estimate gated | 32,000 | 64,000 |
| Reported output tokens | 4,096 | 8,192 |
| Governed tools | 12 | 24 |
| Model iterations | 6 | 12 |

Keep local `gemma4:12b`, temperature zero, thinking off, explicit context 16,384,
2,048 request output, 256 minimum output, 60-second provider timeout, one completion
recovery per phase and identical-call limit two. Require exact
`["python","-m","pytest","-q","test_visible.py"]`; plan gate/root injection/support
off, turn captures on, inspection-only off. Use isolated unseeded memory/storage.
Unused allowance is not transferred. The runtime enforces per-turn limits; the
operator enforces the two-phase count. No third turn, warm-up or paid fallback.

Cancellation may leave an unknown usage tail. Keep known response prefixes and
unknown aggregate usage; do not certify known token compliance from a passing
artifact. Dispatch is not an end-to-end timeout: browser observation, setup,
cleanup, review, storage and grading add time. The CLI's response-timeout option
does not apply to the panel. Retain the existing SSE and 30-second continuation/
reversal request behavior; a browser timeout does not establish server termination.

## Preparation gates before any inference

1. Prepare a fresh full-source disposable fixture outside both working checkouts;
   verify initial visible/external failure, reference success and unchanged
   protected criteria. Do not copy an earlier scored/edited artifact.
2. Use the product-native collaborative browser in an isolated evaluation-owned
   profile/origin (intended loopback port 18751). Check control ownership first.
   Do not disturb another tab, user storage, provider login or Ollama service.
3. Run a real-browser **scripted-provider** preflight through the controls below,
   with real tools/checkpoints and process restart, in separate fixture/stores/
   profile from the scheduled live run. Deliberately control cancellation timing
   only in this labeled simulation, never in live inference. Create SQLite
   resources inside the application lifecycle thread. Record actual outcomes;
   code inspection and a browser-availability probe are not this preflight.
4. Run G0 on the selected clean runtime: focused panel/SSE/cancellation/continuation/
   reversal/persistence checks, offline suite, Ruff/format/mypy and both scripted
   matrices at unique paths; inspect contract labels and metric limits. Distinguish
   deterministic mechanics from model quality. Stop temporary resources afterward.
5. Commit an immutable final execution manifest and independently review it before
   inference. Pin actual workspace/root identity, baseline/criteria/reference,
   owner ID, empty scheduled storage, settings, runtime/tree/lock, served asset and
   recipe hashes, managed Python/Node/PATH, model digest/Ollama version and initial
   normalized source request. The plan's prior CLI model identity is proposed,
   not fresh backend verification. Freeze the complete operator procedure and
   pre-forward first-input guard; the fork's first input depends on real history
   and uses native continuation guards. Changed/unknown preconditions pause work.

No inference is authorized by scripted success alone. Execution stays a separate
slice under Jorge's local-model/Duodoro authorization after these gates. No new
model, paid call, repository, dependency installation or Docker action is assumed.

## Actual panel procedure

Use normal rendered controls through browser automation. Read-only DOM, network,
evaluation-owned local-storage and file observations may record evidence; do not
call application JS functions, mutate local state, inject `fetch`/EventSource,
disable transport features or use CLI/HTTP requests to replace a missing control.

1. Open the actual served panel in a fresh evaluation-owned browser origin. Record
   the generated `panel_user_id`; retain it throughout server restarts/reloads.
   Confirm no retained coding/reversal request exists. Set the frozen workspace,
   provider **ollama**, **Save turn captures**, inspection-only off; click **New
   session** once. Record the acknowledged session/capture policy.
2. Enter the exact frozen request and click **Send** once. Retain the source
   `panel_pending_request` key/input before it clears. Observe native SSE tool cards;
   constructor POST fallback or connection failure is a recorded transport-coverage
   failure, not permission to submit again. No automatic retry or replacement.
3. On a completed, non-error `write_file` card for the allowed target **and** an
   independently observed target-hash change, click **Cancel turn** once while it
   is enabled. Saved metadata is committed before SSE delivery in this runtime,
   but audit it after settlement. Do not attempt **Inspect saved tool progress**
   while streaming: history/review controls are disabled while busy. A mere tool
   start is not an edit. Record screenshots/timestamps and the exact cancellation
   target. A missed/late window, no edit or unavailable control remains failure;
   never pause the live provider to manufacture the window.
4. Wait for owned settlement and retain cancellation/check/diff/capture evidence.
   If SSE fails, use **Check run** or **Refresh history** once controls unlock to
   inspect the original key/run; reads submit no work. Do not click **Retry
   submission** in this probe. An unconfirmed outcome stays unknown. Only a
   cancelled source with an observed target change is eligible for phase two;
   otherwise preserve the outcome and mark phase two not run.
5. Stop the owned API, verify process/listener exit and unload evaluation-owned
   Gemma. Restart identical settings/storage/root on the same origin. Reload the
   same browser profile. Reload must submit no coding/file action. Use **Refresh
   history**, select the exact source run, then **Compare workspace**, **Inspect
   transcript**, **Estimate configuration/context** with the frozen continuation
   input and **Inspect saved tool progress**. Save displayed statuses and identities;
   matching/valid/fit are observations, not permission or backend attestation.
6. Open **Explicit guarded continuation**, click **Load latest saved turn**, record
   its source/turn, enter the exact continuation message, select the same provider
   and acknowledge uncaptured-state risks. Click **Fork and submit task** once;
   preserve the retained exact route/key. Use **Check run** or history selection
   for explicit read-only settlement inspection. The panel does not automatically
   poll. Check that any cancellation target belongs to the fork, not the source.
   Missing/changed evidence or a blocked fork consumes the planned phase; do not
   create a standard session, repair history or inject a direct API workaround.
7. Stop/unload owned resources after settlement. Independently grade the final
   artifact and preserve a full scored copy before any later preservation action.
   Keep failure, scope and output-coverage results. No editable check/prose can
   redefine authoritative acceptance; a fork may only check an already-correct
   source patch, which is not repair of an incorrect patch.
8. Append the declared later sentinel line, outside model execution, only to
   `preexisting-user-work.txt`. Restart the same API. Select eligible fork then
   source differences in order, review captures/diff and use **Reverse one
   checkpoint difference**, the exact target path, required acknowledgment and
   **Reverse selected file** once per eligible difference. Equal before/after
   target state is ineligible; submit no no-op reversal. Preserve rejections or
   uncertain outcomes and stop the chain without resets or repair.
9. After a confirmed applied result and preserved evidence, **Forget retained
   reversal** may release the browser controls for the next planned eligible
   difference; it changes only browser state, not server work. Record this action
   and both keys. Keep the last key retained for restart inspection. The panel has
   no arbitrary historical reversal-key lookup input: forgotten earlier outcomes
   can be audited read-only, but must not be claimed re-opened through a nonexistent
   browser control. Record this coverage limit if both reversals are eligible.
10. After another server restart and page reload, use **Inspect saved outcome**
    for the last retained reversal and history for source/fork records. Verify
    reload/lookups caused zero new coding/reversal submissions, target before bytes/
    mode and later unrelated work remain intact, and archives/journals are unchanged.
    Saved passing checks remain historical after reversal. Stop owned APIs/listeners,
    unload only evaluation-owned Gemma, close the owned tab and retain both complete
    scored/post-reversal artifacts. Preserve all user Git/workspace state.

## Outcomes and limits

Retain the single scheduled workflow even if transport, cancellation, guard,
grading or reversal fails. Report source/fork lifecycle and completion, check
relevance/staleness, artifact acceptance, completed-and-accepted status, phase count,
unknown usage, numeric journals, actual operator actions and cleanup separately.
Record served asset bytes, browser/profile/origin/owner, request keys/routes,
SSE/POST behavior, screenshots, tool metadata, root identity, scope/modes and
unchanged criterion/reference hashes. No browser rendering alone establishes
saved success; no read-only audit substitutes for a missing browser action.

One selected public request variant does not establish general recovery, repair,
client superiority, held-out quality, cross-process/external-writer isolation or
the original ten-task success target. Original counts/no-go, broad G2 and G3–G5
stay unchanged unless their separate acceptance criteria are actually met.

## Validation of this protocol PR

Inspected current panel markup/handlers, API save-before-SSE behavior, relevant
UI/SSE/cancellation/continuation/reversal tests and roadmap. A temporary blank
isolated browser tab confirmed automation availability and was closed. This was
not a panel preflight: no server, model or coding run was started. Source IDs,
labels, budgets, copied request/criterion hashes, unknown execution fields and
local documentation links are checked for this docs/data-only PR. No offline
suite or matrix rerun is claimed; the future preparation gate remains explicit.
