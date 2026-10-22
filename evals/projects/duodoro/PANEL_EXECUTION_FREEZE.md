# Bounded live panel recovery — execution freeze

**Prepared, not run.** Review and merge this document and the
[immutable manifest](panel-execution-freeze.json) before executing the one scheduled
workflow. No inference ran during preparation. This completes the preparation
gates in the [panel protocol](PANEL_RECOVERY_PROTOCOL.md), together with the
[scripted browser preflight](PANEL_PREFLIGHT_RESULTS.md); it does not close G2 or
change the release no-go decision.

## Frozen inputs

- Runtime `cd11e4c33fc89f4331b50891266a9c2fef57caef`, clean detached worktree.
  The manifest pins its tree, lock, inspected server/guard/UI source and served UI
  bytes. No application behavior, defaults, prompt or toolset changed.
- One fresh full-source Duodoro fixture from `5324cab`, original duration criteria
  and reference. Initial visible/external checks fail; reference checks pass.
  The scheduled workspace matches the full acceptance-eligible baseline, including
  pre-existing work and regular-file modes. Duodoro's working checkout is unused.
- The exact source/test-focused request variant and continuation from the earlier
  protocol, compact/whole-file configuration, required visible Python check,
  support/root-instruction injection/plan gate off, captures on, inspection off.
  Full source/instructions remain available. This is not an original-T01 repeat.
- Local `gemma4:12b` digest and Ollama version were freshly checked through metadata
  endpoints; the model remained unloaded. Managed Python/Node, PATH, installed
  distributions and settings are pinned. No dependency installation or paid call.
- Exactly one workflow, at most two identical reserved phases: each 150 dispatch
  seconds, 32K total reported tokens, 4,096 output, 12 tools and six iterations.
  No budget transfer, replacement, warm-up, third turn, manual coding rescue or
  model switch. Cancellation usage may be unknown; dispatch is not total latency.

The new scheduled root is named in the manifest. Its `state/` and `outcomes/`
directories are absent. A scripted first-input capture used separate
`capture-state/` stores and changed no workspace files. Those preparation stores
must never be substituted for scheduled storage.

## Browser and request guard

The pinned static panel was served temporarily on `http://127.0.0.1:18751` solely
to obtain its normal generated owner, `panel-q8dle5lu`. The evaluation-owned
incognito tab is retained, with only `panel_user_id` in origin storage. No session,
coding submission or reversal exists there. The bootstrap process exited and
listener stopped; no API is running. Recheck tab ownership, exact owner and absent
pending keys before launch. Lost/changed browser identity blocks execution; do not
inject storage or silently regenerate inputs to fit this freeze.

The prior scripted browser preflight used the separate origin 18752 and separate
stores/root. It used a new preservation file; this live workflow instead appends
the manifest's exact line to existing `preexisting-user-work.txt` after scoring.
The same eligible target reversal and unrelated-work preservation controls apply.

A no-inference TestClient capture through native SSE records the first normalized
request and execution profile with the scheduled owner/settings/workspace. It is
preparation for a guard, not a browser action substitute or a live result. The
local guarded server asserts the first source request hash and temperature before
forwarding unchanged to Ollama. Offline checks reject altered messages, tools,
allowance and temperature with zero forwarding, confirm unmodified matching
forwarding/async cleanup and reject all review-only provider calls. No guard pauses
or rewrites live responses to create a cancellation window. The fork's first input
depends on actual saved history and uses native continuation guards.

## Operator procedure and local recipes

Follow all ten steps of the original [protocol](PANEL_RECOVERY_PROTOCOL.md); its
exact bytes are retained and hashed with the manifest. The manifest also freezes
the sentinel line, two-phase rules, capture/hash cancel trigger and retained-key
limits. Use rendered browser controls for create/send/cancel/history/reviews/fork/
reversal. No app-function calls, storage mutation, injected requests or direct
HTTP/CLI workflow workaround. On missing/failed/late controls or evidence, preserve
the one scheduled outcome without replacement or patch repair.

Local recipes at the evidence root are hashed in the manifest:

- `check-preconditions.py`: read-only runtime/settings/recipe/binary/package/model/
  root/criterion checks; source additionally requires absent scheduled stores and
  a matching initial manifest. Run in the pinned managed Python environment.
- `guarded-server.py`: source first-input guard, normal guarded-fork application,
  or review-only inference rejection; pass-through HTTP identity observer.
- `panel-ops.py`: explicit start/stop lifecycle only. Start requires a merged PR
  whose committed manifest bytes equal the local manifest, and passing read-only
  preconditions. Fetch the reviewed merge commit before starting. The operator
  must independently check review approval, browser identity and phase eligibility.
  It does not submit coding, cancel, continue or reverse through HTTP.
- Preparation, fixture, input-capture, guard-check, G0 and finalization recipes,
  plus their retained logs and outputs. These are scoped local plaintext evidence;
  hashes identify them, not a remotely downloadable bundle.

For execution, stop/unload each settled source/fork API and evaluation-owned model
before the next phase; the helper verifies model identity and refuses unexpected
ownership. Preserve a scored copy and independently grade before later sentinel
work or any reversal. Review eligible fork then source differences, stopping on
rejection/uncertainty and skipping equal/ineligible pairs. Inspect/save an applied
key before explicitly forgetting it to release controls for the next reversal;
retain the last key for restart inspection. Forgotten earlier keys are audit-only
because the panel has no arbitrary-key input. No-op reversal, retry, new session or
direct API rescue is permitted. Historical passing checks are stale after reversal.

Final review must audit unchanged run/session/response evidence, target baseline
bytes/mode, later preservation line and immutable criteria; no reload/lookups may
submit coding or file actions. Stop owned APIs/listeners, unload only owned Gemma
and close the retained tab when execution ends or is abandoned. Preserve user
worktrees/branches and the original trial/comparison counts.

## Preparation validation and limits

At the pinned runtime, using its managed installed environment:

- 53 focused panel/SSE/cancellation/continuation/reversal/persistence/progress tests.
- 1,363 offline tests passed, five live deselected; existing Chroma deprecation warning.
- Ruff check, formatting check (260 files), mypy (57 source files) passed.
- Coding and support scripted matrices each passed 90/90 escalation decisions at
  unique report paths; scripted-contract labels and metric limits were inspected.
- Fresh fixture/reference/criterion validation, served asset hashes, first-input
  guard checks, matching source preconditions and browser-owner readback passed.

These are preparation/mechanics results, not model-quality evidence. No live run,
successful recovery, complete Gemma task or release acceptance is claimed. Actual
execution remains a separate slice after independent review and merge, under the
existing local-model/Duodoro authorization and unchanged preconditions. Original
cohorts, broad G2 and G3–G5 remain open/unchanged. External writers, ignored contents,
dependency state, ACLs/ownership and hostile-code isolation retain their existing
limits. The adapter checks library behavior, not the full Next/Vitest suite.
