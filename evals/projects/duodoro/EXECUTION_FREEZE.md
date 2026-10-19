# Ten-task execution freeze

The [manifest](execution-freeze.json) pins the trial to runtime commit
`cd9891718193ae4d9bb56056ae8aea8e57a9ef4c`, the PR #147 merge. The manifest and this
document are a review copy added afterward; **do not substitute their later merge
commit or latest main as the execution runtime**. This PR changes documentation
and data only. No trial submission or inference occurred during freezing.

G1 task/settings agreement is complete; its remaining freeze review is pending.
The manifest requires a completed review/merged PR record before dispatch. Its
presence alone is not approval. The review checksum (SHA-256) is
`26f9971e635d204599a38968b10427bef967178890a1c041d26eb3cfc7bfb62d`.
The agreed ten requests, CLI/panel assignments,
limits, recovery rules and independent acceptance remain as approved in the
[protocol](TRIAL_PROTOCOL.md). G2–G5 and the broader held-out/two-model release
requirements remain open. A mechanics gate is not a release or quality result.

## What is pinned

- A clean, detached runtime worktree at the exact commit/tree, plus dependency-lock
  hash, Python/package inventory and managed executable identities.
- All ten verbatim task requests and UTF-8 hashes, allowed paths, categories,
  client assignments, visible setup hashes, external criterion/reference hashes,
  and newly prepared full-source workspace baselines (content and regular-file
  modes). The original three task definitions and criteria remain unchanged.
- Complete explicit per-task settings JSON outside the editable fixture, distinct
  session/memory storage per task, captures enabled, no seeded memory, support/root
  instruction loading disabled, and both configured execution profiles. The T03
  phases use the exact same settings and storage paths across restart.
- The frozen continuation input/key and the rule to pin its actual latest settled
  source turn. Ordinary CLI/panel submission keys remain client-generated: retain
  the actual key before posting and record its IDs; do not invent a CLI option.
- T05's post-grade single-file reversal, including exact later sentinel text and
  preservation checks. Grade and retain the coding artifact before reversal.
- Decision rules separating artifact acceptance, completed-and-accepted, observed
  checks, client workflow, unknown budget compliance and operator intervention.
  T08 still requires independent final-patch review of actual helper reuse; an
  automated passing behavior grade alone remains provisional.

## Model and environment observation

Read-only local `/api/tags` and `/api/version` reported Ollama `0.33.2` and
`gemma4:12b` with digest
`4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`.
The model metadata reports 11.9B / Q4_K_M. This identifies the installed model
available through that endpoint; it is not loaded-model attestation. No `/api/chat`,
embedding, model pull or inference request was made. The pre-existing Ollama
service was left running; no evaluation-owned inference process was started.
Recheck the digest and backend version before dispatch and stop on drift.

The frozen managed Node is the concrete `24.22.0` executable under
`~/.vite-plus/js_runtime/node/24.22.0/bin/`, not the version-selecting shim.
This shell initially resolved Node 26, including through the old shim path.
The final fixture validation, focused tests and full G0 were rerun with the
concrete Node 24 directory ahead of the shell PATH. The final manifest records
that PATH, executable paths and binary hashes. Python is the recorded existing
harness `.venv` interpreter, version 3.14.5; use it without dependency syncing.
An editing invocation briefly created a separate unused ignored runtime `.venv`;
it was removed. It was not the environment used for G0 or the frozen settings.
No Duodoro dependencies or working-checkout files were changed.

## Evidence and runtime gate

The persistent local bundle is outside both source checkouts:
`/Users/jorge/.local/share/agent-harness/trials/duodoro-cd98917-freeze-ifjtk9jq/`.
It contains the pinned `runtime/`, ten `tasks/Txx/fixture` directories, external
criteria/baselines, per-task `settings.json`, preparation/verification scripts,
model metadata, gate logs, reports and the unrun execution ledger. The repository
manifest is byte-identical to the bundle's `execution-manifest.json`; its sidecar
`execution-manifest.sha256` provides the review checksum. These are trusted-local
artifacts protected by recorded hashes, not an OS isolation or hostile-code claim. Input
protocol/template hashes refer to the pinned runtime copies; later documentation
links do not replace those frozen inputs.

At the pinned revision and final Node 24 environment:

- All ten fresh full-source tasks failed their initial visible/external checks and
  passed both checks with the published reference; criterion bytes stayed unchanged.
- 41 focused preparation/acceptance regressions passed.
- G0: 1,291 offline tests passed, five live tests deselected; Ruff, formatting
  (253 files) and mypy (56 files) passed. The dependency deprecation warning is
  recorded in the raw pytest log.
- Both scripted matrices passed 90/90 escalation decisions. Their reports retain
  `scripted-contract` labels and explain that overlap/trace metrics are not
  independent acceptance or model-quality evidence.
- Independent rehashing confirmed every fresh baseline, request, criterion,
  validated settings file and unrun state; the pinned runtime remained clean.

The local ledger has ten unrun rows bound to the manifest digest; outcome fields
remain null. The repository ledger remains a template. Never overwrite either
frozen criteria or the committed manifest with trial outcomes.

## Dispatch checklist after review

1. Verify the merged review record and the manifest checksum; rehash the pinned
   runtime, lock, executable/package identities, criteria, settings and full fresh
   workspace baselines. Stop on missing assets or drift; do not silently refresh
   the cohort or lower criteria. Fresh preparation elsewhere must reproduce the
   frozen baselines and be recorded as a new reviewed mapping.
2. Launch the **pinned runtime** with the recorded interpreter and PATH, one API
   process, no reload/extra workers, loopback only. Set its working directory and
   `PYTHONPATH` to the pinned runtime; launch with only the managed environment
   allowlist, rather than inheriting provider credentials/settings. Construct
   `Settings(_env_file=None, **json.loads(task_settings_file.read_text()))` so all
   fields are explicit and operator `.env`/ambient keys do not leak into the run.
   Use each task's separate storage and its supplied workspace through its assigned
   real client. Do not expose reference/criterion directories as workspace roots.
3. Verify model digest/version again. Create a fresh client session with captures
   acknowledged; submit the exact frozen request once. Retain actual user/session,
   key/run IDs and full evidence in the external ledger. Transport uncertainty
   requires lookup, not a fresh key, extra attempt or hidden HTTP workaround.
4. Follow the protocol for the one planned T03 cancellation/restart/fork and T05
   post-grade reversal. Only T03 gets two reserved phases, never a third full-budget
   retry or transfer of unused allowance. Record missed cancellation/blocked forks
   honestly; never bypass a failed guard.
5. Grade every final coding artifact with the pinned external acceptance runner
   and unchanged criteria; perform T08 independent source review. Record all ten
   outcomes, failures, interventions, known/unknown usage and elapsed scopes.
   Interrupted usage remains unknown, never zero or an established within-budget
   success. Preserve the original artifact/grade before planned reversal.

No inference is part of this freeze PR. Actual trial execution remains a separate
next action after review; no new feature or premium supervision is introduced.
