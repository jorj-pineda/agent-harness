# Paired exact-edit feedback comparison: execution freeze

**Prepared for review; no inference has run.** This is a new public development
comparison, separate from the completed ten-task trial. Its original 2/10 accepted
and 1/10 completed-and-accepted results and all release gates remain unchanged.
Review this protocol and the [manifest](edit-feedback-comparison-freeze.json)
before dispatch. Do not substitute the later reporting PR revision for either
pinned runtime, refresh drifted inputs or overwrite the original trial manifest.

## Question and isolated variable

Does the mismatch-feedback change help local `gemma4:12b` complete the three tasks
that exhibited the escaping failure? T01, T04 and T09 were selected because of
known failures; this is not held-out evidence or an unbiased repository sample.

Both arms use runtime base `5974d2201cb12829de2b593b602c649515e98b0d`, including
PR #151's safe failure reporting. Treatment is that exact published revision.
Control is a clean local experiment commit that restores only
`_replace_mismatch_error` from pre-fix revision `84d6ab8`. The committed
[control patch](edit-feedback-control.patch) and manifest pin its bytes, function,
Git commit and tree. All other runtime content is identical. No new production
switch or alternate harness loop was introduced. Both use the actual source API,
provider and real tools, with the same prompt/schema/check configuration.

The local control commit is retained in a detached worktree and a Git bundle with
the published base as prerequisite, not pushed as a remote branch. Its bundle is
local evidence, not included in this PR. A fresh reconstruction can apply the
patch to the published base; compare the resulting tree to the frozen control
tree. A different commit/path mapping requires a recorded reviewed mapping before
execution, not silent acceptance of drift. The preparation recipe is retained
locally and hashed in the manifest; this is not a self-contained public trace kit.

## Fixed attempts, settings and clients

There are **12 coding submissions**: three tasks × two arms × two repeats.
Each attempt has a fresh full-source disposable fixture, external acceptance,
settings file and independent unseeded storage. Requests, allowed files, baseline
bytes/modes, visible checks, reference patches and authoritative criteria match
the original freeze. No edits to the user's Duodoro checkout, commits or pushes
there are part of this comparison.

| Task | Assigned client | Repeat 1 | Repeat 2 |
|---|---|---|---|
| T01 duration formatting | CLI | control, treatment | treatment, control |
| T04 share storage | Panel | treatment, control | control, treatment |
| T09 tag filter | CLI | control, treatment | treatment, control |

Execute rows in manifest order: all repeat-1 pairs in table order, then all
repeat-2 pairs. Use the same real client for both arms of a task. There are no
T03 recovery phases, T05 reversal actions or replacement tasks in this slice.

Each attempt retains the approved 300-second dispatch allowance, 64,000 total
reported tokens, 8,192 output tokens, 24 tools and 12 iterations. Requests retain
60 seconds and 2,048 output tokens (minimum allowance 256); context is 16,384,
compact/full tools, temperature zero, thinking off, root instructions/support off,
required verification and turn captures on. Exact validated settings are saved
outside editable roots; only four per-attempt storage paths differ from the
original one-turn settings. File/scan/diff/policy limits remain those settings.
These are estimate/dispatch gates, not hard end-to-end cleanup/return bounds.

Use the pinned model digest, Ollama version, concrete Node 24.22.0 and Python
3.14.5 executables and dependency lock/environment identities in the manifest.
No dependency installation, warm-up prompt, hosted fallback, extra model, paid
call, larger prompt, timeout extension or budget increase is authorized here.

## Dispatch, settlement and preservation

1. After review/merge, verify the manifest checksum, clean pinned runtime commits,
   exact control-only diff, executable/environment identities, model metadata,
   settings and all fresh workspace/criterion/reference hashes. Missing assets,
   unavailable versions or drift stop dispatch; do not regenerate silently.
2. Use one loopback API process and one active writer, with the attempt's pinned
   runtime as working directory/PYTHONPATH. Use the recorded interpreter/PATH and
   managed environment allowlist; pass every setting explicitly with
   `Settings(_env_file=None, **saved_settings)`. Do not inherit credentials or use
   an installed package from the reporting checkout as the execution runtime.
3. Start each attempt with Gemma unloaded, no other loaded models and no other
   inference owned by this comparison. This fixed cold-start rule applies equally
   to both arms and is separate from the historical trial's cache/order conditions.
   Submit the exact request once through its assigned CLI/panel, retaining actual
   user/session/key/run identities. Read-only lookup resolves client timeout or
   stream uncertainty; never resubmit with a new allowance/key or use direct HTTP
   to rescue a missing client control.
4. Settle API-owned work and save the final artifact/evidence before moving on.
   Stop the owned API; unload only the evaluation-loaded Gemma runner and verify
   it is absent. Leave the existing Ollama service and unrelated processes alone.
   Unexpected concurrent activity, unsettled work, storage corruption or missing
   acceptance assets pauses the cohort. Ordinary terminal provider errors/budget
   stops remain failures and consume their scheduled row; after settlement the
   next fixed row can proceed. No hidden retry, replacement or patch repair.
5. Grade every final coding artifact using the frozen external runner/criteria,
   preserving all unmodified paths, pre-existing work and modes. Save hashes,
   bounded diffs, visible checks and run records. Rehash acceptance afterward.
   Do not modify criteria or use editable tests to redefine acceptance.

## Evidence and decision rules

Retain all 12 planned rows, including not-started rows if execution must pause.
Report raw acceptance counts out of six attempts per arm and completed-and-
accepted separately, plus per-task repeat counts and six paired outcomes. Record
observed checks, mismatch errors, later exact-edit corrections, file changes,
known/unknown usage, failure categories, interventions and scoped timings.
Unknown failed/interrupted usage is never zero or proven budget compliance.
Provider failure metadata is an observation, not proof of backend cancellation.

A later edit after feedback is a recovery observation, not proof that the message
caused it. Retain prefix divergence, order, loading and sampling limitations;
temperature zero is not deterministic replication. If no affected mismatch occurs,
the attempt does not test recovery from that hint. A small or mixed change cannot
establish a repeatable general quality gain. Do not add tasks/repeats after seeing
scores. This comparison cannot meet G2's reversal requirement, G3's ten-task target
or G4's held-out/two-model gate. A favorable result does not authorize release.

## Offline preparation evidence

Three selected tasks passed fresh initial-visible/external-fail and reference-
visible/external-pass validation. All 12 prepared fixture/criterion baselines match
the original freeze. Runtime/profile hashes, the exact schedule and unrun/null
outcome fields are recorded in the manifest.

Treatment runs the complete offline gate. Control deliberately restores the old
behavior, so its common-regression gate excludes exactly the five parameter cases
of `test_ollama_exact_edit_feedback_preserves_arguments_and_requires_explicit_correction`
that assert the new diagnostic. This is a declared experimental control exclusion,
not a claim that control passes the new behavior's tests or is a release candidate.
All remaining tests, including literal/stale/ambiguity/preservation/provider failure
coverage, run on control. The full results and static/scripted gate logs are pinned
in the manifest. No model-quality inference follows from these gates.

Pinned offline gate results: treatment **1,330 passed / five live deselected**;
control common regressions **1,325 passed / ten deselected** (five live plus the
five declared treatment-diagnostic cases). Both arms passed Ruff, formatting of
256 Python files and mypy on 56 source files. Both scripted sets on each arm
passed 90/90; their scripted-contract labels and metric limitations were inspected.
Offline profile captures agree across all 12 attempts. The existing Chroma
warning remains. An initial static command created an alternate worktree `.venv`
without the requested static tools; it was removed and the checks were rerun
using the absolute pinned environment. It was not used for inference or the
recorded pytest gates. No evaluation model was loaded during preparation.
