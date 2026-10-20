# Duration toolset comparison — execution freeze

**Prepared for review; no inference has run.** The [manifest](toolset-freeze.json)
pins four fresh T01 attempts using two existing presets at the same published
runtime, `ce5d54c7394c476afcee29a8981fecd96631cc9f`. Review and merge must complete
before dispatch under Jorge's ongoing instruction to continue focused work.
Scripted success alone grants no inference or release approval. Changed/missing
assets or preconditions pause execution; do not silently rebuild a different cohort.

## Question and intervention

The [same-root batching comparison](BATCH_READ_RESULTS.md) completed no workflows:
all four attempts stopped before checks after seven requests, and the new sentence
never activated in available normalized history. Keep that sentence out of production.
Before implementing compaction, test whether the existing small-file editing
preset can complete this selected task under the unchanged limits.

Compare `CODING_TOOLSET=full` with `CODING_TOOLSET=whole_file`, both compact.
Whole-file mode removes only `replace_text` and uses the existing matching
`write_file` instruction. Every retained schema is identical; runtime source,
sampling, other prompts/settings and full instruction/tool-result retention match.
This tests a **preset plus its necessary instruction**, not tool removal alone,
automatic batching, an escape decoder, a smaller instruction file or compaction.
The initial prompt heuristics are 3,118 full / 2,796 whole-file tokens; these are
not measured backend counts or predicted task savings.

The historical October 7 whole-file observations in [BASELINES](../../BASELINES.md)
were mixed (9/15 versus the preceding 10/15 harness artifacts; not a clean paired
same-input contrast). They do not justify switching the default. Here the selected
file is small; whole-file writes still lack exact-edit stale-hash protection and
can introduce unrelated changes. Do not generalize to large files or concurrent
writers. `full` remains the production default regardless of a single sample.

## Fixed cohort and budgets

Use the exact original T01 request, Duodoro source `5324cab`, full-source fixture,
visible command, external acceptance, protected baseline and reference. Only
`client/src/lib/format.ts` may change. Preserve both formatter contracts and tag
behavior; editable tests cannot redefine authoritative acceptance.

| Order | Repeat | Preset |
|---:|---:|---|
| 1 | 1 | full |
| 2 | 1 | whole_file |
| 3 | 2 | whole_file |
| 4 | 2 | full |

Each actual CLI submission uses a fresh session and independent unseeded storage,
user `duodoro-toolset-cli`, checkpoints enabled, inspection off and the **same
literal workspace**:

`~/.local/share/agent-harness/trials/duodoro-toolset-pl69m2cm/workspace`

Keep 300 dispatch seconds, 64,000 total reported tokens, 8,192 output tokens,
24 governed tools, 12 model iterations, 60-second requests, 2,048 request output
and 256 minimum output. Context is 16,384; temperature zero, thinking off, required
verification, plan gate off, one existing completion retry, root injection/support
off. Model is local `gemma4:12b`, frozen digest/Ollama 0.33.2, with pinned Python
3.14.5, Node 24.22.0, dependency lock/environment and managed command PATH.
No larger allowance, warm-up, new model, paid fallback or dependency installation.
Begin each attempt with Gemma unloaded and no other inference/loaded model.
Setup/cleanup/final review/storage are not a hard return-time guarantee.

No replacements, manual patch repairs, new coding hints or follow-up coding
submissions. A timeout consumes its scheduled row. The CLI's 120-second response
timeout can precede server settlement: inspect the original key/run read-only;
never resubmit with a new key to obtain a better result.

## Boundary and transfer checks

Both arms use the same retained `guarded-server.py`: normal API, provider and shared
runtime execution, with an evaluation-only provider wrapper that checks the frozen
first normalized request SHA and temperature zero **before forwarding**. It does
not rewrite requests, retry or prescribe later model choices. Within-preset repeat
messages/schemas are identical. Across presets, only the matching edit instruction
and removed tool differ; output allowance and all other profile/settings fields
match. Guard hashes/probes are frozen. These observations do not attest raw wire
serialization, hidden backend state, deterministic sampling or remote accounting.

Before any scheduled state/API/model submission, verify clean pinned runtime/tree,
lock/environment/executable hashes, each settings/criterion/reference/baseline,
owner marker and literal root identity with no symlink substitution. The manifest's
scheduled state paths must still be absent. Start no writer on uncertain ownership.

After each run, wait for owned work to settle and export its original saved run,
normal transcript, numeric budget evidence, tool/check evidence and bounded diffs.
Grade final files with frozen external checks and verify protected criteria.
Stop the owned API, verify listener absence, unload only evaluation-loaded Gemma,
and preserve the **entire** evaluation-owned common root under that attempt's
outcomes. Never overwrite an archive. Only after the common path is absent and
no writer remains, copy the next fresh fixture there, verify its full baseline
and record its new root identity. Never seed from an earlier artifact/store.

This is explicit operator staging, not resumed editing. Saved history grants no
execution permission. Process-local admission and path checks are not cross-process
locking or host isolation. Do not touch the Duodoro working checkout, its Git
index/branches, Docker or unrelated Ollama service. Preserve failed/partial work;
uncertain cleanup or external activity leaves remaining rows unrun/unknown.

## Evidence, activation and disposition

Primary outcome: **completed-and-independently-accepted out of two per preset**.
Report all four rows, artifact acceptance, completion and observed checks separately.
Failed/blocked/missing responses remain in the denominator. A partial cohort does
not become an equal-cohort success ranking. Unknown/truncated evidence stays unknown.

Record request count, reported per-request/cumulative usage, gate estimates/stops,
unknown failure tails, timing scopes and every operator intervention. Preserve
bounded review evidence and raw available transcript outside model context. Numeric
request observations do not attest provider delivery/usage or hard budget compliance;
tool-call records can include errors/rejections rather than governed dispatches.

Inspect actual edit-tool choices: whole-file treatment activation requires a
successful `write_file` result on the allowed target, not a scripted probe, tool
availability or model prose. Full mode can also choose `write_file`; retain that
overlap. Record malformed/blocked calls, repeated reads, exact-edit mismatch
recovery, checks and unrelated rewrites. Fewer requests or zero exact-edit errors
alone do not establish working code or completion. No claim that the model learned
recovery when its failing tool was simply removed.

Two repeats on one selected public task/backend cannot establish general causal,
held-out, category-level or release-quality improvement. Do not change production
presets, budgets, original trial/feedback/batching outcomes or release gates from
this comparison. G2's real partial-edit/reversal evidence, G3's ten-task target,
G4's held-out/two-model requirements and G5's decision remain open/unmet.

## Offline preparation

All four fresh baselines/criteria match the original; initial visible/external
acceptance fails and the reference passes. Scripted shared-runtime/API probes
perform real reads, retain complete `AGENTS.md`, capture checkpoints, match the
saved profiles/first requests and replay the same key without new forwarding.
Eight altered path/schema/allowance/temperature cases forward nothing; matching
inputs and provider cleanup pass through unchanged. The fake read batches test
mechanics only; they are not model activation or actual CLI coding submissions.
Probe stores are separate temporary resources; scheduled stores remain absent.

The **one shared pinned runtime** passed 71 focused regressions (including both
toolsets/profiles), 1,344 offline tests (five live deselected), Ruff, formatting
(259 files), mypy (57 source files) and both scripted 90/90 escalation matrices
with contract labels/metric limits inspected. The existing Chroma deprecation
warning remains in the development environment. These checks do not measure
model quality. Reporting commits are not the execution runtime.

Raw fixtures, profiles/messages, settings, guard, probes, G0 logs and recipes remain
at the local trial root. The manifest freezes their hashes; this is not a portable
full-source bundle. Gemma remained unloaded. No inference, paid call, Duodoro
working-checkout edit, default-prompt change or Docker action occurred.
