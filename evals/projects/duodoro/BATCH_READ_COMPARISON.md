# Duration read-batching experiment: execution freeze

**Prepared for review; no inference has run.** The [manifest](batch-read-freeze.json)
pins four fresh T01 duration attempts. Review/merge and a subsequent execution
instruction are required before dispatch. This does not publish a prompt change,
refresh historical results or authorize a release.

## Question and intervention

Can one instruction encouraging independent read batches help local `gemma4:12b`
finish the duration workflow under the existing limits? The
[diagnosis](DURATION_BUDGET_DIAGNOSIS.md) found two accepted patches stopped before
verification with 11,057 total tokens left and a 12,762-token next-prompt heuristic.
Seven model requests repeatedly carried a large instruction result. The retired
read-elision trial never activated and preserved that instruction result; it does
not establish that compaction will help. Here every instruction and tool result
remains intact. The intervention targets repeated round trips, not content removal.

Treatment adds exactly this sentence to the shared compact prompt:

> - Batch independent read-only tool calls in one response when their arguments are already known.

The [patch](batch-read-treatment.patch) changes only `harness/prompts.py` in a local
experiment runtime. Control is published base
`45f4182cde0f7652f7e2f9cad23696b5f7373f11`; treatment is local commit
`1f7e0f6804edccb154ec2542012074f5922a03b7`. Both retain the inherited
`coding-v3-compact-v1` label: distinguish variants by the pinned commit/tree,
prompt-file/base-prompt hashes and first-request fingerprints, never that label
alone. No production option, automatic batching, parallel tool execution or new
recovery allowance is added. Existing tool gates and sequential execution remain.

The treatment Git bundle, with published base as prerequisite, stays local; the
patch and tree hash permit reconstruction. Missing local assets, a different
commit/path or environment drift require a reviewed mapping before execution,
not silent regeneration. The reporting PR revision is neither inference runtime.

## Fixed scope and limits

Use the original exact T01 request, pinned Duodoro source
`5324cabba4ad3c8ad2e1cd1480218cf087d9b741`, full-source fixtures, protected baseline,
visible command, external acceptance and reference. Only
`client/src/lib/format.ts` may change. This is one public development task selected
after known failures, not held-out evidence or a ten-task release trial.

All four coding submissions use the actual CLI, one fresh session and independent
unseeded storage per attempt, with checkpoints enabled:

| Order | Repeat | Arm |
|---:|---:|---|
| 1 | 1 | control |
| 2 | 1 | treatment |
| 3 | 2 | treatment |
| 4 | 2 | control |

No replacement tasks, manual patch repairs, new coding hints or follow-up coding
submissions are allowed. Existing bounded runtime recovery remains configured.
Lost responses use the original key and read-only CLI inspection, never a new key.

Each attempt retains **300 dispatch seconds**, 64,000 total reported tokens,
8,192 output tokens, 24 tools, 12 iterations, 60-second requests and a 2,048
per-request output cap with minimum 256. Use context 16,384, compact/full tools,
temperature zero, thinking off, root injection/support off and required verification.
The plan-before-edit gate stays off. Exact other settings remain in each hashed settings file;
only four storage paths differ between attempts. No timeout extension or larger
budget. Setup/cleanup/review are not a hard return-time guarantee.

Use the recorded model digest/Ollama 0.33.2, Python 3.14.5, Node 24.22.0, package
environment and dependency lock. No dependency installation, model warm-up,
additional model, paid/hosted fallback, Docker or user's Duodoro checkout edits.
Begin every attempt with Gemma unloaded and no other loaded models/inference.
Unload only evaluation-loaded Gemma afterward; retain the existing Ollama service.

## Workspace identity and first-request assertion

All attempts use the **same literal execution workspace** from the manifest:

`~/.local/share/agent-harness/trials/duodoro-batch-read-1haw4enm/workspace`

Prepared per-attempt fixtures and storage are elsewhere. Arm/repeat staging paths
must never become session roots or injected task text. First-request probes prove
identical messages across repeats within each arm, identical tool schemas and
2,048 output allowance, and exactly the intended system-prompt sentence across
arms. Initial estimates are 3,119 control / 3,151 treatment tokens, not tokenized
backend counts. Full instruction/tool-result text is preserved.

Both arms use the same hashed local `guarded-server.py` evaluation recipe. Its
provider-boundary assertion compares the first normalized request SHA to that
attempt's frozen identity and requires temperature zero **before forwarding**.
Matching messages/tools/allowances pass through unchanged; mismatch pauses the
cohort without rewriting inputs or retrying. The normal source API, provider and
shared runtime own execution. This is an experiment precondition, not a shipped
application feature or premium-model supervisor. Later requests are outcomes and
are not constrained to follow a scripted path. Hashes do not attest provider wire
serialization, server defaults, deterministic sampling or hidden backend state.

Workspace transfer is explicit operator-owned staging, not automatic resume:

1. Before the first run, verify the frozen owner marker/hash, literal root with no
   symlink substitution, initial root identity, every fixture byte/mode, settings,
   criteria/reference, clean pinned runtime, environment and first-request identity.
   Scheduled `state` paths must still be absent. Stop on drift; do not reset it.
2. Submit the exact request once through CLI as `duodoro-batch-cli`, with the common
   root and checkpoints. Retain session/key/run identity, normal archive, budget
   observations, tool evidence, bounded diffs and final files. Do not use saved
   history as permission to continue editing.
3. Settle API-owned work, grade the final artifact with frozen external checks and
   rehash protected criteria. Stop the owned API and unload Gemma before transfer.
   Archive the entire evaluation-owned root to that attempt's outcome directory;
   preserve it, including failed/partial work. Never overwrite an existing archive.
4. Only after the common path is absent and no writer remains, copy the next fresh
   prepared fixture there, verify its full baseline and record its new root
   identity. Do not reuse a previous attempt's artifact, facts, checkpoint or store.
   No user checkout, Git index, stash or unrelated path is reset. External activity,
   uncertain ownership or unsettled work stops the cohort with remaining rows unrun.

Existing admission covers one process; neither transfer checks nor path validation
provide cross-process locking, an atomic external-writer snapshot or host isolation.

## Evidence and disposition

Primary workflow outcome is **completed-and-independently-accepted out of two per
arm**; report artifact acceptance, observed checks and all four individual rows
separately. Failures consume their scheduled rows. A paused/partial cohort retains
unknown/unrun rows and cannot become an equal-cohort success ranking. No editable
test can redefine acceptance, and no answer wording proves that a patch works.

Record model-request counts, reported per-request/cumulative usage, unknown failures,
allowances/estimates/first-input identities, wall/loop timing scopes and every
operator intervention. Missing/truncated evidence is not zero usage or compliance.
Retain the raw transcript outside model context and inspect provider-boundary/tool
evidence; tool-record count alone includes rejected or undispatched calls.

Report requested batches and successful read batches separately. A requested read
batch has at least two distinct read-only calls in one assistant response from
`read_file`, `list_dir`, `grep_repo` or `tree`. Successful activation additionally
requires at least two non-interrupted, error-free corresponding tool results.
Calls whose arguments depend on an earlier result are not independent; record such
misuse. Do not count blocked calls, model prose or scripted probes as live activation.
If treatment does not activate, do not attribute timing/token variation to batching.
If control also batches, retain that evidence rather than crediting every batch to
the new sentence. A smaller request count alone is not coding success.

Two repeats, one selected task, counterbalanced fixed order and a single local
backend cannot establish a general/causal quality or efficiency gain. The original
ten-task 2/10 accepted and 1/10 completed-and-accepted, the selected feedback
comparison and G2–G5 remain unchanged. Do not ship the sentence merely because a
sample succeeds. No live inference or new comparison result is included here.

## Offline preparation evidence

Fresh T01 initial visible/external checks fail and its reference passes. All four
initial fixture/criterion baselines match. Both final pinned runtimes pass **1,342
offline tests, five live deselected**, Ruff, formatting (258 files), mypy (57 source
files), and both scripted 90/90 escalation matrices with contract labels/metric
limits inspected. These validate mechanics, not model quality.

Scripted shared-runtime and API probes use real reads, full `AGENTS.md` retention,
captures, budget observations and keyed replay in separate temporary stores.
All four API first identities match before fake-provider forwarding. Eight altered
path/allowance/schema/temperature cases forward nothing; matching inputs and owned
provider cleanup pass through. These are scripted probes, not a real CLI/model trial.

Raw preflight/G0 logs, recipes, owner marker and treatment bundle remain under the
recorded local trial directory; hashes are in the manifest. This is not a portable
full-source/trace kit. Preparation corrected a detached bundle command and an
AST-equivalent Python literal wrap, plus offline assertion fixtures; these are
logged separately. No scheduled state or model inference was created. Gemma stayed
unloaded. The existing Chroma deprecation warning remains.
