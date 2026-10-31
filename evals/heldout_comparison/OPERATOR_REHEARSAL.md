# Named-workspace native-adapter operator rehearsal

This CLI can use only an in-memory mock transport. It cannot start a service,
select a live provider or authorize generation. Prepare a **new** no-forward
capture and stage it as described in STAGING.md, then run:

```sh
uv run python -m evals.heldout_operator_rehearsal \
  --staged-directory /absolute/path/to/original-capture-path \
  --capture-directory /absolute/path/to/archive/captured \
  --index-sha256 SELECTED_INDEX_SHA256 \
  --directory /absolute/path/outside/git/new-operator-evidence
```

The capture must record the same actual clean Git revision and validated Settings
pin as this runtime. Do not change source files between capture and rehearsal. The selected hash binds
bytes, not approver authority. Preflight verifies the archived bundle, staged
receipt/root inventory and all sixteen fresh fixture copies before allocation.
Output must be absent, outside Git workspaces and disjoint from the selected
capture and staged root. Existing and partial directories remain preserved.

The fixed original plan supplies task/mode/repeat order and budgets. Controls run
before the first cell. The full cohort ledger is saved before provider work and
each cell. Each cell owns a new single-use prepared workspace, session and provider.
The production Ollama adapter is wrapped by the normalized/native first-input
guard, then the protocol model/usage wrapper. The transport can only return
canned reference edits, visible checks and answers using the original scripted
oracle; native payloads traverse the production response decoder. Canned usage
is explicitly scripted, not measured model usage. All real tools, budgets,
configured commands, preserved grading inputs and original external acceptance
remain shared with the application runtime.

The first normalized and wire inputs must exactly match the archived pins before
reaching the mock oracle. Exact initial paths are preserved; nothing is masked.
The complete classified Settings pin is compared before allocation, before each
cell admission and before every delegate request. A detected mismatch latches
closed; restoring values does not permit a retry. After each cell, observed
runtime_config must also equal its capture record. Settings equality does not
freeze command environment, executable/dependency bytes, resources or host state.
Runtime Git identity is checked before each cell and after the last; the archive
is reverified before finishing. Native request/response files are private, bounded
and exclusive; output/input overlap is checked before
evidence writes. Later native tool-call IDs may vary, so only the first request
is bound by the original capture. No history equality claim follows.

`--fault missing_usage`, `--fault wrong_model` and `--fault transport` fail the
first mock response before tool dispatch and retain a stopped report with fifteen
unrun cells. An ordinary artifact rejection may proceed only under the existing
protocol result rules. First-input mismatch, runtime/configuration drift, missing
usage, unexpected model/termination, grading/preservation failure, reporting or
provider-close error stops further cells. No failure creates a retry slot.
Recorded rows stay linked to ledger cells; started work without a row becomes
no_result, with no synthesized acceptance or usage. Providers close on every
post-construction exit; a primary failure is retained separately from cleanup
failure. All admitted workspaces remain consumed on success or failure.

Reports use scripted-real-tools-rehearsal labels; operator.json additionally
identifies scripted-native-adapter-real-tools, scripted usage, zero forwarding
and no inference/approval. Report replacement retains the existing bounded
persistence limits. Missing final evidence means incomplete preservation, not a
passing rehearsal. No automatic reset, rollback, resume or cleanup of evidence.
The archived capture remains unchanged; the now-admitted staged roots cannot
be used for a live attempt. Prepare a separately reviewed fresh capture/staging
sequence for any future launch; never remove admission markers to reuse work.

This combines production adapter serialization, first-input enforcement, actual
tools and original grading in process. It does not exercise a real HTTP socket,
application HTTP/CLI transport, model generation or remote backend settlement.
The separate native metadata-only backend rehearsal does not close those gaps.
Neither rehearsal provides isolation from host code/external writers or authority
to execute a comparison. The full live operator/refusal/settlement recipe,
complete runtime/environment/resource pins, independently reviewed exact execution
freeze, parent approval and fresh launch gates remain incomplete. Original no-go,
held-out model quality and release gates remain open. Do not tune on v1 outcomes
and retain a held-out claim.

New captures retain settings.json with a hash in index.json. Historical indexes
without that optional hash remain readable for preservation; this operator
refuses them before provider allocation/admission. Do not edit old indexes or
insert a newly generated pin to make historical evidence admissible. Generate a
new capture at a new path. See SETTINGS.md for the bounded Settings policy.

An explicit --command-path capture can additionally bind the effective command
environment. Supply the same selection to this mock operator; omission or a
different mapping refuses when a pin is present. See COMMAND_ENVIRONMENT.md.
Captures without the optional pin retain unfrozen mock behavior, never launch
readiness. Do not retrofit historical evidence.
