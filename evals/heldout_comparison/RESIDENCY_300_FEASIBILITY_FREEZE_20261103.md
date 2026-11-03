# Fresh 300-second residency feasibility freeze

Jorge requested the next three slices: preserve stop causes, prepare a fresh
small 300-second feasibility test, then run and document it. The stop fix merged
in #354. This is the second slice and selects exactly one new attempt. The actual
parent may issue its bound receipt only after this exact freeze is independently
reviewed, locally validated, CI-passing and merged, with fresh launch gates.
No actual parent receipt, candidate admission or generation has occurred yet.

## One planned attempt

Follow the [separate feasibility protocol](RESIDENCY_300_FEASIBILITY_PROTOCOL.md).
Only fresh original residency ordinal 2 is planned: ho_iso_week, harness, repeat 1,
keep-alive 300 seconds. Other captured fixtures are offline verification inputs,
not planned live attempts. This is known retired-v1 development material and
cannot establish a matched or held-out residency effect. The old stopped cohort
remains closed and untouched.

Use the same local Ollama 0.33.2/gemma4:12b digest
`4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`, original
compact/full prompt/tools and grading, context 16384, thinking off and temperature
zero. Keep 150 dispatch seconds, 32K total/4096 completion tokens, 12 tools,
6 iterations, request output 2048/min 256 and all other original settings. Setup,
grading, preservation and cleanup remain outside dispatch time. There is no
warmup, preliminary inference, control attempt, repair, substitution or retry.
No dependency/model installs or downloads, Docker, hosted/paid calls or user
Ollama service/configuration/identity management.

## Selected implementation and offline preparation

Retained root:
`/Users/jorge/.local/share/agent-harness/trials/residency-300-feasibility-6wx3aale`.
Clean detached runtime: `3e5caa9cdc60fd5fd6c6c9a955bfff960725656c`. This commit
includes the merged stop-cause fix and the one-cell implementation in this PR;
its eventual merge must contain that runtime ancestor. Review the implementation
and freeze together at the final exact head before any actual receipt.

The strict optional manifest field residency_feasibility_ordinal=2 requires the
original residency schedule and no global arm override. It changes manifest bytes
and invalidates old approval hashes. Preflight revalidates constructed values
before path inspection. The full-cohort entry point refuses this selection; the
one-cell entry point refuses absent selection. The provider refuses any other
ordinal before pin/ownership inspection. No admission callback or JSON field
independently authenticates the operator or authorizes inference.

A distinct one-cell ledger and start marker reuse the shared executor, tools,
original controls/grading, first-input/native-response guards, post-response owned
GET observation and settlement path. Controls still precede backend startup.
Results persist before cleanup; only a clean settled callback records the cell.
A stopped started cell can retain its result. No agent loop or provider adapter
is duplicated. All failure paths retain the existing closed-latch/stop policy.

Fresh candidate and rehearsal bundles use distinct independently verified APFS
clonefile model images and exact-path staged fixtures. The old unwithdrawn t8rosvgt
image is a read-only copy source. Existing sixteen-input mock production-adapter
capture/staging is reused to retain its tested preservation checks. Captured
consumed fixtures are archived before fresh fixtures are staged at the same roots.
Only ordinal 2 can become live work; the other 15 fixtures must remain unused.
The first normalized/wire request, Settings, command environment, selected
executable/dependency/runtime bytes, model image and catalog/criterion seals are
hash-bound in the concrete specification.

Every command uses the existing external interpreter
`/Users/jorge/Projects/agent-harness/.venv/bin/python` directly with -B, explicit
retained PYTHONPATH and recorded PATH. No resolver or source-local environment is
allowed. The standard packaging contract test copies only this project's wheel
into a disposable offline/no-deps target. Selected dependency files and package
versions do not establish a complete import/shared-library or host-state closure.
All observations are bounded/non-atomic and supply no process isolation.

## Mechanical evidence and remaining gates

One mandatory-mock scripted workflow completed and passed original acceptance
through the shared runtime, actual owned metadata lifecycle and independent
cell-specific image; no generation was forwarded. Only rehearsal ordinal 2 was
admitted and its other 15 fixtures stayed unused. Sixteen private refusal probes
passed, including invalid feasibility selection, global override and wrong executor,
plus original drift/receipt/withdrawal/occupied-port/caller-failure probes. A real
GET-only observation probe recorded not_loaded at selection 300 without generation.
All three metadata-owned groups settled cleanly without force and were observed
absent; model-copy receipts and independent inodes verify. These records establish
mechanics, not real residency, cache reuse or a working model patch.

Fifty-five focused manifest/scheduler/executor/feasibility tests passed; the final
controls-before-start assertion passed on a separate rerun. Fifty machine-recipe
contract tests pass. Retained G0 passed **2,938 offline tests, five live deselected,
one dependency warning**, Ruff/format/mypy and both 90/90 scripted matrices, with
labels and metric limits inspected. The actual copied recipe passed check-only;
missing execution arguments and missing mode refused before admission. No provider
or generation was created. The separate final PR suite passed **2,952 tests,
five live deselected, one dependency warning**, plus static checks and both 90/90
scripted matrices with labels/limits inspected. The
[machine specification](residency-300-feasibility-freeze-20261103.json) binds the
actual manifest, recipe, G0 and selected evidence. This proposal still requires
exact-head independent review, both CI checks, merge, actual parent receipt and
fresh launch gates.

## Parent-controlled execution and reporting

The machine recipe requires explicit mutually exclusive --check-only or --execute.
It selects the fixed candidate manifest and one-cell executor, uses zero transport
retries and disables proxy environment inheritance. Check-only inspects the clean
runtime/interpreter, candidate/withdrawal, artifact preflight, exact G0 and free
owned port without allocation or requests. Execute additionally requires the
actual parent-selected receipt/hash and exact reviewed/merged commits, checking
merge ancestry and recipe/spec/G0/manifest bindings. Local consistency does not
authenticate parent authority, review, CI, arguments or current resource ownership.

The actual trusted parent must verify Jorge's existing one-attempt authorization,
review/CI/merge and fresh source/artifact/resource gates before issuing its receipt.
Only owned 127.0.0.1:11447 may be started/stopped; preserve user 11434 and configuration.
Historical absence is not a lease. No synthetic rehearsal receipt grants authority.

One final outcome closes this attempt. Preserve original acceptance, native
completed-and-accepted status, tool errors/corrections, applied edits, visible
checks, interventions, usage known/unknown and owned loaded-model snapshots.
Cancelled generation remains unknown and follows its real cause; observation
errors/cancellation remain observation failures. Unknown submission/usage,
provider/identity/runtime/environment/criteria/preservation failure or failed/forced
cleanup stops without resubmission. No future inference follows from these files.

Native complete-response timing prefixes and loaded/not-loaded snapshots are
supporting observations, not continuous residency, GPU settlement, cache proof or
a causal comparison. Publish all logs and results even if unchanged or rejected.
No automatic default adoption, broader model-quality/held-out gain or no-go/
8-of-10/G2–G5 closure follows.
