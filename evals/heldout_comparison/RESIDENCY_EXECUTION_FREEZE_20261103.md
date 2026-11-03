# Residency development execution freeze

Jorge requested the residency experiment and asked to continue until it finishes.
That authorization covers this fixed local development comparison. The trusted
parent will issue its selected execution receipt only after exact-head independent
review, local gates, both CI checks and merge, followed by fresh launch checks.
This proposal grants no generation authority. No inference or actual parent
receipt has occurred in this candidate.

## Scope and outcomes

Follow [the residency protocol](RESIDENCY_DEVELOPMENT_PROTOCOL.md) and
[fixed schedule](residency-plan.json): sixteen harness-only cells on four retired
v1 tasks, two repeats, keep-alive zero/300 seconds then 300/zero per task.
These are known development tasks, not held-out evidence. Source, criteria,
visible checks, runtime prompt/tools, sampling and budgets stay fixed. Each cell
gets fresh named fixtures, an independent APFS clonefile model image and its own
owned backend lifecycle with empty loaded-model startup metadata. Only residency
selection differs; there is no warmup or extra inference.

Use local Ollama 0.33.2, gemma4:12b digest
`4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`.
Each cell retains 150 dispatch seconds, 32K total/4096 completion tokens,
twelve tools, six iterations, request cap 2048/min 256, context 16384, thinking off
and temperature zero. Full settings are pinned in the machine specification.
The forty-minute dispatch allowance excludes setup, grading, storage and cleanup;
there is no hard end-to-end deadline.

Report artifact and structured-answer acceptance separately, completed-and-accepted
counts, raw task/repeat/arm outcomes, unknown usage, tool errors, edits, checks and
interventions. Native timing prefixes and GET-only loaded-model snapshots support
diagnosis; neither proves continuous residency, effective caching or causal gain.
No default adoption, quality claim or no-go/8-of-10/G2–G5 closure follows.

## Retained runtime and preparation

Root: `/Users/jorge/.local/share/agent-harness/trials/residency-dev-2dan9i1e`.
Runtime: clean detached `06502c680f0702195f1aa401f9dfc3e1074ebcf9` (through #351).
Use the existing external interpreter
`/Users/jorge/Projects/agent-harness/.venv/bin/python` directly with `-B`, retained
PYTHONPATH and the recorded PATH. Never invoke a resolver from retained sources.
No dependency/model download or install, hosted/paid call, Docker, user-service
management or closed/withdrawn-cohort reuse is permitted. The standard packaging
contract test copies this project's wheel into a disposable offline/no-deps target.

Private `prepare.py` and `preparation_api.py` adapt the existing offline preparation
helper to residency capture/staging. They are selected by hash for independent
review, not installed product code. They retain interpreter/source-local-environment
refusal, clean revision checks, exact runtime/dependency selection, independently
verified image copies, mandatory mock capture, staging and preflight. Candidate
and rehearsal use distinct image/cell bundles. The old unwithdrawn t8rosvgt image
is a read-only clone source. Captured consumed fixtures are archived before fresh
fixtures are staged at those exact paths; candidate/run remains absent.

Sixteen normalized/wire first inputs, Settings, command environment, selected
runtime/executable/dependency files, images and original catalog/criteria seals are
hash-bound. The eight within-task/repeat pairs match after replacing only each
exact workspace-root string with `<workspace>` and omitting the derived normalized
identity hash for comparison. Raw originals are unchanged and retained. Selected
dependency files and distribution versions do not establish a complete import,
shared-library or host-state closure; observations are bounded and non-atomic.

## Rehearsal and gates

Mandatory-mock production-adapter rehearsal completed sixteen scripted accepted
workflows through shared tools/runtime/grading. Sixteen actual owned metadata
sessions used separate verified model copies and settled cleanly without force;
no generation reached Ollama. Thirteen refusal probes passed, including drift,
invalid receipt, withdrawal, occupied port and caller-failure cleanup. A separate
real owned GET-only observer probe recorded not_loaded with selection 300 and clean
unforced settlement. Read-only audit found all eighteen historical groups absent,
verified independent image inodes/receipts and absent candidate admission.
The audit checks historical receipts and file identity; it does not rehash image
bytes, reserve ports or prove continuing ownership. Startup helpers independently
rehash selected images before actual use.

The T3 restart interrupted the initial retained G0 around 24%; its partial log is
preserved in g0-interrupted-1 and is not a passing gate. The complete retained G0 rerun passed **2,901 tests, five live deselected, one
warning**, plus Ruff, formatting, mypy and both 90/90 scripted matrices. Labels
and metric limits were inspected. The actual copied recipe passed check-only;
--execute without parent arguments and invocation without an explicit mode both
refused before admission. No provider or generation was created. The
[machine specification](residency-execution-freeze-20261103.json) binds the final
G0, recipe, private preparation/rehearsal/probe scripts and selected evidence.
This proposal still cannot launch before review, CI, merge and parent receipt.

## Parent-controlled launch and stop rules

The [machine recipe](operator_residency_20261103.py) has required, mutually
exclusive --check-only and --execute modes. It connects selected admission to the
dedicated residency executor, with zero transport retries and proxy inheritance
disabled. It checks retained source/interpreter, original candidate bytes,
withdrawal, preflight and exact G0 before launch; execute additionally binds the
actual parent receipt to exact reviewed/merged freeze commits, recipe and G0.
Git/hash consistency does not authenticate the parent, review, CI or argument
source. Only the actual trusted parent may issue the receipt using verified facts
and Jorge's existing bounded authorization. Synthetic rehearsal fixtures are not
actual approval. No receipt generator or implicit execution mode is included.

Before launch, recheck source, artifacts, fixtures, model and resource state.
Only owned 127.0.0.1:11447 may be managed; preserve user 11434, configuration and
identity. Historical absence is not a lease. Source/pin/withdrawal, exact first-input,
native model/done/usage and owned transport guards remain enforced around dispatch.
After each complete response, real GET-only loaded-model observations precede tool
dispatch. Cancelled requests have unknown usage/state, not inferred zeros.

Provider, transport, missing-usage, identity, environment, criteria, preservation
or failed/forced settlement stops both arms and retains all unrun cells. An ordinary
rejected artifact or native budget stop may continue only with known usage, intact
pins and clean settlement. Never retry uncertain submission, reset, substitute,
repair, enlarge budgets or transfer allowances. Persist results before cleanup;
record the ledger cell only after settlement. A stopped started cell can retain
its final ordinal result. Retain raw logs and EXECUTION_LOG.md even if no product
change results from this experiment.
