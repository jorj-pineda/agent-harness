# Provider-timeout comparison: actual execution freeze

**Prepared, unrun; independent review and merge required before inference.**
The [protocol](PROVIDER_WAIT_PROTOCOL.md) is unchanged. The machine-readable
[freeze](provider-wait-freeze.json) pins runtime
`6721cc9997a00a473cee39508aba8da169eb2758`, 273 source hashes, 17 local recipes
and 228 local evidence hashes. Recipes and private inputs remain at
`/Users/jorge/.local/share/agent-harness/trials/duodoro-provider-wait-ph07ytu3`;
the published manifest contains hashes, not raw model payloads.

## Fixed inputs and launch boundary

Exactly four original T09 `tag_filter` actual CLI submissions are prepared:
**C1:60, W1:180, W2:180, C2:60** seconds for the provider timeout scalar.
All four HTTP timeout phases change together. Each task retains its 300-second
dispatch allowance, 64K total/8,192 output tokens, 24 tools, 12 iterations,
16,384 context, 2,048 request output/256 minimum, compact/whole_file, thinking off,
temperature zero, checkpoints and required configured verification. CLI response
inactivity stays 600 seconds. Cleanup and final review can exceed dispatch.

Each cell has independently validated original initial-fail/reference-pass criteria,
fresh prepared bytes and isolated state/input paths. All use the same literal owned
`workspace` path, starting with the same full bytes and modes. First normalized
and exact wire inputs match; profiles differ only in the timeout scalar. Guards
check these inputs before forwarding. Source session/input stores and outcomes are
absent; Gemma is unloaded. No task was submitted to a model during preparation.

The launcher requires PR #193 merged and the local manifest byte-identical to the
committed freeze. It rechecks runtime, source/recipe/evidence hashes, environment,
model identity, prepared artifacts, root identity and absent stores/outcomes.
Existing outcomes prevent replay. A cell gets one CLI submission and its printed
key; uncertain delivery allows GET inspection only.

Before replacing the owned common workspace, staging requires the previous full
archive, unchanged grade/regrade artifact, identical saved review after restart,
complete input audit and current full-tree/root-identity agreement. Unexpected
edits or missing evidence block staging. Duodoro's working checkout is not staged,
reset, committed or pushed. Each outcome is archived separately.

## Completed preparation checks

- Exact-runtime G0: 116 focused and 1,453 offline tests passed; five live tests
  deselected. Ruff, format and mypy passed. Both scripted matrices passed 90/90;
  contract labels and metric limits were inspected.
- Four actual CLI scripted flows exercised read, plan, reference write, configured
  check and a synthetic fifth-call timeout. Each artifact was independently graded
  and regraded. Default source and guarded GET-only restarts preserved saved JSON,
  private input hashes and profiles; actual phase timeout construction matched
  60/180. All twelve owned API processes shut down and released the port.
- Guards blocked 16 normalized and 12 wire mismatches, and rejected 24 malformed
  capture/scalar cases before resources opened. Four unsafe staging cases preserved
  the workspace. Missing copied input evidence remained incomplete and blocked the
  next cell; the aggregate recipe validated four scripted records and eight
  source/review cleanup records. These are harness checks, not inference results.

## Observation and limits

The same evaluation-only recorder is used in both arms: twelve requests maximum,
1 MiB per file and 8 MiB combined, private plaintext storage. It records normalized
and wire inputs, not backend receipt or response chunks. Recording failure is
retained; incomplete input analysis stops remaining cells without replacements.
Owned APIs must settle and stop; evaluation Gemma is unloaded without stopping the
existing service. Restart review uses GET-only guarded mode.

Report artifact acceptance separately from completed-and-accepted, observed checks,
failure status, known prefix usage and unknown in-flight usage. No repair, extra
turn, failed-input replay or reactive escalation is allowed. Two repeats per arm
on one public task cannot establish general or causal model quality. No production
default, supported envelope, release target, original cohort or G2–G5 status changes.
