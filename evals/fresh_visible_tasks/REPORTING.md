# Descriptive fresh task and paired reporting

Fresh cohort reports now add `fresh_comparison` without changing raw results or
ledger states. The pure summary checks the fixed v1 twelve-cell schedule and
binds result task/mode/repeat, selection hash and ordinal to a recorded prefix plus
at most one started cell. It reads no source inputs and performs no regrading.

The summary retains six weak/stronger pairs across three tasks and two repeats,
with per-task/arm and aggregate accepted and completed-and-accepted counts. Visible
check exits, verification, completion and original acceptance remain separate.
Started results remain started even if they passed; unrun cells have null outcomes.
Incomplete or unknown usage keeps full planned totals null, while explicitly named
partial known-result totals remain available. Task-level unknown usage does not
invalidate another task's known totals. Row latency is not end-to-end timing.

This is descriptive reporting, not a benefit decision. Independent copied grades,
intervention/regression review and authentic authority are not supplied. Benefit
stays `not_assessed`; no default, held-out or release conclusion follows. Scripted
reports remain labeled as mechanics. An invalid snapshot is refused before atomic
report replacement, preserving the previous report bytes.
