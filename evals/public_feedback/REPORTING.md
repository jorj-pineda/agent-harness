# Descriptive paired feedback reporting

Feedback executor reports now include `feedback_comparison` alongside unchanged
raw results, the existing summary and the eight-cell ledger. Other report types
keep their existing shape. The new summary labels scripted versus real-tool rows,
shows counts for control/bool_check, and retains all four paired repeats with their
original ordinal order inside each arm selection.

Accepted and completed-and-accepted counts are separate. Each cell also retains
its completion/verification/termination state, recorded row latency, token fields
and visible project-check exit codes. A visible pass or verified state never
substitutes for recorded independent artifact acceptance. These are descriptive
recorded outcomes, not fresh regrades or a new acceptance policy.

A started cell may have a durable result before cleanup. Its outcome appears with
`ledger_status: started`; it does not certify clean settlement. Missing/unrun cells
retain null outcomes and are counted as missing results, not failed attempts. Full
planned per-arm usage totals stay null unless all four result rows have known
prompt/completion usage. Separately named known-result totals are partial, not
estimates of missing usage. Recorded latency is not hard end-to-end timing.

The helper checks ordered distinct ordinals, paired arm/repeat identities, the
recorded/started/unrun prefix and row identity/count correspondence. Preserved
plan/ordinal metadata prevents swapping two arms that share task/mode/repeat. It reads no
source fixtures, checks or external artifacts while summarizing: report persistence
must remain possible when an input guard has already stopped execution. It never
resubmits cells, changes results or runs a grader. Invalid correspondence refuses
before replacing an existing report.

`benefit_assessment` remains `not_assessed`, independent regrades `not_supplied`,
and manual rescues/new regressions unknown. Even favorable counts require preserved
artifact regrades and intervention/regression review under the original protocol
before describing a promising public follow-up. Scripted successes do not measure
model quality; no causal, held-out, default-adoption or release claim follows.

## Evidence

Synthetic aggregation tests cover accepted versus completed outcomes, visible-pass
separation, all paired ordinals, stopped cells with durable started results,
missing cells, unknown full-cohort usage, malformed snapshot refusal, input-free
aggregation and atomic report preservation. Existing owned execution cases check
compatibility of the new summary with scheduler persistence.

A read-only aggregation of the retained eight-row native-mock report reproduced
4/4 accepted and completed-and-accepted in each arm, with all four pairs present,
scripted labeling and benefit assessment unchanged at `not_assessed`. It wrote a
new temporary report and verified the original retained report hash stayed
`42f2f6f4ee82f6d06032deb7a0ea907764fecaca0f5f5e1bb34de649a2b2f51f`.
No inference, backend, admission or regrade was performed by this aggregation.

Next: reusable mandatory-mock owned rehearsal, fresh retained candidate/rehearsal
and refusal probes, then a concrete reviewed freeze/operator recipe. Actual approval
and fresh launch gates still precede live inference. Release remains no-go.
