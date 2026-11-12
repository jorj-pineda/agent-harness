# Offline feedback review join

`review_feedback_run` joins three explicitly selected bounded byte strings: the
original report, copied regrade summary and a `FeedbackReview` record. Each requires
an expected SHA-256. The function reads no paths, runs no grader, constructs no
provider and writes nothing. Callers retain its output separately with their own
preservation policy; it never changes original outcomes or issues authority.

Strict review records bind the manifest/report/regrade digests, caller-supplied
reviewer attribution and all eight fixed cells. Each cell binds its artifact pin,
coverage, evidence references/digests and notes, plus explicit manual-rescue,
test-weakening, tool-misuse, other-regression and owned-settlement observations.
Unknown values must be supplied explicitly. References are caller assertions;
the join does not open them, verify their contents or authenticate the reviewer.
Use the [review protocol](REVIEW_PROTOCOL.md) to collect actual observations.

The join checks fixed ledger/result/plan correspondence, matching evidence labels,
original outcomes and copied artifact pins. Regrade disagreements and paired
control-pass/treatment-fail acceptance regressions remain visible. Missing pairs
stay unknown. Started results never certify settlement, and absent execution cannot
receive a clean task review. Complete coverage requires references; unknown usage
or review dimensions block eligibility. Review assertions remain unauthenticated
even when every consistency check passes.

The result includes the existing descriptive outcomes alongside four copied-grade
pairs and unknown/disagreement ordinals. `public_followup_rule_eligible_for_human_review`
only indicates that the selected records satisfy the protocol's limited rule:
finished live-labelled cohort, greater original completed-and-accepted treatment
count, matching copied acceptance, known usage/coverage and clean review dimensions,
with no paired regression. It is not a benefit decision. Scripted records, ties,
missing or started outcomes, rescue, weakening, misuse, other regressions, forced
cleanup, evidence gaps or regrade disagreement cannot qualify. `benefit_assessment`
stays `not_assessed`; release remains no-go. Input labels cannot authenticate that
inference occurred. Actual approval/fresh launch gates and genuine human evidence
review remain external prerequisites; this helper supplies none of them.

Focused synthetic tests exercise the limited rule, scripted separation, all review
blocks, stopped/started/missing cells, regrade regressions/disagreements, ties,
unknown usage and mismatched hashes/identity/artifact/outcomes. These validate join
mechanics, not inspection quality, model performance or complete host attribution.
A retained scripted exercise is the next slice; no live execution is authorized.
