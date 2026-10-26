# Explicit recovery-context preview

`harness.recovery_context.build_recovery_context` joins a saved recovery review
with deliberately selected historical source spans. Retained Beam attempts stopped
at context/total-token gates even when their artifacts passed acceptance; assembling
reviewable smaller evidence is a prerequisite for a later recovery experiment.
This slice supplies deterministic assembly only. No provider history is replaced,
no system prompt is generated and no continuation is dispatched.

The selection pins the complete validated run-record serialization SHA-256 (from
`recovery-review.source_record_sha256`) and zero to four explicit checkpoint-span
selections. Each names a canonical relative path, whole-file hash and inclusive
line range under the existing [span limits](CHECKPOINT_SPANS.md). Same-file ranges
must be disjoint and use one hash. Input order is retained, including disjoint
ranges supplied out of order; nothing is automatically ranked, merged or selected.
Zero spans supports review of legacy or failed work without a final checkpoint.
It does not imply that source inspection is unnecessary.

Only a terminal run's linked `turn_after` checkpoint can supply source. Its session,
run ID and paired-before ID must match the saved run before any blob is read.
A manual, before-turn, foreign or unrelated checkpoint cannot be substituted.
The saved-record hash is checked before checkpoint access. Callers must enforce
run ownership before invoking this helper. Checkpoints are immutable in ordinary
application operations; this does not protect against an operator rewriting SQLite.

The output embeds the complete existing bounded recovery-review packet and exact
selected spans, retaining their hashes, original status, missing-final unknowns,
truncation, omission counts and historical-source scope. A combined evidence hash
covers compact review JSON, a newline, and compact span JSON joined by newlines,
in selection order. It identifies this representation, not current state or truth.
Total selected source is at most 16,384 UTF-8 bytes and the complete compact output
at most 65,536 JSON bytes, including escaping/metadata. Any failed range, encoding,
identity, corruption or aggregate bound refuses the whole bundle without returning
a partial selection or silently dropping evidence. Rendering whitespace can enlarge
separately displayed output. Original source and run records remain unchanged.

This is untrusted, potentially sensitive owner evidence. All unselected files and
lines remain outside the preview; a selected span does not certify that instructions
or a whole file were read. Existing diffs are bounded per-turn review evidence,
not a reconstructed current patch. The bundle does not inspect current files,
load instructions/facts, estimate a prospective provider request, validate selected
configuration, alter/reset budgets or grant permission to execute. It is explicitly
`read_only=true`, `execution_supported=false`. Context fit, preservation of required
instructions and guarded execution need a separate design and validation. No
measured quality/recovery gain, release approval, default change or closure of the
broader compaction milestone is claimed. API and CLI access follow separately.

Fifty focused assembly/recovery/span regressions and Ruff/format/mypy pass.
Coverage includes exact saved source after restart without the original workspace,
failed checks and incomplete state, missing-final unknowns, pre-blob identity guards,
overlapping/excess selections, aggregate source/escaped JSON limits and all-or-error
selection. The initial test fixture incorrectly used manual checkpoint storage for
turn checkpoints; it was corrected to exercise real atomic run binding. The full
offline gate is required before merge; no live model result is claimed.
