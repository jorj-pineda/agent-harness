# Required independent refactor review

An automated behavior-check exit code of zero is provisional for this task.
Full artifact acceptance also requires a recorded independent source review:

1. Identify the exported boundedDuration implementation in the final artifact.
2. Confirm both parseStartSession duration fields route through that same helper.
3. Confirm the two inline coercion/default/clamp implementations were removed.
4. Confirm fractions, Symbol errors, defaults/clamps, modes, session IDs and other
   parsers retain their stated behavior; executable checks are supporting evidence.
5. Confirm no unrelated edits/test weakening; record reviewer, verdict, criterion
   hash, final diff hash and explanation in the external trial ledger.

An unused but behaviorally correct exported helper fails this review. Do not use
filename overlap, exact patch matching, a sole model judge, or task.passed from a
behavior-only evaluator as a substitute. Pending/unsupported review is unknown,
not accepted. The reviewed reference extracts one shared helper and uses it for
both fields; that review does not pre-approve any later model patch.
