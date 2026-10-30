# Scripted shared-runtime rehearsal

Run `uv run python -m evals.heldout_rehearsal --report /tmp/unique-rehearsal.json`.
This CLI can construct only a canned provider. It makes no HTTP/model request and
has no live-provider argument. Its report mode is `scripted-real-tools-rehearsal`;
success counts establish mechanics only. Reference solutions and canned usage
are explicit simulation overrides, not model responses or measured model usage.

All sixteen fixed cells use real disposable fixture copies, the shared application
runtime, file tools, configured visible commands and original external graders.
Coding scripts write references, check and answer; question scripts read and
return the reference facts under inspection-only policy. Each cell owns a fresh
provider, session and workspace. Mode order, request/category identities, seal and
fixed budgets come from the reviewed protocol. Standard application environment
variables are bypassed with a rehearsal-specific settings prefix; .env is disabled.
Actual execution will still need a frozen fully resolved configuration.

The evaluation provider wrapper latches transport exceptions, wrong response model
and missing normalized usage before returning tool-bearing responses. No later
request can pass that failed wrapper. `--fault missing_usage`, `--fault wrong_model`
and `--fault transport` exercise first-cell refusals, preserving one failed result,
fifteen unrun cells and provider closure. Result-boundary policy additionally
refuses unknown criteria/preservation/usage, model drift and unexpected terminal
states. Ordinary artifact rejection may continue; native completion stays separate.

Report preconditions, private replacement and the bounded cohort ledger are reused.
Exceptions stop forward work, close the owned provider and retain the plan. This
scripted implementation does not prove backend settlement, cancellation of remote
generation, native production wire inputs, client transport behavior or real model
identity. It is an in-process shared-runtime rehearsal, not a CLI-over-HTTP test.
Host code remains trusted and path checks/non-atomic preservation are not isolation.

There is no launch authorization here. The exact execution freeze, production
adapter input capture, current owned model/environment pins, resource ownership,
refusal/cleanup recipe, independent review/CI/merge, parent approval receipt and
fresh launch gates remain outstanding. No prompt/tool tuning or quality/release
claim follows from a scripted reference artifact passing its original criterion.
