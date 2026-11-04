# Direct coding prompt experiment

Set CODING_PROMPT_PROFILE=direct to select the shared, opt-in direct profile.
It keeps the compact safety/edit/check contract, but encourages reading named
files directly and editing a small clear fix before spending a turn on a formal
plan. Full and whole_file toolsets have separate recorded prompt versions.

This targets the observed six-iteration development trace that stopped after
inspection, command recovery and emit_plan without an edit. It is a hypothesis,
not evidence of a working patch or a general model-quality improvement.

Tools remain available, including emit_plan and tree. Existing budgets,
require_plan_before_edit and verification rules still enforce their configured
policy; prompt advice cannot bypass them. Legacy remains the default and compact
is unchanged. Application, recovery/configuration review and evaluations select
through the shared prompt helper. The prior closed trials are never rewritten
or resumed. A fresh reviewed freeze and actual parent approval must precede a
new local inference attempt; no inference is authorized by this documentation.

The emit_plan tool description now also states that planning is optional unless
configured policy requires it. This corrects its previous unconditional
before-edit advice. Tool names, arguments and enforcement are unchanged; the
description changes the captured schema identity, so old execution profiles may
require explicit configuration review rather than silently continuing. This is
a consistency fix, not measured evidence of better model behavior.

## Optional planning-tool experiment

Set CODING_TOOLSET=no_plan to omit only emit_plan from the shared coding
registry. Full remains the default; whole_file retains its existing behavior.
The no_plan selection retains both edit tools, discovery, commands and memory.
REQUIRE_PLAN_BEFORE_EDIT=true is incompatible and refuses at settings validation.
Other edit, verification and budget policies remain enforced.

Application and real-task evaluations already forward the shared toolset selection.
Captured settings and actual schema hashes record it; prompt text/version remains
that of the selected prompt profile because prompt bytes are unchanged. Existing
configuration drift checks reject incompatible continuation without rewriting
historical records. The fixed retired experiment protocol still selects full.
This option supplies no fresh launch authority or evidence of coding benefit.
