# Held-out v1 local comparison protocol — unrun

This fixed protocol prepares one local same-model comparison. It authorizes no
inference. Validate offline with `uv run python -m evals.heldout_comparison`.
The validator bounds the plan, verifies the registered catalog seal and exact
ordered request/category identities, checks the fixed modes/budgets and validates
application Settings without loading a .env file. It constructs no provider.
Ambient configuration outside the listed overrides is not an execution freeze.

Use all four original held-out v1 tasks in manifest order, two repeats each, with
both shared-runtime `harness` (compact/full) and `minimal` prompts. Preserve the
existing alternating mode order from the real-task runner. This yields sixteen
fresh disposable attempts, eight per mode. No source sharing, manual patch repair,
extra attempts, task substitution, skipped repeats after success or budget transfer.
Question answers use the declared structured-fact rubric and inspection-only
policy; coding artifacts use the original external acceptance. Visible checks,
prose, filename overlap and model judges do not redefine independent acceptance.

Only local Ollama gemma4:12b is proposed, with the exact digest and expected version
in plan.json. These are historical candidate pins, not current service evidence.
Fresh version/image/endpoint verification is required at freeze/launch. No hosted
or paid calls, downloads, installs, Docker, new models or warmup generation.
Preserve the user's existing Ollama service/configuration; only evaluation-owned
resources may be managed after inspecting current processes and listeners.

Per cell: 150 dispatch seconds, 32,000 total tokens, 4,096 output tokens, twelve
tools and six iterations. Request output cap/minimum are 2,048/256, temperature
zero, thinking off, explicit context 16,384, ratio 3, one completion correction,
identical-call limit two, file budget two, root instructions off. Visible coding
check: python -m pytest -q test_visible.py. Questions require no command. The
request timeout is 180 seconds; dispatch limits do not bound setup, storage,
grading, cancellation cleanup or remote generation settlement. Totals per mode:
1,200 dispatch seconds, 256K total tokens, 32,768 output tokens, 96 tools and 48
iterations. Physical totals double these; they are allowances, not pooled budgets.

Report artifact acceptance and structured-answer acceptance separately, plus native
completed-and-accepted counts per mode/task/category/repeat. Keep raw usage unknown
when unavailable. Include every planned cell; failed, blocked, no-result and unrun
cells cannot become retry slots. Ordinary independently rejected artifacts and
native budget exhaustion may proceed only with intact preservation and complete
settlement. Provider/transport error, missing usage after submission, runtime,
environment, model identity, criterion or preservation failure stops the cohort.
Do not resubmit an uncertain request. The generic CLI's returned-error policy is
not sufficient for this protocol; enforcement belongs in the next rehearsal slice.

Before generation: merge an independently reviewed scripted shared-runtime rehearsal,
then an exact execution freeze passing local gates, independent exact-head review
and CI. The freeze must pin the actual clean runtime, full resolved settings,
normalized/native first inputs, model image, source/criteria seals, command recipe,
report locations and resource ownership, with refusal/cleanup probes. It must have
a parent approval receipt and fresh launch gates. Neither this validator nor the
generic real-task CLI implements launch approval. No launch path is added here.

This is one manually selected public four-task set and one intended model. Two
repeats may have little sampling variation at temperature zero. Report raw counts
and paired variation without a general causal, representative or two-model gain.
Author/review exposure and the tuning boundary in heldout_tasks/README.md remain
explicit. Do not tune prompts/tools on v1 outcomes and retain a held-out claim;
retire the version for tuning and preserve old bytes/results. Original no-go,
8/10 target and G2–G5 remain open. No release approval follows.
