# Held-out task preparation

`v1/tasks.yaml` is a separate four-task catalog covering an ISO-week bugfix,
rolling-window feature, immutable-record refactor and cache-policy question.
The default public development corpus remains twenty tasks. Select this catalog
explicitly for offline controls:

```sh
uv run python -m evals.real_run --task-manifest evals/heldout_tasks/v1/tasks.yaml --validate-only
```

This is preparation, not a launch recipe. No product-model inference has used these
tasks. Preservation enforcement is the next PR; a later fixed comparison still
needs exact runtime/model/input/budget pins, independent review, CI, merge and fresh
approval/launch gates. This document authorizes no inference or additional spend.

The holdout boundary is relative to harness prompt/tool tuning at preparation base
`39967b472e337ac8422dbaf54eb4f22588e5848e`. This batch changes no shared prompt,
tool schema or model profile. Task authoring/reference validation, deliberately
incorrect artifacts, scripted shared-runtime controls and independent development
review are permitted preparation checks; they are not product-model attempts.
Once version 1 is merged, do not use its failures to tune prompts/tools and then
report the same version as held out. Retire it for tuning and define a new version
before another held-out claim. Corrected criteria require an explicit new version
with the old bytes/results retained; completed cells are not retry slots.

This is a public, manually selected set, not secret material, an independently
sampled benchmark or proof of pretraining non-exposure. Author/review models have
seen it and cannot be claimed unexposed on these tasks. Four small tasks do not
establish a representative supported envelope, the original 8/10 target, repeated
two-model improvement or release readiness. Keep answer acceptance separate from
patch acceptance; the question checks only its declared structured facts.

Each coding fixture fails its visible/external initial controls; its reference
passes both. Selected incorrect variants fix January only, emit only the first
rolling mean, or strip host spaces: all pass their visible example and fail the
external criterion. An independent executable cache oracle validates reference
facts, and a false-value cache-miss answer rejects. All controls use trusted
disposable copies and the shared application runtime; no patch matching or model
judge is used. Host permissions, scoped/non-atomic workspace checks and legacy
evidence limitations remain as documented in the development corpus.

The catalog label and manifest hash identify declared inputs, not preservation or
absence of contamination. The next preservation slice must refuse missing/changed
bytes or modes before evaluation allocation, bind the pin index itself and avoid a
header edit bypass. No live result or preservation gate is claimed here. Existing
development tasks, closed trial criteria/results and release no-go/G2–G5 remain.
