# Fresh Gemma development run: timing diagnosis

This is read-only analysis of the five complete native responses from the
[closed development run](DEVELOPMENT_RESULTS_FIXES_20261103.md), not new inference,
a repaired attempt, a timing experiment or a performance remedy. The sixth request
was cancelled with no complete response, so all sums below are **prefixes only**.
The run remains one rejected artifact and fifteen unrun cells.

## Native reported duration fields

The table converts native integer duration fields to seconds by dividing by 1e9,
consistent with the [official Ollama API duration units](https://github.com/ollama/ollama/blob/main/docs/api.md?plain=1).
These are backend-reported measurements, not independently measured host spans.
They must not be added to client wall latency as if disjoint. Component sums can
differ from reported total duration; do not force exact accounting equality.

| Complete response | total_duration s | load_duration s | prompt_eval_duration s | eval_duration s |
|---|---:|---:|---:|---:|
| request-01.response.bin | 24.551 | 8.564 | 15.526 | 0.432 |
| request-02.response.bin | 20.964 | 2.127 | 17.250 | 1.577 |
| request-03.response.bin | 21.907 | 1.848 | 18.473 | 1.576 |
| request-04.response.bin | 24.536 | 1.853 | 20.085 | 2.588 |
| request-05.response.bin | 26.198 | 2.129 | 21.186 | 2.872 |
| Known prefix | 118.157 | 16.520 | 92.521 | 9.045 |

Prompt evaluation dominates these reported complete-response durations. There is
also a nonzero load duration in every complete response. The explicit owned backend
environment includes `OLLAMA_KEEP_ALIVE=0`; this configuration deserves investigation
before assuming another prompt/tool change will make the whole task fit the wall
budget. The observations do not establish that a nonzero keep-alive removes model
loads, improves prefix reuse, or produces a working patch. Changes to prompt/tool
request shape and backend state can also affect timings. No causal attribution,
throughput benchmark or total-run timing breakdown follows from five responses.

## Next bounded hypothesis

A separately selected development experiment could hold task, model, prompt/tool
configuration and original budgets fixed while varying only owned-backend model
residency. It would need an explicit supported residency setting in the selected
execution manifest, exact fresh first-input/resource pins, independent review,
fresh workspaces, ownership and final settlement verification, and separately
recorded launch approval. Do not silently change application defaults, existing
owned-backend defaults, frozen recipes or old evidence. A warmup request would be
extra inference and is not part of this proposal.

The primary outcome must remain independent artifact acceptance. Record native
completion, unknown usage, full timing prefixes and failures as supporting evidence.
Stop on uncertain submission or failed settlement; do not resubmit. This document
authorizes no new run, budget expansion, dependency/model download or management
of the user's Ollama service. The old v1 corpus remains retired from held-out claims.

## Reproduction without inference

Root: `/Users/jorge/.local/share/agent-harness/trials/gemma-fixes-dev-15yxnrks`.
Read only `candidate/run/cell-01/native/request-01.response.bin` through
`request-05.response.bin`; parse `total_duration`, `load_duration`,
`prompt_eval_duration`, `eval_duration` as integer fields, divide by 1e9 and sum
each column. The existing results evidence index binds all five native files.
The sixth request's failure record has no timing payload and is never treated as
zero. `candidate/run/cell-01/backend/environment.json` records the selected
keep-alive environment. No backend command or network request is needed.
