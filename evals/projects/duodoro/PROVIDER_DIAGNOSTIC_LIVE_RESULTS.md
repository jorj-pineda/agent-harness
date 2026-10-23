# Two-task provider diagnostic observations

After the corrected [fresh freeze](PROVIDER_DIAGNOSTIC_FRESH_FREEZE.md) was
independently reviewed and PR #183 merged, both scheduled attempts ran once through
the actual CLI at frozen runtime `a9d60720481703ad5ea0a54dc032f5ce6032f9c3`.
**0/2 accepted artifacts; 0/2 completed-and-accepted.** Both failed with provider
read timeouts before any edit. This diagnoses observation coverage, not a remedy,
causal model comparison or new supported envelope. Original cohorts, historical
setup failure, unattended no-go and G2–G5 remain unchanged.

[Structured evidence](../../results/2026-10-23-duodoro-provider-diagnostic.json)
records both run/submission IDs, allowlisted observations, frozen identities,
regrades, hashes, lifecycle and cleanup. No attempt replacement, model switch,
operator coding rescue, timeout/budget expansion or Duodoro working-checkout edit.

| Task | API lifecycle | Requests / responses / failures | Artifact acceptance | Failed-call HTTP duration | Known response prefix prompt / output |
|---|---|---|---|---|---|
| T05 timer preferences | failed, read_timeout | 5 / 4 / 1 | rejected, unchanged | 60,015.958 ms | 14,761 / 115 |
| T09 tag filter | failed, read_timeout | 5 / 4 / 1 | rejected, unchanged | 60,004.900 ms | 13,560 / 201 |

Both aggregate token totals are **unknown**, including the failed in-flight
request. Prefix counts are not totals, zero usage, billing evidence or proven
within-budget success. Final response envelopes are absent, so their completion/
check fields remain unrecorded; complete progress journals show no command check
or edit was dispatched. Do not fabricate a final response from partial progress.
The original 300-second dispatch allowance and all other approved limits stayed
fixed; API startup-to-source-settlement intervals were about 144.8 and 138.0
seconds, separate from HTTP-call timing, pure inference or hard return deadlines.

## What the new observations establish

All **10/10 response/failure events** retained observations: four normalized
responses and one duration-only failed request per task. Both saved prefixes are
untruncated. All eight responses report native prompt/output usage fields present,
allowlisted terminal reason `stop` and zero-length text. The first response per
task has zero tool calls and uses the existing bounded completion recovery; the
next three responses each propose one tool (tree, read, plan). Empty text alongside
a tool call is not itself a premature final answer or a model-quality score.

Each fifth request raises `read_timeout`, with approximately 60-second observed
HTTP post/status-check duration. Its native usage/terminal/content/tool-count fields
remain unknown, as intended for a failed request. The exact bounded failure
observation also survives as the separate saved provider-failure record. The new
metadata measures the boundary duration directly; it does not establish queue time,
generation progress, time to first byte, raw wire output or backend root cause.
No raw failed payload or backend logs were retained, and there is no proposed
write to reconstruct from a failure record.

T05's historical missing-usage symptom **did not reproduce**: its four returned
responses had present usage fields and its fifth request timed out. That does not
refute the earlier saved empty/unknown-usage observation, prove a common cause,
or justify disabling usage guards. T09 reproduces the read-timeout category,
not a byte-identical failed request or a causal comparison across workspace paths.

## Review, grading and preservation

Both initial first-request identities and saved profiles match the reviewed freeze.
Each task has exactly one durable run, no reversal and no replacement submission.
The actual CLI retained its printed key and retrieved the final saved JSON/human
observations. After owned API settlement/shutdown and model unload, review-only
restart returned identical JSON; review-phase HTTP logs contain GETs only and the
provider rejects inference. Historical failed run/session evidence is unchanged.

External checks reject both final unchanged artifacts. Regrading their unchanged
archived full copies reproduces rejection, with file hashes unchanged by regrade.
All protected files and modes, criteria/references/settings, frozen recipes and
preflight evidence match. Duodoro HEAD/status/refs/index fingerprints are identical.
The failed older preparation's manifest and empty/absent stores are also unchanged.

All four owned source/review APIs exited with complete shutdown logs; port 18757
is released. Each source model phase was unloaded before review/next task, and the
final loaded-model list is empty. The existing Ollama service responds and was
preserved. No dependency install, paid/additional-model call or Docker action.
No execution setup correction or manual repair occurred in this fresh cohort;
preparation's two reporting/audit corrections remain separately recorded.

Local immutable artifacts, saved records, logs, regrades and recipes remain under:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-provider-probe-fresh-_f7lehs6`

## Validation and disposition

The retained audit verifies frozen input/profile/first-request identities, one-run
counts, unchanged scored artifacts/regrades, all criterion/recipe/preflight hashes,
review equality and preservation. Final assertions verify all ten observations,
complete process cleanup, unloaded model and preserved historical failed freeze.
The structured report hashes 51 outcome/cleanup evidence files plus its recipes.
This docs/data report does not rerun preparation's 78 focused, 1,415 offline/static
or two 90/90 scripted matrices as model-quality evidence; those remain historical
mechanics checks at the exact execution runtime.

The diagnostic path now works through actual clients and persists the intended
bounded metadata, but the coding attempts still fail. Do not ship more features,
raise limits or repeat the cohort automatically. A further investigation must
separately freeze a specific backend/timing hypothesis or explicitly narrow the
proposed task envelope. These two public selected attempts do not close the
held-out/two-model or release gates; retain no-go and unknown backend cause.
