# Go access and replacement-model preparation

Observed 2026-10-25; no inference or coding submission ran.

After Jorge refreshed OpenCode Go, an authenticated read-only
`GET /inference/go/v1/models` returned HTTP 200, where earlier checks returned
401. This records a changed metadata response, not verified model dispatch,
remaining subscription allowance, billing or the cause of the earlier 401.
The Go-specific model-list route has not been established here as a documented
authentication test; do not ask for another credential reset solely from it.
No credential, header value or response body is published. Existing credentials
were read privately and no account, billing or subscription settings were changed.

`opencode models opencode-go` lists `beam-free` but not the planned
`longcat-2.5-preview-free`. The documented public
[model catalog](https://opencode.ai/v2/docs/console/api/inference/)
returned 200 and did not list the old LongCat ID either. The CLI catalog is
metadata, not an authenticated generation or account-entitlement test.

Jorge requested the cheapest available model within his existing Go subscription.
The replacement candidate is **Beam Free (`beam-free`)**, listed at zero input,
output and cache-read cost in the current
[Go documentation](https://opencode.ai/v2/docs/console/go/). It is limited-time and
region-limited. Installed CLI metadata lists tool calling, reasoning and the
compatible API base `https://opencode.ai/inference/go/openai/v1`. Metadata cannot
attest actual tool behavior, served model identity or reasoning-history compatibility.
Those remain preflight/execution questions, not a reason to silently discard fields.
No paid fallback, balance spend, subscription upgrade or different model is allowed.
Unknown pricing/access/allowance blocks dispatch; declared free is not observed billing.

The reviewed [LongCat plan](intended-model-plan.json) and
[protocol](INTENDED_MODEL_PROTOCOL.md) remain unchanged historical preparation,
with every result unrun. They must not be used to launch Beam: model, API route,
profile and first inputs differ. A separate replacement protocol must retain the
original requests, external criteria/source-review rubric, fixed cell denominators
and approved limits, then freeze actual selected-provider client flows and exact
inputs under independent review. No cell, including local cells in that comparison,
may run from the obsolete plan. Existing cohorts, 8/10 target, no-go, defaults and
G2–G5 are unchanged. This document is an access/selection update, not that freeze.

Next offline preparation addresses two concrete prerequisites: the newly listed
Go compatible route is outside the recorder's explicit path allowlist, and eager
recording allocation cannot reopen retained input directories for GET-only review.
Add bounded mechanics and actual CLI regression coverage separately; neither
will by itself prove hosted connectivity, account limits or model quality.
