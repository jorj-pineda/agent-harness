# Owned-endpoint inference transport boundary

OwnedInferenceTransport requires an explicit inner transport and a synchronous
ownership callback. It supplies no default network delegate, provider constructor,
service discovery or live CLI. The future approved operator must provide its own
owned-child/listener check and must pass all freeze/approval/launch gates first.
The callback must raise for unknown, foreign or dead ownership; returning a false
value is not a refusal.

Only POST http://127.0.0.1:11447/api/chat is permitted, without query, credentials,
fragment or a different origin. Ownership is checked before delegate invocation
and again before returning its response. Route/ownership failure, transport error,
cancellation, redirects and HTTP errors latch closed. No resubmission is allowed.
A concurrent submission is refused and invalidates the original response too.
Rejected responses are closed best-effort; close errors are retained separately
without replacing the primary failure. Caller-owned cleanup must still run.

This surrounds transport delegation, not subsequent streamed-body consumption,
protocol decoding, tool dispatch or a whole cell. Those remain guarded by the
pinned provider, protocol checks and owned process lifecycle. A response, native
done marker or ownership callback does not prove remote generation settlement.
PID/port checks are bounded observations, not atomic leases or process isolation.
A dishonest callback can return success; this component grants no authority.
The existing mandatory-mock operator is unchanged. Tests use MockTransport only,
never a live backend. Full live operator/refusal/settlement, exact execution freeze,
parent approval and fresh launch gates remain incomplete. No inference or quality
claim follows.
