# Exact first-input guard — evaluation prerequisite only

The guard loads one cell's normalized and wire bytes from a fully verified
capture bundle using its caller-selected index hash and original canonical root.
It verifies file modes/hashes again after bundle validation. The shared capture
serializer is reused; it preserves paths, order, temperature, output limits and
tool definitions, and refuses unavailable or oversized identities.

`PinnedOllamaProvider` compares the first normalized request before invoking the
production adapter. Its transport compares the first actual `/api/chat` body
before delegating to the explicitly supplied transport. A changed model, context
option, thinking option or serialization therefore cannot slip through an unchanged
normalized-input check. Origin validation is literal-loopback-only and credential
free. No default transport or live CLI is provided by this module.

Mismatches latch the guard closed without forwarding. Exceptions, cancellation,
HTTP errors and malformed adapter responses also prevent retry through that guard.
Concurrent calls refuse while a submission is unresolved. The transport's attempt
count means delegation was attempted; it is not successful inference, known usage,
backend settlement or proof of zero billing on a failed submission.

After a successful first response, subsequent serial tool-loop requests may pass;
they must still use the existing protocol budget, usage/model and preservation
checks. The eventual operator must create one guard per cell and compose it with
the checked-response provider. This first-input component does not independently
implement all stop rules, resource ownership or process lifecycle management.
Its supplied hashes bind inputs, not approver identity or authorization.

Tests use only mocked transports. The capture/staging regression now loads actual
archived pins and recaptures the first request through this guard at the original
named path, matching both byte representations. Separate controls reject changed
normalized inputs and wire options before forwarding, retain failure latches and
permit a normal successful follow-up. These establish mechanics, not model quality.

## Freeze status

The current local metadata observation is unavailable/ConnectError. No service was
started or changed to bypass it. Owned runtime/model-image identity, fresh service
metadata, backend settlement and the complete operator/refusal recipe remain
unverified. The exact execution freeze, independent review/CI/merge, parent approval
receipt and fresh launch gates remain incomplete. This fifth development slice
adds an enforceable prerequisite; it does not complete the initially proposed
freeze, authorize generation or establish a useful unattended envelope.

## Reported native response identity and completion

The evaluation-only pinned Ollama provider also requires a nonempty explicit model
string and literal done=true in every decoded native payload before returning it
to the runtime. The shared production adapter retains its existing fallback
contract; the evaluation guard must not treat that fallback as observed identity.
The separate protocol wrapper still compares an explicit returned model with the
selected model and requires normalized usage. Missing/empty native identity or a
missing/false/non-boolean completion marker fails and latches closed, even if the
response contains tool calls and valid usage. No retry or later response escapes.

The mandatory-mock operator has --fault missing_model and --fault not_done probes.
They retain the raw scripted native response, one failed result and fifteen unrun
cells without tool dispatch or second admission. A normal scripted 16-cell run
still passes. These checks establish reported payload fields, not backend
settlement, authenticated model identity, inference quality or launch readiness.
The live operator/freeze/approval and fresh-launch gates remain incomplete.
