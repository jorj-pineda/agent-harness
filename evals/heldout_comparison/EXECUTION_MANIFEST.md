# Execution manifest and parent-receipt binding

The read-only load_execution API takes a manifest path/hash and separate parent
approval path/hash. It bounds private regular files, validates strict typed schemas,
requires the parent receipt to bind the exact manifest hash and runtime revision,
then rechecks both artifacts. It creates no directories, providers, services,
requests, receipts or approvals. There is no CLI or approval generator.

The manifest explicitly selects the runtime commit, protocol/capture/runtime/
dependency/image-receipt/binary hashes, staged/capture/output/image/binary paths,
and effective command environment. Paths must be absolute and known bounded
runtime environment keys must include the forced pytest startup protections.
The manifest itself always records inference_authorized=false: preparing bytes
is not permission to execute them.

The separate parent receipt names an approval ID, the manifest/runtime binding,
freeze reviewed head and merge revision, and affirmative review/CI/merge/launch
assertions. Those fields are assertions from the parent, not facts authenticated
by this schema. Hashes select bytes; they do not authenticate an issuer or prove
Git ancestry, review results, CI, merge state, user consent or freshness. A trusted
operator must obtain the receipt from the actual authorized parent and verify
those prerequisites independently. An arbitrary JSON file with true fields must
never become self-approval. Tests use clearly scripted receipts only.

This is a binding check, not complete launch admission. Future runner code must
still verify the actual clean retained runtime and selected artifacts, source/
criteria and full sixteen-cell schedule, fresh unconsumed prepared workspaces,
nonoverlapping canonical output paths, current model image/version and owned
resource startup, request guards, single-use output reservation, final preservation
and owned settlement. Failure or unknown evidence stops without rebinding/retry.
The live runner and concrete exact freeze are still incomplete; no inference or
launch authority follows from loading this tuple.
