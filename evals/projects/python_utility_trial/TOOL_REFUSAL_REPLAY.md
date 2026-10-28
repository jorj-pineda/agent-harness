# Recorded Python tool refusals replayed offline

The completed [Python trial](RESULTS.md) exposed tool-argument failures before
useful edits or verification. Replaying its original arguments through the shared
tool registry at `a48e68a` reproduces **17 refusals exactly: nine `replace_text`
and eight `run_command` calls**. Every copied workspace stays unchanged and no
process dispatch occurs. [Recorded evidence](tool-refusal-replay.json) binds the
retained driver and raw result; all 639 original outcome hashes verified first.

The driver uses fresh copies of the original public fixtures and the original
argument dictionaries, without decoding, repairing or substituting arguments.
Only recorded errors for these two tools are selected. Policy-level “Tool not
executed” records are excluded because they never reached the registry. Edit
calls are replayed only when their expected hash matches the fresh original
fixture; this excludes stale-state substitutions after a successful edit.
Process execution is replaced with a sentinel that fails if reached. These are
offline contract observations, not additional model attempts or coding results.

The nine exact-match failures already identify literal backslash-n characters
where the source contains line breaks, preserve the stale-edit hash check, and
instruct the caller to copy actual read content. The command failures already
reject quoted JSON arrays or malformed executable tokens, with corrective
separate-token guidance. The recorded model responses nevertheless include
repeated encoding mistakes and, for commands, invented separator strings. The
replay confirms their existing refusal boundary; it cannot establish why the
model produced them or whether different feedback would change its behavior.

The current `replace_text` schema describes its fields only as exact text and
replacement text, and its tool description recommends small edits. Its detailed
encoding guidance arrives after failure. This makes a small, explicit schema
example a candidate for a future matched experiment, not a demonstrated remedy.
Likewise, changing to a different existing tool preset could change the model's
behavior, but is not measured by this replay.

Do not automatically unescape edit strings or split command strings: literal
escapes can be intended source, and string repair would change submitted edit
meaning or command validation. Any proposed schema/feedback change must preserve
exact argument semantics, stale/ambiguous refusal and command allowlists, and
first pass deterministic regression checks. A separately frozen fresh comparison
is required to claim improved accepted completions. No prompt, tool behavior,
budget, default, criterion or release decision changes in this diagnosis.

The original cohort stays closed at 2/10 accepted artifacts and 0/10 accepted
completions, with three unknown usage totals. No inference ran, no old attempt
was resumed, and no user repository or retained runtime was used as an editable
fixture. This finite replay supports selecting a narrow next experiment; it does
not authorize generation or establish that an implementation change will help.
