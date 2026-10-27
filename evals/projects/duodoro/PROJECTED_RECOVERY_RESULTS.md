# Projected recovery: comparison stopped at the first source

**2026-10-27 — No policy comparison observed.** The independently reviewed
[execution freeze](PROJECTED_RECOVERY_EXECUTION_FREEZE.md) merged as
`28a43324f250885a9d8337d34f42e4436ad4678c` after both CI checks passed on reviewed
head `6650d21572f369b70f24025a630b17e7de76f483`. Parent receipts and fresh launch
checks passed before one original T02 source ran at archived runtime
`2e7dce9099aaaa25461fcccabb72fe4f5e50aec3`. The
[sanitized retained rows](../../results/2026-10-27-gemma-projected-recovery-stopped.json)
include every planned source and followup, with unknown fields preserved.

T02-S1's second provider request received a read timeout after approximately
60.007 seconds. The finalized run is failed, has no final TurnResponse, and has
unknown aggregate usage. Its one returned response reports a known prefix of
1,964 prompt and 22 output tokens; that prefix is not the failed-call or task total.
Saved run creation to finalization is 84.744 seconds, including setup/storage;
it is not a dispatch-only timing or proof of known token-budget compliance.

| Source | Observed source outcome | C followup | P followup |
|---|---|---|---|
| T02-S1 | Failed: second-call read timeout; unchanged rejected artifact | Unrun | Unrun |
| T04-S1 | Unrun: cohort stop | Unrun | Unrun |
| T02-S2 | Unrun: cohort stop | Unrun | Unrun |
| T04-S2 | Unrun: cohort stop | Unrun | Unrun |

This is zero accepted artifacts and zero completed-and-accepted results among
one attempted source, with three sources and all eight followups unrun. It is
not a completed 0/4-versus-0/4 policy comparison, a recovery failure, or evidence
that C and P perform equally. Shared source cost would be charged to each policy
workflow but counted once physically; its aggregate cost is unknown. No arm usage
or rescue rate is observed. No replacement, extra attempt, resubmission, limit or
selection change, model fallback or manual patch repair occurred.

## Evidence and preservation

The first normalized/wire input matches the frozen source capture and the saved
execution profile matches preparation. Both actual requests have paired private
normalized and native wire captures, matching the two saved request identities.
Wire bodies are 9,467 and 27,330 bytes. The first response requests `read_file` on
AGENTS.md; its tool reply records lines 1–265 of 265. This is observed read coverage,
not comprehension or compliance. The second request includes that history; the
larger payload and timeout do not establish a backend cause or a remedy.

The operator retained the failed run/archive, both checkpoint records and closed
SQLite stores, complete paired inputs, observations and the final eligible artifact.
Its original independent grade/regrade both reject with exit 1. A separate post-stop
copy reconstructed from hash-checked saved blobs also rejects with exit 1 under the
same original external criteria. Eligible file bytes/modes and non-root directory
modes match the initial source. No model edit or configured verification result is
observed; the absent final response must not be displayed as completion/check success.
Protected fixture, criteria and reference seals remain unchanged.

The cohort stop marker records provider/input failure. The operator's nonzero exit
is the expected refusal at the next phase gate after preservation, not permission
to resume. It did not create parent/arm stores or start the next source. All twelve
planned rows remain in the raw ledger. Its source lifecycle label `settled` means
process/evidence settlement; the retained run status is **failed**.

Owned API and backend exit zero, process groups are absent and ports 18843/11447
are free. API lifespan reports normal return and completed tasks; backend cleanup
retains unchanged existing identity-file metadata. User Ollama remains on 11434;
it was neither stopped nor unloaded. Duodoro working head/status/index/refs remain
unchanged. Post-stop frozen runtime/recipe/evidence/environment/model verification
passes. Paths, allowlists and process groups remain limited ownership checks, not
OS isolation or proof about arbitrary detached workers/ignored content.

Private evidence is retained at
`/Users/jorge/.local/share/agent-harness/trials/projected-recovery-live-irl9s2w1`:
`sources/`, `STOP_ALL.json`, `live-operator.log`, parent approval/review receipts and
`results-audit/`. The public JSON contains hashes and bounded outcome metadata,
not source/message bodies, headers, credentials or `.env` contents.

Preparation G0 (196 focused/2,178 offline, static gates and both scripted matrices)
is historical exact-runtime mechanics evidence. It was not rerun as a model-quality
check for this reporting-only change. Rehearsals remain scripted evidence. This
finite comparison is closed at its mandated stop, without measuring either recovery
policy. Further inference needs a separately justified scope and fresh freeze;
these unrun rows are not replay slots. Original cohorts, defaults, 8-of-10 target,
no-go and broad G2–G5 remain unchanged. No release or quality gain is claimed.
