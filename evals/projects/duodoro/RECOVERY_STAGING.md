# Evaluation-owned paired staging

`recovery_staging.py` supplies the preservation primitives for PR 2 of the
[five-PR local comparison](PROJECTED_RECOVERY_PROTOCOL.md). It runs no inference,
starts no service and exposes no application restore route. Actual CLI/API
preflight, resource lifecycle integration, exact execution freeze and live results
remain separate work. The fixed plan/ledger is still unrun.

`OwnedWorkspace.create` requires a new directory beneath a canonical existing
parent. It creates one private workspace and an outside ownership receipt. It
cannot adopt an existing directory. Every operation checks the canonical root,
device/inode and receipt; staging preserves that root rather than replacing it.
These are trusted evaluation primitives, not a security boundary against Python
callers who fabricate objects or an external process writing concurrently.

`seal_tree` retains the shared checkpoint scope of eligible regular-file bytes,
hashes and modes, plus eligible directory paths/modes (including empty directories),
root identity and metadata at excluded boundaries. It reuses the 2,000-entry/
16,000,000-byte content limits, bounds directory/ignored entries separately at 2,000,
and caps serialized manifests at 1,000,000 bytes. Hash-pinned manifests and
deduplicated content blobs live outside the workspace, with directory mode 0700 and
file mode 0600. Failure can leave incomplete evidence; it grants no staging proof.
Plaintext is neither encrypted nor OS-enforced immutable: hashes detect later
changes when the evidence is consumed. No `.env` or excluded contents are copied.

`score` requires unchanged seals of the prepared fixture, external acceptance and
reference trees. It runs the existing isolated-interpreter owned acceptance runner
twice against the current workspace, retaining both bounded raw `ProcessResult`
outputs (including truncation flags). Both must agree on exit 0 or 1. Rejection is
a scored artifact, not permission to hide that row. Timeout, cancellation,
infrastructure error, changed criteria, grader mutation or a change between grading
and preservation produces no scored replacement authorization. The artifact and
its original protection handles stay linked even when the next task has different
protected inputs. External grading results remain outside model inputs.

`stage` accepts a retained approved initial/source image and, after the very first
empty-root initialization, an exact last-scored artifact. It verifies the original
and incoming protected trees, all sealed blobs and grade-output hashes, and exact
current eligible files/modes/directories/root plus excluded-boundary metadata.
Later additions, deletions, edits, mode changes, empty directories, redirected roots,
symlinks, special files or damaged evidence refuse before replacement. An excluded
entry's parent cannot be removed incidentally. Eligible files are recreated from
the approved bytes/modes; ignored entries stay in place. No `git reset`, `clean`,
checkout, commit or push is involved. No arbitrary directory is a restoration target.

Every scoring/staging/store-copy call requires a caller-supplied settlement checker
that raises while any owned API, tool worker or listener is unresolved. Staging
checks it again immediately before replacement. The function is an integration
obligation, not evidence that this module can discover detached processes. The
[offline client preflight](PROJECTED_RECOVERY_PREFLIGHT.md) now supplies/tests an
append-only owned-API process/group/listener checker over real local HTTP lifecycles.
Actual backend shutdown and fresh launch gates still remain; a successful no-process
unit fixture cannot stand in for them.

`clone_closed_stores` creates a fresh private directory containing exactly
`sessions.sqlite` and `facts.sqlite`. Each input is a bounded regular standalone
rollback-journal database (64,000,000 bytes per file), with caller-pinned exact
bytes and a successful SQLite quick check. WAL headers, sidecars, unexpected files,
symlinks, drift and overlapping directories refuse. Read-only integrity connections
are explicitly closed. The settled caller must already have closed the actual
application stores; missing sidecars alone do not establish that fact. Copy a
source pair into one parent image, then clone each arm from its same hashes. There
are no archive/run/checkpoint/fact/path/ID/time edits. The source and parent are
rechecked; a partial copy on failure is retained and must never launch an arm.

The module does not provide crash-atomic replacement, ACL/ownership/xattr capture,
process isolation, cross-process locks or an execution lease. A concurrent writer
or filesystem error during mutation can leave a partially staged owned tree:
stop the cohort and retain it, without automatic cleanup/retry. Excluded contents
are neither restored nor attested; unchanged boundary metadata cannot certify
their contents, and retained caches are not experimentally identical arm inputs.
The root's private mode stays fixed rather than adopting a prepared root's mode.

Offline validation covers preservation failures, actual independent grade/regrade,
mode/binary/empty-directory restoration, SQLite checkpoint-blob retention and
independent sibling journals. Fresh pinned full Duodoro T02/T04 archives also pass
initial-fail/reference-pass/restaged-initial-fail checks at one owned root/inode,
with unchanged protected criteria and Duodoro working head/status/index/refs.
Private receipt:
`/Users/jorge/.local/share/agent-harness/trials/paired-recovery-staging-pw4e38l9/validation.json`,
SHA-256 `d0b46842bf22268fa2c249d06c1beeb9f61fc0eb7776153fa7fafd21a1505331`.
This is fixture/mechanics evidence, not a model-quality comparison, measured recovery
gain, default change or closure of the original cohort/no-go/8-of-10/G2–G5 gates.
