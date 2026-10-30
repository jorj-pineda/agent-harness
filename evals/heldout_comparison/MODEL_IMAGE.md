# Read-only model-image verification

```sh
uv run python -m evals.heldout_model_image \
  --models /absolute/path/to/evaluation-owned/models \
  --report /tmp/new-model-image-report.json
```

The CLI pins the Gemma manifest path and digest from the reviewed protocol. It
bounds the manifest before parsing, validates declared digest/size entries and
caps the declared aggregate at 16 GB. Only declared config/layer blobs are read;
other models, configuration/identity files and credentials are not inspected.
Each regular file is hashed in 1 MiB chunks with exact length and before/after
file-identity checks. Known directory links, final links, special files, missing
files, digest mismatch and observed changes refuse. The manifest is checked again
after blob verification. Unknown expected digest is never inferred from contents.

The private exclusive report contains hashes, sizes and observed modes, without
model/config contents. Report overlap with the selected image refuses. The tool
starts no service, loads no model and changes no image permissions or contents.
It does not download, clone or hardlink anything. Hashing can take substantial disk
I/O; byte bounds are not a hard wall-clock deadline or an atomic image snapshot.
External path races remain outside these scoped observations.

An image report does not verify service version, resource ownership, backend
compliance or later freshness. A historical evaluation-owned copy can be inspected
read-only, but its old stopped cohort and approval must never be resumed or reused.
Any new owned backend needs fresh lifecycle ownership, current metadata checks,
model-image rechecks and the exact execution freeze/approval/launch gates. File
verification is a prerequisite, not model-quality or inference evidence.

## Fresh file preparation

```sh
uv run python -m evals.heldout_model_copy \
  --source-models /absolute/path/to/existing/models \
  --directory /absolute/path/outside/git/new-image-container
```

The parent must already exist; the container must not exist. Source/destination
overlap and Git workspace destinations refuse. The source is verified against
the protocol before allocation. Only its pinned manifest and declared blobs are
cloned with the macOS clonefile API: exclusive distinct files using filesystem
copy-on-write, with no ordinary-copy, hardlink, cross-filesystem or download
fallback. Unsupported platforms/filesystems fail and retain partial artifacts.
Source inode equality refuses before permission changes. Copied files are mode
0400; directories are mode 0700. The original contents/modes are not modified.

The source is independently reverified after cloning and observations must match;
the new image is independently verified too. The mode-0600 prepared-image.json
receipt is written last and exclusively, including source/copy hash observations.
Receipt absence means incomplete preparation. Occupied and partial destinations
are never deleted, repaired or reused. The receipt denotes file preparation,
not service ownership or approval authority. No identity/configuration files,
other models, historical cohort receipts or launch approvals are copied.

This utility starts no process and makes no network request. It has byte bounds
from image verification, but no hard duration bound, external-writer exclusion,
atomic snapshot or guarantee of later freshness. Host permissions still permit
external writers. A new backend must separately establish its environment,
process/listener ownership, settlement and current metadata, followed by an exact
reviewed execution freeze, parent approval and fresh launch gates. Never resume
the earlier stopped cohort or reuse its approval. Scripted tests substitute a
small ordinary-copy primitive; they prove refusal/preservation orchestration,
not APFS behavior. An actual macOS preparation separately checks clonefile.
