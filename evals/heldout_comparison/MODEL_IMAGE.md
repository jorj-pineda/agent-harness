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
