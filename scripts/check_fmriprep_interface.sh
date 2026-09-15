#!/usr/bin/env bash
set -euo pipefail

IMAGE="nipreps/fmriprep:latest"
HELP="$(docker run --rm "$IMAGE" --help 2>&1)"
VERSION="$(printf '%s\n' "$HELP" | sed -n 's/.*workflows v\([0-9][^ ]*\).*/\1/p' | head -1)"

for flag in \
  --participant-label --session-label --subject-anatomical-reference \
  --nprocs --omp-nthreads --mem --low-mem --anat-only --level \
  --ignore --force --output-spaces --output-layout --skull-strip-t1w \
  --fs-no-reconall --skip_bids_validation --stop-on-first-crash; do
  if ! grep -q -- "$flag" <<< "$HELP"; then
    echo "Missing expected fMRIPrep option: $flag" >&2
    exit 1
  fi
done

echo "fMRIPrep interface is compatible. version=${VERSION:-unknown}, image=$IMAGE"
