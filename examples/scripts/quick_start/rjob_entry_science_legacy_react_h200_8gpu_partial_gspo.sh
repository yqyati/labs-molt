#!/usr/bin/env bash
# RJob entrypoint for the 8-GPU partial-rollout + GSPO experiment.
# Reuse the standard entrypoint's mount checks and offline dependency install;
# only the training recipe differs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STANDARD_ENTRYPOINT="$SCRIPT_DIR/rjob_entry_science_legacy_react_h200_8gpu.sh"
PARTIAL_TRAIN_SCRIPT="$SCRIPT_DIR/rl_science_legacy_react_h200_8gpu_partial_gspo.sh"

test -f "$STANDARD_ENTRYPOINT" || {
  echo "Missing standard RJob entrypoint: $STANDARD_ENTRYPOINT" >&2
  exit 1
}
test -f "$PARTIAL_TRAIN_SCRIPT" || {
  echo "Missing partial GSPO training script: $PARTIAL_TRAIN_SCRIPT" >&2
  exit 1
}

export TRAIN_SCRIPT="${TRAIN_SCRIPT:-$PARTIAL_TRAIN_SCRIPT}"
export RJOB_NAME="${RJOB_NAME:-science-agent-async-training-gspo}"
exec bash "$STANDARD_ENTRYPOINT"
