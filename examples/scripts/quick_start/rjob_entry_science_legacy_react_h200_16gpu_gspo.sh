#!/usr/bin/env bash
# RJob entrypoint for the 16-GPU EP=8 GSPO experiment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STANDARD_ENTRYPOINT="$SCRIPT_DIR/rjob_entry_science_legacy_react_h200_16gpu.sh"
GSPO_TRAIN_SCRIPT="$SCRIPT_DIR/rl_science_legacy_react_h200_16gpu_gspo.sh"

test -f "$STANDARD_ENTRYPOINT" || { echo "Missing standard 16-GPU entrypoint: $STANDARD_ENTRYPOINT" >&2; exit 1; }
test -f "$GSPO_TRAIN_SCRIPT" || { echo "Missing 16-GPU GSPO script: $GSPO_TRAIN_SCRIPT" >&2; exit 1; }

export TRAIN_SCRIPT="${TRAIN_SCRIPT:-$GSPO_TRAIN_SCRIPT}"
export RJOB_NAME="${RJOB_NAME:-science-agent-async-training-gspo-16gpu}"
exec bash "$STANDARD_ENTRYPOINT"
