#!/usr/bin/env bash
# RJob container entrypoint for the Qwen3.5-9B 8-GPU science RL run.
# Reuse the quick-start entrypoint for mounts, dependency setup, and preflight;
# this wrapper only selects the 9B training recipe.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
STANDARD_ENTRYPOINT="$REPO_ROOT/examples/scripts/quick_start/rjob_entry_science_legacy_react_qwen35_9b_h200_8gpu.sh"
TRAIN_SCRIPT="$REPO_ROOT/examples/scripts/quick_start/rl_science_legacy_react_qwen35_9b_h200_8gpu.sh"

test -f "$STANDARD_ENTRYPOINT" || {
  echo "Missing Qwen3.5-9B quick-start entrypoint: $STANDARD_ENTRYPOINT" >&2
  exit 1
}
test -f "$TRAIN_SCRIPT" || {
  echo "Missing Qwen3.5-9B training script: $TRAIN_SCRIPT" >&2
  exit 1
}

export TRAIN_SCRIPT
export RJOB_NAME="${RJOB_NAME:-science-agent-qwen35-9b-rl-8gpu}"
exec bash "$STANDARD_ENTRYPOINT"
