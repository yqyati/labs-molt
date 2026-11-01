#!/usr/bin/env bash
# Train the mixed science/math dataset with a 94k limit and 84k finalization.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPFS2_ROOT="${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}"
export DATASET="${DATASET:-$GPFS2_ROOT/yangqingyu/workspace/science_training_reports/science_mix1200_20261031/train.jsonl}"
export MAX_SAMPLES=1200
export GROUP_LENGTH_PENALTY_ENABLE=0
export MAX_LEN=96256
export MAX_NEW_TOKENS=96256
export TRAIN_MAX_TOKENS_PER_GPU="${TRAIN_MAX_TOKENS_PER_GPU:-25000}"
export AGENT_CONTEXT_COMPACTION=0
export AGENT_EVAL_FORCE_FINALIZE=1
export AGENT_CONTEXT_COMPACT_THRESHOLD_TOKENS=86016
export RJOB_NAME="${RJOB_NAME:-science-agent-mix1200-94k-f84k-bs8x8}"
export RUN_NAME="${RUN_NAME:-science_agent_mix1200_batch8x8_94k_forcefinal84k_8gpu_4plus4_$(date +%Y%m%d_%H%M%S)}"

exec bash "$SCRIPT_DIR/rjob_entry_science_vendor6000_sandbox_batch8x8_length_penalty_h200_8gpu_4plus4.sh" \
  --actor.adam.lr 5e-7 "$@"
