#!/usr/bin/env bash
# Train with a 96k context and finalize at 86k without context compaction.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GROUP_LENGTH_PENALTY_ENABLE=0
export MAX_LEN=98304
export MAX_NEW_TOKENS=98304
# CP=4 distributes the 96k sequence across four policy GPUs.
export TRAIN_MAX_TOKENS_PER_GPU="${TRAIN_MAX_TOKENS_PER_GPU:-25000}"
export AGENT_CONTEXT_COMPACTION=0
export AGENT_EVAL_FORCE_FINALIZE=1
export AGENT_CONTEXT_COMPACT_THRESHOLD_TOKENS=88064
export RJOB_NAME="${RJOB_NAME:-science-agent-vendor3443-grpo8x8-lenpen-96k-f86k}"
export RUN_NAME="${RUN_NAME:-science_agent_vendor3443_math40_grpo_batch8x8_length_penalty_96k_forcefinal86k_8gpu_4plus4_$(date +%Y%m%d_%H%M%S)}"
exec bash "$SCRIPT_DIR/rjob_entry_science_vendor6000_sandbox_batch8x8_length_penalty_h200_8gpu_4plus4.sh" "$@"
