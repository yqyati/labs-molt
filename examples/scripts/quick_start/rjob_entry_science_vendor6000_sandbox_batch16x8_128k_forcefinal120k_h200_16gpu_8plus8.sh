#!/usr/bin/env bash
# Use two eight-GPU nodes: one for rollout and one for policy training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ACTOR_GPUS_PER_NODE=8
export VLLM_TP_SIZE=8
export FSDP_EP_SIZE=8
export FSDP_CP_SIZE=8
export ROLLOUT_BATCH_SIZE=16
export N_SAMPLES_PER_PROMPT=8
export ROLLOUT_VLLM_GENERATE_BATCH_SIZE=12
export TRAIN_BATCH_SIZE=128
export GROUP_LENGTH_PENALTY_ENABLE=0
export MAX_LEN=131072
export MAX_NEW_TOKENS=131072
# CP=8 keeps the per-GPU sequence budget near the existing 64k/CP=4 recipe.
export TRAIN_MAX_TOKENS_PER_GPU="${TRAIN_MAX_TOKENS_PER_GPU:-17000}"
export AGENT_CONTEXT_COMPACTION=0
export AGENT_EVAL_FORCE_FINALIZE=1
export AGENT_CONTEXT_COMPACT_THRESHOLD_TOKENS=122880
export RJOB_NAME="${RJOB_NAME:-science-agent-vendor3443-grpo16x8-128k-f120k}"
export RUN_NAME="${RUN_NAME:-science_agent_vendor3443_math40_grpo_batch16x8_128k_forcefinal120k_16gpu_8plus8_$(date +%Y%m%d_%H%M%S)}"
exec bash "$SCRIPT_DIR/rjob_entry_science_vendor6000_sandbox_batch8x8_length_penalty_h200_8gpu_4plus4.sh" "$@"
