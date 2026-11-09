#!/usr/bin/env bash
# Frozen-router GRPO with geometric-mean IS gating and asynchronous partial rollout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPFS2_ROOT="${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}"
export DATASET="${DATASET:-$GPFS2_ROOT/yangqingyu/workspace/science_training_reports/vendor593_quality_gate_round14_20261105/supported212_train.jsonl}"
export MAX_SAMPLES=212
export N_SAMPLES_PER_PROMPT=8 ROLLOUT_BATCH_SIZE=8 TRAIN_BATCH_SIZE=64
export DYNAMIC_FILTERING_ENABLE=1 GROUP_LENGTH_PENALTY_ENABLE=0
export PARTIAL_ROLLOUT_ENABLE=1
export ROLLOUT_VLLM_GENERATE_BATCH_SIZE=6
export MAX_LEN=96256 MAX_NEW_TOKENS=96256
export TRAIN_MAX_TOKENS_PER_GPU="${TRAIN_MAX_TOKENS_PER_GPU:-25000}"
export AGENT_CONTEXT_COMPACTION=0 AGENT_EVAL_FORCE_FINALIZE=1
export AGENT_CONTEXT_COMPACT_THRESHOLD_TOKENS=86016
export RJOB_NAME="${RJOB_NAME:-vendor212-grpo-frozen-router-no-tis-no-sc}"
export RUN_NAME="${RUN_NAME:-vendor212_grpo_frozen_router_no_tis_no_sc_94k_forcefinal84k_8gpu_4plus4_$(date +%Y%m%d_%H%M%S)}"

exec bash "$SCRIPT_DIR/rjob_entry_science_vendor6000_sandbox_batch8x8_length_penalty_h200_8gpu_4plus4.sh" \
  --actor.adam.lr 1e-6 --actor.freeze_moe_router \
  --algo.advantage.estimator grpo --algo.advantage.no_whiten \
  --algo.dynamic_filtering_enable --algo.dynamic_filtering_range 0 1 \
  --actor.loss_agg_mode token-mean --actor.eps_clip_low_high 0.2 0.27 \
  --algo.advantage.is_correction_level geo \
  --algo.advantage.is_correction_gating ratio \
  --algo.advantage.is_correction_mode mask --algo.advantage.is_correction_threshold 0.5 5.0 \
  --algo.advantage.score_centering_top_k 0 \
  --train.async_queue_size 2 --train.partial_rollout_enable \
  --rollout.temperature 1.0 --rollout.top_p 1.0 --train.max_epochs 1 \
  "$@"
