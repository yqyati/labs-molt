#!/usr/bin/env bash
# Same Math baseline, with a larger context and an 8-group asynchronous rollout pool.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPFS2_ROOT="${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}"
export DATASET="${DATASET:-$GPFS2_ROOT/yangqingyu/workspace/science_training_reports/math_v2_2829_20261105/train_dedup1440.jsonl}"
export MAX_SAMPLES=1440
export ACTOR_GPUS_PER_NODE=8 VLLM_TP_SIZE=8 FSDP_EP_SIZE=8 FSDP_CP_SIZE=8
export N_SAMPLES_PER_PROMPT=8 ROLLOUT_BATCH_SIZE=8 TRAIN_BATCH_SIZE=64
export DYNAMIC_FILTERING_ENABLE=0 GROUP_LENGTH_PENALTY_ENABLE=1
export PARTIAL_ROLLOUT_ENABLE=1
export ROLLOUT_VLLM_GENERATE_BATCH_SIZE=8
export MAX_LEN=131072 MAX_NEW_TOKENS=131072
export TRAIN_MAX_TOKENS_PER_GPU="${TRAIN_MAX_TOKENS_PER_GPU:-25000}"
export AGENT_CONTEXT_COMPACTION=0 AGENT_EVAL_FORCE_FINALIZE=0
export AGENT_CONTEXT_COMPACT_THRESHOLD_TOKENS=86016
export RJOB_NAME="${RJOB_NAME:-math1440-frozen-router-no-tis-no-sc-128k-no-forcefinal-16gpu-cp8}"
export RUN_NAME="${RUN_NAME:-math1440_frozen_router_no_tis_no_sc_128k_no_forcefinal_16gpu_8plus8_cp8_$(date +%Y%m%d_%H%M%S)}"

exec bash "$SCRIPT_DIR/rjob_entry_science_vendor6000_sandbox_batch8x8_length_penalty_h200_8gpu_4plus4.sh" \
  --actor.adam.lr 1e-6 --actor.freeze_moe_router --actor.log_entropy \
  --algo.advantage.estimator reinforce_baseline \
  --actor.loss_agg_mode token-mean --actor.eps_clip_low_high 0.2 0.27 \
  --algo.advantage.is_correction_level geo \
  --algo.advantage.is_correction_gating ratio \
  --algo.advantage.is_correction_mode mask --algo.advantage.is_correction_threshold 0.5 5.0 \
  --algo.advantage.score_centering_top_k 0 \
  --reward.group_length_penalty.enable --reward.group_length_penalty.max_penalty 0.1 \
  --train.async_queue_size 2 --train.partial_rollout_enable \
  --ckpt.save_steps 100 \
  --rollout.temperature 1.0 --rollout.top_p 1.0 --train.max_epochs 1 \
  "$@"
