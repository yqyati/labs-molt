#!/usr/bin/env bash
# GEO group baseline with routing replay and asynchronous partial rollout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPFS2_ROOT="${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}"
export DATASET="${DATASET:-$GPFS2_ROOT/scientific_reasoning_and_research/yangqingyu/rft/molt_science_legacy/train.jsonl}"
export MAX_SAMPLES=3000
export N_SAMPLES_PER_PROMPT=8 ROLLOUT_BATCH_SIZE=8 TRAIN_BATCH_SIZE=64
export DYNAMIC_FILTERING_ENABLE=0 GROUP_LENGTH_PENALTY_ENABLE=0
export PARTIAL_ROLLOUT_ENABLE=1
export ROLLOUT_VLLM_GENERATE_BATCH_SIZE=6
export MAX_LEN=96256 MAX_NEW_TOKENS=96256
export TRAIN_MAX_TOKENS_PER_GPU="${TRAIN_MAX_TOKENS_PER_GPU:-25000}"
export AGENT_CONTEXT_COMPACTION=0 AGENT_EVAL_FORCE_FINALIZE=0
export AGENT_CONTEXT_COMPACT_THRESHOLD_TOKENS=86016
export RJOB_NAME="${RJOB_NAME:-science-v1-router-replay-geo-no-lenpen-no-forcefinal}"
export RUN_NAME="${RUN_NAME:-science_v1_router_replay_geo_no_lenpen_94k_no_forcefinal_8gpu_4plus4_$(date +%Y%m%d_%H%M%S)}"

exec bash "$SCRIPT_DIR/rjob_entry_science_vendor6000_sandbox_batch8x8_length_penalty_h200_8gpu_4plus4.sh" \
  --actor.adam.lr 1e-6 --train.routing_replay \
  --algo.advantage.estimator reinforce_baseline \
  --actor.loss_agg_mode token-mean --actor.eps_clip_low_high 0.2 0.27 \
  --algo.advantage.is_correction_level geo \
  --algo.advantage.is_correction_gating ratio \
  --algo.advantage.is_correction_mode mask --algo.advantage.is_correction_threshold 0.5 5.0 \
  --algo.advantage.score_centering_top_k 0 \
  --train.async_queue_size 2 --train.partial_rollout_enable \
  --ckpt.save_steps 50 \
  --rollout.temperature 1.0 --rollout.top_p 1.0 --train.max_epochs 1 \
  "$@"
