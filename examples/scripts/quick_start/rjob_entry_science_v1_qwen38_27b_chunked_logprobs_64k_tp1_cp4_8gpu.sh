#!/usr/bin/env bash
# Experimental chunked LM-head logprobs: policy TP1/CP4, rollout TP4.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
GPFS2_ROOT="${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}"
SCI_RESEARCH_ROOT="${SCI_RESEARCH_ROOT:-$GPFS2_ROOT/scientific_reasoning_and_research}"

export MODEL_PATH="${MODEL_PATH:-$GPFS2_ROOT/yangqingyu/workspace/model/qwen38_27b_sft500_visual}"
export DATASET="${DATASET:-$SCI_RESEARCH_ROOT/yangqingyu/rft/molt_science_legacy/train.jsonl}"
export MAX_SAMPLES="${MAX_SAMPLES:-3000}"
export N_SAMPLES_PER_PROMPT=8 ROLLOUT_BATCH_SIZE=8 TRAIN_BATCH_SIZE=64
export ACTOR_GPUS_PER_NODE=4 VLLM_TP_SIZE=4 FSDP_EP_SIZE=1 FSDP_CP_SIZE=4 FSDP_TP_SIZE=1
# PyTorch TP styles require torch Linear; attention can still use TE.
export MOLT_LINEAR_BACKEND=torch
export MOLT_EXPERIMENTAL_LINEAR_LOGPROBS="${MOLT_EXPERIMENTAL_LINEAR_LOGPROBS:-1}"
# Reduce-scatter each microbatch to avoid retaining full dense gradients.
export MOLT_DEFER_GRAD_SYNC=0
export ROLLOUT_VLLM_GENERATE_BATCH_SIZE=6
export DYNAMIC_FILTERING_ENABLE=0 GROUP_LENGTH_PENALTY_ENABLE=0
export PARTIAL_ROLLOUT_ENABLE=1
export MAX_LEN=65536 MAX_NEW_TOKENS=65536
export MAX_IMAGES_PER_PROMPT=1 AGENT_EVAL_ENABLE_MODEL_IMAGE_INPUTS=1
export AGENT_CONTEXT_COMPACTION=0 AGENT_EVAL_FORCE_FINALIZE=0
export AGENT_CONTEXT_COMPACT_THRESHOLD_TOKENS=65536
export TRAIN_MAX_TOKENS_PER_GPU="${TRAIN_MAX_TOKENS_PER_GPU:-25000}"
export SCIENCE_AGENT_SANDBOX_RESOURCE="${SCIENCE_AGENT_SANDBOX_RESOURCE:-10.12.111.143/ailab-ai4sgi}"
export SCIENCE_AGENT_ROOT="${SCIENCE_AGENT_ROOT:-$GPFS2_ROOT/yangqingyu/workspace/harness/science-agent-sandbox}"
export AGENT_PATH="$REPO_ROOT/examples/python/agents/science_legacy_react_sandbox.py"
export TRAIN_SCRIPT="$SCRIPT_DIR/rl_science_vendor6000_sandbox_batch8x8_length_penalty_h200_8gpu_4plus4.sh"
export RJOB_NAME="${RJOB_NAME:-science-v1-qwen38-27b-geo-no-lenpen-64k-chunked-tp1-cp4-8gpu}"
export RUN_NAME="${RUN_NAME:-science_v1_qwen38_27b_geo_no_lenpen_64k_chunked_tp1_cp4_batch8x8_48conc_8gpu_4plus4_$(date +%Y%m%d_%H%M%S)}"

exec bash "$SCRIPT_DIR/rjob_entry_science_legacy_react_h200_8gpu_4plus4.sh" \
  --vllm.disable_custom_all_reduce \
  --actor.adam.lr 1e-6 --actor.freeze_visual_encoder --actor.log_entropy \
  --algo.advantage.estimator reinforce_baseline \
  --actor.loss_agg_mode token-mean --actor.eps_clip_low_high 0.2 0.27 \
  --algo.advantage.is_correction_level geo \
  --algo.advantage.is_correction_gating ratio \
  --algo.advantage.is_correction_mode mask --algo.advantage.is_correction_threshold 0.5 5.0 \
  --algo.advantage.score_centering_top_k 0 \
  --train.async_queue_size 2 --train.partial_rollout_enable \
  --train.rollout_dump_dir "" \
  --ckpt.save_steps 50 --ckpt.save_hf \
  --rollout.temperature 1.0 --rollout.top_p 1.0 --train.max_epochs 1 \
  "$@"
