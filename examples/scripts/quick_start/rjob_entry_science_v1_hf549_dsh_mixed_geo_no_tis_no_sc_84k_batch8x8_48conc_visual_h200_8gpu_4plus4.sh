#!/usr/bin/env bash
# Science V1 RL with a fixed per-question Science/DSH harness split.
# Starts from the original SFT479 base: 84k context, 8x8 rollouts, 48 concurrent,
# 4 actor GPUs + 4 vLLM GPUs, and HF checkpoints every 100 steps.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
GPFS2_ROOT="${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}"
GPFS1_ROOT="${GPFS1_ROOT:-/mnt/shared-storage-user/ai4sgi-gpfs1}"
YANGQINGYU_MOUNT_ROOT="${YANGQINGYU_MOUNT_ROOT:-$GPFS2_ROOT/yangqingyu}"
GPFS1_USER_ROOT="${GPFS1_USER_ROOT:-$GPFS1_ROOT/yangqingyu}"
SCI_RESEARCH_ROOT="${SCI_RESEARCH_ROOT:-$GPFS2_ROOT/scientific_reasoning_and_research}"
SCIENCE_AGENT_ROOT="${SCIENCE_AGENT_ROOT:-$YANGQINGYU_MOUNT_ROOT/workspace/harness/science-agent-sandbox}"
DEEPSEEK_ROOT="$SCIENCE_AGENT_ROOT/deepseek-harness"

export MODEL_PATH="${MODEL_PATH:-$SCI_RESEARCH_ROOT/yangqingyu/model/qwen35_sft479_nemo_moe_visual}"
export DATASET="${DATASET:-$SCI_RESEARCH_ROOT/yangqingyu/rft/molt_science_legacy/train.jsonl}"
export SCIENCE_DATASET="${SCIENCE_DATASET:-$DATASET}"
export SCIENCE_LABEL_KEY="${SCIENCE_LABEL_KEY:-science_item}"
export MAX_SAMPLES="${MAX_SAMPLES:-3000}"
export N_SAMPLES_PER_PROMPT=8 ROLLOUT_BATCH_SIZE=8 TRAIN_BATCH_SIZE=64
export ACTOR_GPUS_PER_NODE=4 VLLM_TP_SIZE=4 FSDP_EP_SIZE=4 FSDP_CP_SIZE=4
export ROLLOUT_VLLM_GENERATE_BATCH_SIZE=6
export DYNAMIC_FILTERING_ENABLE=0 GROUP_LENGTH_PENALTY_ENABLE=0
export PARTIAL_ROLLOUT_ENABLE=1
export MAX_LEN=86016 MAX_NEW_TOKENS=86016
export MAX_IMAGES_PER_PROMPT=1 AGENT_EVAL_ENABLE_MODEL_IMAGE_INPUTS=1
export AGENT_ANSWER_TYPE="${AGENT_ANSWER_TYPE:-boxed}"
export AGENT_CONTEXT_COMPACTION=0 AGENT_EVAL_FORCE_FINALIZE=0
export AGENT_CONTEXT_COMPACT_THRESHOLD_TOKENS=86016
export TRAIN_MAX_TOKENS_PER_GPU="${TRAIN_MAX_TOKENS_PER_GPU:-25000}"
export SCIENCE_AGENT_SANDBOX_RESOURCE="${SCIENCE_AGENT_SANDBOX_RESOURCE:-10.12.111.143/ailab-ai4sgi}"
export SCIENCE_AGENT_ROOT
export SCIENCE_MOLT_AGENT_PATH="${SCIENCE_MOLT_AGENT_PATH:-$REPO_ROOT/examples/python/agents/science_legacy_react_sandbox.py}"
export TRAIN_SCRIPT="$DEEPSEEK_ROOT/molt_adapter/scripts/train_science_dsh_mixed_8gpu.sh"

export RJOB_NAME="${RJOB_NAME:-science-v1-sft479-dsh-mixed-geo-84k-bs8x8-48conc-visual-8gpu}"
export RUN_NAME="${RUN_NAME:-science_v1_sft479_dsh_mixed_geo_no_tis_no_sc_84k_batch8x8_48conc_visual_8gpu_4plus4_$(date +%Y%m%d_%H%M%S)}"
export RUN_ROOT="${RUN_ROOT:-$GPFS1_USER_ROOT/workspace/labs-molt/runs/$RUN_NAME}"
export MIXED_DATASET="${MIXED_DATASET:-$DEEPSEEK_ROOT/data/mixed/$RUN_NAME.jsonl}"
export DSH_FRACTION="${DSH_FRACTION:-0.5}" MIXED_DATA_SEED="${MIXED_DATA_SEED:-1}"
export SAVE_STEPS="${SAVE_STEPS:-100}"

exec bash "$SCRIPT_DIR/rjob_entry_science_legacy_react_h200_8gpu_4plus4.sh" \
  --actor.adam.lr 1e-6 --actor.freeze_moe_router --actor.freeze_visual_encoder --actor.log_entropy \
  --algo.advantage.estimator reinforce_baseline \
  --actor.loss_agg_mode token-mean --actor.eps_clip_low_high 0.2 0.27 \
  --algo.advantage.is_correction_level geo \
  --algo.advantage.is_correction_gating ratio \
  --algo.advantage.is_correction_mode mask --algo.advantage.is_correction_threshold 0.5 5.0 \
  --algo.advantage.score_centering_top_k 0 \
  --train.async_queue_size 2 --train.partial_rollout_enable \
  --train.rollout_dump_dir "" \
  --ckpt.save_steps "$SAVE_STEPS" --ckpt.save_hf \
  --rollout.temperature 1.0 --rollout.top_p 1.0 --train.max_epochs 1 \
  "$@"
