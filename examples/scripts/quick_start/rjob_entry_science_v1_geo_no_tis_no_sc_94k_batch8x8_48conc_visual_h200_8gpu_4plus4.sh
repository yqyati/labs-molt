#!/usr/bin/env bash
# Science v1 RL from the HF SFT checkpoint with image support, geo correction,
# 8 prompts x 8 samples, with 48 concurrent trajectories (6 prompt groups x 8).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
GPFS2_ROOT="${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}"
SCI_RESEARCH_ROOT="${SCI_RESEARCH_ROOT:-$GPFS2_ROOT/scientific_reasoning_and_research}"

SFT_MODEL_PATH="${SFT_MODEL_PATH:-$SCI_RESEARCH_ROOT/science_model/tool_models/sft_qwen35_35b_distill_K3_tool_4set_k26_nemotron_science_k3_four_tools-128k-1019-233733/checkpoint/saved/hf_iter_0000479}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$SCI_RESEARCH_ROOT/model/Qwen3.5-35B-A3B}"
export MODEL_PATH="${MODEL_PATH:-$GPFS2_ROOT/yangqingyu/workspace/science_training_reports/models/qwen35_sft479_with_base_visual}"
python3 "$SCRIPT_DIR/prepare_qwen35_sft_visual_overlay.py" \
  --sft-model "$SFT_MODEL_PATH" --base-model "$BASE_MODEL_PATH" --output "$MODEL_PATH"
export DATASET="${DATASET:-$SCI_RESEARCH_ROOT/yangqingyu/rft/molt_science_legacy/train.jsonl}"
export MAX_SAMPLES="${MAX_SAMPLES:-3000}"
export N_SAMPLES_PER_PROMPT=8 ROLLOUT_BATCH_SIZE=8 TRAIN_BATCH_SIZE=64
export ROLLOUT_VLLM_GENERATE_BATCH_SIZE=6
export DYNAMIC_FILTERING_ENABLE=0 GROUP_LENGTH_PENALTY_ENABLE=0
export PARTIAL_ROLLOUT_ENABLE=1
export MAX_LEN=96256 MAX_NEW_TOKENS=96256
export MAX_IMAGES_PER_PROMPT=1 AGENT_EVAL_ENABLE_MODEL_IMAGE_INPUTS=1
export AGENT_CONTEXT_COMPACTION=0 AGENT_EVAL_FORCE_FINALIZE=0
export AGENT_CONTEXT_COMPACT_THRESHOLD_TOKENS=86016
export TRAIN_MAX_TOKENS_PER_GPU="${TRAIN_MAX_TOKENS_PER_GPU:-25000}"
export SCIENCE_AGENT_SANDBOX_RESOURCE="${SCIENCE_AGENT_SANDBOX_RESOURCE:-10.12.111.143/ailab-ai4sgi}"
export SCIENCE_AGENT_ROOT="${SCIENCE_AGENT_ROOT:-$GPFS2_ROOT/yangqingyu/workspace/harness/science-agent-sandbox}"
export TRAIN_SCRIPT="$SCRIPT_DIR/rl_science_vendor6000_sandbox_batch8x8_length_penalty_h200_8gpu_4plus4.sh"
export RJOB_NAME="${RJOB_NAME:-science-v1-hf479-geo-no-tis-no-sc-94k-bs8x8-48conc-visual}"
export RUN_NAME="${RUN_NAME:-science_v1_hf479_geo_no_tis_no_sc_94k_batch8x8_48conc_visual_8gpu_4plus4_$(date +%Y%m%d_%H%M%S)}"

exec bash "$SCRIPT_DIR/rjob_entry_science_legacy_react_h200_8gpu_4plus4.sh" \
  --actor.adam.lr 1e-6 --actor.freeze_moe_router --actor.freeze_visual_encoder \
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
