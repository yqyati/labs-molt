#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/labs-molt"
MODEL_PATH="/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/science_model/tool_models/hf_models/p1_rollout1_kimi2.6_subset_kimi_prompt_sft_qwen3.5_35b-0511-201941"
export MODEL_PATH MODEL_NAME=base_observation_mask
export OUTPUT_ROOT=/mnt/shared-storage-user/ai4sgi-gpfs1/yangqingyu/eval/molt_hle100/base_observation_mask
export CONVERT_ROOT=/mnt/shared-storage-user/ai4sgi-gpfs1/yangqingyu/eval/molt_hle100/base_observation_mask/hf_model
export TP=4 MAX_WORKERS=8
# Keep a single model response large enough to finish a difficult HLE item,
# while allowing the complete multi-turn ReAct trajectory to use up to 64K.
export MAX_NEW_TOKENS=8192 MAX_MODEL_LEN=98304
export SCIENCE_TRAJECTORY_MAX_TOKENS=65536
export AGENT_CONTEXT_COMPACTION=0 AGENT_OBSERVATION_MASKING=1 AGENT_OBSERVATION_MASKING_WINDOW=5
exec bash "$REPO_ROOT/evl/eval_molt_hle100_rollout.sh"
