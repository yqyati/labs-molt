#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/labs-molt"
MODEL_PATH="/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/science_model/tool_models/hf_models/p1_rollout1_kimi2.6_subset_kimi_prompt_sft_qwen3.5_35b-0511-201941"
export MODEL_PATH MODEL_NAME=base_context_compact
export OUTPUT_ROOT=/mnt/shared-storage-user/ai4sgi-gpfs1/yangqingyu/eval/molt_hle100/base_context_compact
export CONVERT_ROOT=/mnt/shared-storage-user/ai4sgi-gpfs1/yangqingyu/eval/molt_hle100/base_context_compact/hf_model
export TP=4 MAX_WORKERS="${MAX_WORKERS:-16}"
export AGENT_CONTEXT_COMPACTION=1 AGENT_OBSERVATION_MASKING=0
exec bash "$REPO_ROOT/evl/eval_molt_hle100_rollout.sh"
