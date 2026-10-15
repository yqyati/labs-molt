#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/labs-molt
export HARNESS_ROOT=/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/harness/science-agent-sandbox
export MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/science_model/tool_models/hf_models/p1_rollout1_kimi2.6_subset_kimi_prompt_sft_qwen3.5_35b-0511-201941}"
export MODEL_NAME="${MODEL_NAME:-base-sandbox}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/eval/molt_hle100_sandbox/$(date +%Y%m%d-%H%M%S)}"
export CONVERT_ROOT="${CONVERT_ROOT:-$OUTPUT_ROOT/hf_model}"
export TP=4 MAX_WORKERS="${MAX_WORKERS:-16}"
export SCIENCE_TRAJECTORY_MAX_TOKENS="${SCIENCE_TRAJECTORY_MAX_TOKENS:-65536}"
export MAX_NEW_TOKENS="$SCIENCE_TRAJECTORY_MAX_TOKENS" MAX_MODEL_LEN=98304
export AGENT_CONTEXT_COMPACTION=0 AGENT_OBSERVATION_MASKING=0
export AGENT_TOOL_BACKEND=local SCIENCE_AGENT_REWARD_MODE=judge
mkdir -p "$OUTPUT_ROOT"
export SCIENCE_AGENT_TRACE_FILE="$OUTPUT_ROOT/sandbox_trace.jsonl"
exec bash "$HARNESS_ROOT/run_sandbox_harness.sh" bash "$REPO_ROOT/evl/eval_molt_hle100_sandbox.sh"
