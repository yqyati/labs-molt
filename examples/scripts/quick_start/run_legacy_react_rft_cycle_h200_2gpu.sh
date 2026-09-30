#!/usr/bin/env bash
# Configuration wrapper. The Python CLI owns rollout, vLLM, SFT, and epoch chaining.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/science_model/tool_models/hf_models/p1_rollout1_kimi2.6_subset_kimi_prompt_sft_qwen3.5_35b-0511-201941}"
INPUT="${INPUT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/yangqingyu/rft/molt_science_legacy/train.jsonl}"
BASE_URL="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
RUN_ROOT="${RUN_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/yangqingyu/rft/runs/legacy_react_rft}"
EPOCHS="${EPOCHS:-${NUM_CYCLES:-1}}"
TARGET_ACCEPTED="${TARGET_ACCEPTED:-800}"
export SCIENCE_AGENT_ROOT="${SCIENCE_AGENT_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/liuzhuo/science-agent}"
export SCIENCE_AGENT_GATEWAY_URL="${SCIENCE_AGENT_GATEWAY_URL:-http://100.102.21.127:8765}"
export SCIENCE_AGENT_JUDGE_BASE_URL="${SCIENCE_AGENT_JUDGE_BASE_URL:-${SCIENCE_AGENT_GATEWAY_URL%/}/v1/judge}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}100.102.21.127"
export no_proxy="$NO_PROXY"
export QWEN_API_MAX_OUTPUT_TOKENS="${QWEN_API_MAX_OUTPUT_TOKENS:-98000}"
export MAX_OUTPUT_TOKENS="$QWEN_API_MAX_OUTPUT_TOKENS"
export AGENT_TOOL_BACKEND=remote OPENAI_COMPAT_USE_SYSTEM_PROMPT=1 AGENT_EVAL_FORCE_FINALIZE=1 WANDB_DISABLED=true RAY_USAGE_STATS_ENABLED=0
cd "$REPO_ROOT"
args=(cycle --input "$INPUT" --model-path "$MODEL_PATH" --run-root "$RUN_ROOT"
  --round "${ROUND:-$(date +%Y%m%d_%H%M%S)}" --epochs "$EPOCHS"
  --base-url "$BASE_URL" --served-model "${SERVED_MODEL_NAME:-local-policy}"
  --vllm-tp "${VLLM_TP_SIZE:-2}" --vllm-max-model-len "${VLLM_MAX_MODEL_LEN:-131072}"
  # The 35B MoE checkpoint needs several minutes for TP=2 loading, CUDA graph
  # capture, and kernel warmup on GPFS before /v1/models becomes ready.
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}" --vllm-wait-seconds "${VLLM_WAIT_SECONDS:-900}"
  --vllm-tool-call-parser "${VLLM_TOOL_CALL_PARSER:-qwen3_coder}"
  --target-accepted "$TARGET_ACCEPTED" --workers "${ROLLOUT_WORKERS:-32}" --rollout-n "${ROLLOUT_N:-4}"
  --post-target-grace-sec "${POST_TARGET_GRACE_SEC:-600}" --max-attempts "${ROLLOUT_MAX_ATTEMPTS:-0}"
  --min-score "${RFT_MIN_SCORE:-0.0}" --gpus "${GPUS_PER_NODE:-2}" --max-len "${MAX_LEN:-131072}"
  --judge-base-url "${SCIENCE_AGENT_JUDGE_BASE_URL:?Set SCIENCE_AGENT_JUDGE_BASE_URL for formal RFT filtering}"
  --judge-api-key "${SCIENCE_AGENT_JUDGE_API_KEY:-EMPTY}" --judge-model "${SCIENCE_AGENT_JUDGE_MODEL:-gpt-5.4-mini}"
  --judge-min-confidence "${JUDGE_MIN_CONFIDENCE:-0.8}" --judge-timeout "${JUDGE_TIMEOUT:-180}"
  --sft-max-samples "${SFT_MAX_SAMPLES:-0}" --save-steps "${SAVE_STEPS:-50}"
  --train-max-epochs "${TRAIN_MAX_EPOCHS:-1}" --train-batch-size "${TRAIN_BATCH_SIZE:-2}" --micro-batch-size "${MICRO_BATCH_SIZE:-1}"
  --attn-implementation "${FSDP_ATTN_IMPLEMENTATION:-te}" --lr "${LR:-1e-6}")
if [[ -n "${OPENAI_API_BASE:-}" ]]; then args+=(--external-endpoint); fi
exec setsid python3 -u -m molt.cli.legacy_react_rft "${args[@]}"
