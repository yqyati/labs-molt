#!/usr/bin/env bash
# Evaluate an HF checkpoint on AMO with the shared agent and isolated Sandbox tools.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOLT_REPO="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
WORKSPACE_ROOT="$(dirname "$MOLT_REPO")"
GPFS2_ROOT="${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}"
SCI_RESEARCH_ROOT="$GPFS2_ROOT/scientific_reasoning_and_research"
HARNESS_ROOT="${SCIENCE_AGENT_ROOT:-$WORKSPACE_ROOT/harness/science-agent-sandbox}"
AMO_SOURCE="${AMO_SOURCE:-$SCI_RESEARCH_ROOT/jiangyichen/p1-shared-2/eval/math/amobench.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$GPFS2_ROOT/yangqingyu/eval/molt_amo_sandbox/amo-$(date +%Y%m%d-%H%M%S)}"
EVAL_DATASET="$OUTPUT_ROOT/amobench_molt.jsonl"
mkdir -p "$OUTPUT_ROOT"
python3 "$MOLT_REPO/examples/python/utils/prepare_amobench.py" --source "$AMO_SOURCE" --output "$EVAL_DATASET"
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a consolidated HF model directory}"
test -f "$MODEL_PATH/config.json"
compgen -G "$MODEL_PATH/*.safetensors" >/dev/null || { echo "Expected consolidated HF safetensors: $MODEL_PATH" >&2; exit 1; }
TP="${TP:-2}"
MAX_WORKERS="${MAX_WORKERS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-122880}"
export SCIENCE_TRAJECTORY_MAX_TOKENS="${SCIENCE_TRAJECTORY_MAX_TOKENS:-122880}"
MAX_NEW_TOKENS="$SCIENCE_TRAJECTORY_MAX_TOKENS"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
export SCIENCE_AGENT_ROOT="$HARNESS_ROOT"
export PYTHONPATH="$MOLT_REPO:$HARNESS_ROOT:$HARNESS_ROOT/inference${PYTHONPATH:+:$PYTHONPATH}"
WHEEL_DIR="${QWEN_AGENT_WHEEL_DIR:-$SCI_RESEARCH_ROOT/yangqingyu/rft/offline_wheels/qwen_agent_py312_cp312_clean}"
INSTALLER="${QWEN_AGENT_INSTALLER:-$SCI_RESEARCH_ROOT/yangqingyu/rft/offline_wheels/install_qwen_agent_offline.sh}"
bash "$INSTALLER" "$WHEEL_DIR"
export SCIENCE_AGENT_MODEL_PATH="$MODEL_PATH"
export SCIENCE_AGENT_GATEWAY_URL="${SCIENCE_AGENT_GATEWAY_URL:-http://100.102.21.127:8765}"
export SCIENCE_AGENT_EVAL_BENCHMARK=amobench
export SCIENCE_AGENT_NEXT_HLE_EVAL_PROTOCOL=0
export QWEN_API_MAX_OUTPUT_TOKENS="$MAX_NEW_TOKENS"
export SCIENCE_AGENT_JUDGE_BASE_URL="${SCIENCE_AGENT_JUDGE_BASE_URL:-${API_BASE%/}}"
export SCIENCE_AGENT_JUDGE_MODEL="${SCIENCE_AGENT_JUDGE_MODEL:-bailian/deepseek-v4.1-flash}"
export SCIENCE_AGENT_JUDGE_THINKING="${SCIENCE_AGENT_JUDGE_THINKING:-enabled}"
export SCIENCE_AGENT_JUDGE_MAX_TOKENS="${SCIENCE_AGENT_JUDGE_MAX_TOKENS:-6000}"
export SCIENCE_AGENT_REWARD_MODE=judge
export SCIENCE_AGENT_TOOLS="${SCIENCE_AGENT_TOOLS:-search,visit,python,google_scholar}"
export SCIENCE_AGENT_READ_PAGE_MODE=file
export PYTHON_TOOL_RETURN_MAX_CHARS="${PYTHON_TOOL_RETURN_MAX_CHARS:-50000}"
export SCIENCE_AGENT_READ_PAGE_MAX_CHARS="${SCIENCE_AGENT_READ_PAGE_MAX_CHARS:-6000}"
export SCIENCE_AGENT_READ_PAGE_MAX_EVIDENCE_BLOCKS="${SCIENCE_AGENT_READ_PAGE_MAX_EVIDENCE_BLOCKS:-4}"
export SCIENCE_AGENT_SEARCH_MAX_RESULTS="${SCIENCE_AGENT_SEARCH_MAX_RESULTS:-10}"
export SCIENCE_AGENT_SEARCH_SNIPPET_CHARS=0
export SCIENCE_AGENT_SCHOLAR_MAX_RESULTS=0 SCIENCE_AGENT_SCHOLAR_SNIPPET_CHARS=0
export AGENT_OBSERVATION_MASKING="${AGENT_OBSERVATION_MASKING:-0}"
export AGENT_CONTEXT_COMPACTION=0
export OPENAI_COMPAT_USE_SYSTEM_PROMPT=1 AGENT_TOOL_BACKEND=local AGENT_EVAL_FORCE_FINALIZE=0
export SCIENCE_AGENT_TRACE_FILE="$OUTPUT_ROOT/sandbox_trace.jsonl"
export SCIENCE_AGENT_SANDBOX_RECOVERY="${SCIENCE_AGENT_SANDBOX_RECOVERY:-1}"
export TASK_TIMEOUT_SEC="${TASK_TIMEOUT_SEC:-9000}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export WANDB_DISABLED=true RAY_USAGE_STATS_ENABLED=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
python3 -u "$HARNESS_ROOT/sandbox_smoke.py" --search --judge 2>&1 | tee "$OUTPUT_ROOT/sandbox_preflight.log"

cd "$MOLT_REPO"
echo "Molt AMO eval: rows=50 model=$MODEL_PATH dataset=$EVAL_DATASET output=$OUTPUT_ROOT tp=$TP workers=$MAX_WORKERS" >&2
python3 -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$MODEL_PATH" \
  --data.prompt_dataset "$EVAL_DATASET" \
  --data.input_key prompt \
  --data.label_key science_item \
  --data.apply_chat_template \
  --data.max_len "$MAX_MODEL_LEN" \
  --data.max_images_per_prompt "${MAX_IMAGES_PER_PROMPT:-0}" \
  --data.dataloader_num_workers 0 \
  --rollout.batch_size "$MAX_WORKERS" \
  --rollout.n_samples_per_prompt 1 \
  --rollout.max_new_tokens "$MAX_NEW_TOKENS" \
  --rollout.temperature "$TEMPERATURE" \
  --rollout.top_p "$TOP_P" \
  --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
  --vllm.num_engines 1 \
  --vllm.tensor_parallel_size "$TP" \
  --vllm.gpu_memory_utilization 0.8 \
  --vllm.distributed_executor_backend mp \
  --vllm.enable_auto_tool_choice \
  --vllm.tool_call_parser "${VLLM_TOOL_CALL_PARSER:-qwen3_coder}" \
  --train.agent_path "$MOLT_REPO/examples/python/agents/science_legacy_react_sandbox.py" \
  --train.rollout_dump_dir "$OUTPUT_ROOT/rollout_dump" \
  --eval.dataset "$EVAL_DATASET" \
  --eval.batch_size "$MAX_WORKERS" \
  --eval.n_samples_per_prompt 1 \
  --eval.max_new_tokens "$MAX_NEW_TOKENS" \
  --eval.temperature "$TEMPERATURE" \
  --eval.top_p "$TOP_P" \
  --eval.eval_only \
  --logger.tensorboard_dir "$OUTPUT_ROOT/tensorboard" \
  2>&1 | tee "$OUTPUT_ROOT/eval.log"
