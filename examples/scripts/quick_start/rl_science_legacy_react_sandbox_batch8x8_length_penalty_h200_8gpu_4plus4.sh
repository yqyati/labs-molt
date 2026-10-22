#!/usr/bin/env bash
set -euo pipefail

# Sandbox GRPO 8x8 with group-relative length penalties on successful trajectories.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
GPFS2_ROOT="${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}"
SCI_RESEARCH_ROOT="${SCI_RESEARCH_ROOT:-$GPFS2_ROOT/scientific_reasoning_and_research}"
export SCIENCE_AGENT_ROOT="${SCIENCE_AGENT_ROOT:-$GPFS2_ROOT/yangqingyu/workspace/harness/science-agent-sandbox}"
if [[ "${SCIENCE_TRAIN_SANDBOX_ENV_LOADED:-0}" != "1" ]]; then
  export SCIENCE_TRAIN_SANDBOX_ENV_LOADED=1
  exec bash "$SCIENCE_AGENT_ROOT/run_sandbox_harness.sh" bash "${BASH_SOURCE[0]}" "$@"
fi
unset SCIENCE_TRAIN_SANDBOX_ENV_LOADED

MODEL_PATH="${MODEL_PATH:-$SCI_RESEARCH_ROOT/science_model/tool_models/hf_models/p1_rollout1_kimi2.6_subset_kimi_prompt_sft_qwen3.5_35b-0511-201941}"
DATASET="${DATASET:-$SCI_RESEARCH_ROOT/yangqingyu/rft/molt_science_legacy/train.jsonl}"
RUN_NAME="${RUN_NAME:-science_agent_sandbox_grpo_batch8x8_length_penalty_8gpu_4plus4_$(date +%Y%m%d_%H%M%S)}"
GPFS1_USER_ROOT="${GPFS1_USER_ROOT:-/mnt/shared-storage-user/ai4sgi-gpfs1/yangqingyu}"
RUN_ROOT="${RUN_ROOT:-$GPFS1_USER_ROOT/workspace/labs-molt/runs/$RUN_NAME}"
LOG_ROOT="${LOG_ROOT:-$RUN_ROOT/logs}"
SAVE_ROOT="${SAVE_ROOT:-$RUN_ROOT/checkpoint}"
MAX_SAMPLES="${MAX_SAMPLES:-3000}"

mkdir -p "$LOG_ROOT" "$SAVE_ROOT"
# Reserve the state directory so reruns cannot overwrite existing checkpoints.
if [[ "${NODE_RANK:-0}" == "0" ]] && ! mkdir "$SAVE_ROOT/state"; then
  echo "Cannot create a fresh checkpoint directory: $SAVE_ROOT/state. Choose a new SAVE_ROOT." >&2
  exit 1
fi
exec > >(tee -a "$LOG_ROOT/entrypoint.log") 2>&1
echo "checkpoint_root=$SAVE_ROOT"

RAY_NUM_GPUS=8
ACTOR_GPUS_PER_NODE=4
VLLM_TP_SIZE=4
FSDP_EP_SIZE=4
FSDP_CP_SIZE=4
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
ROLLOUT_VLLM_GENERATE_BATCH_SIZE="${ROLLOUT_VLLM_GENERATE_BATCH_SIZE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
dynamic_filter_args=()
if [[ "${DYNAMIC_FILTERING_ENABLE:-1}" == "1" ]]; then
  dynamic_filter_args=(--algo.dynamic_filtering_enable --algo.dynamic_filtering_range -0.1 1.1)
fi
TRAIN_MAX_TOKENS_PER_GPU=17000
ACTOR_ENTROPY_COEF=0
MAX_LEN="${MAX_LEN:-65536}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-$MAX_LEN}"
FSDP_OFFLOAD=optimizer
CKPT_SAVE_STEPS=100

export SCIENCE_AGENT_MODEL_PATH="$MODEL_PATH"
export SCIENCE_AGENT_GATEWAY_URL="${SCIENCE_AGENT_GATEWAY_URL:-http://100.102.21.127:8765}"
export SCIENCE_AGENT_JUDGE_BASE_URL="${SCIENCE_AGENT_JUDGE_BASE_URL:-${API_BASE%/}}"
export SCIENCE_AGENT_JUDGE_MODEL="${SCIENCE_AGENT_JUDGE_MODEL:-o3-mini}"
export SCIENCE_AGENT_JUDGE_MAX_TOKENS="${SCIENCE_AGENT_JUDGE_MAX_TOKENS:-6000}"
export SCIENCE_AGENT_REWARD_MODE="${SCIENCE_AGENT_REWARD_MODE:-judge}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}100.102.21.127"
export no_proxy="$NO_PROXY"
export OPENAI_COMPAT_USE_SYSTEM_PROMPT=1 AGENT_TOOL_BACKEND=local AGENT_EVAL_FORCE_FINALIZE=0
export LLM_RETRY_INCOMPLETE_OUTPUT=0
export MAX_LLM_CALL_PER_RUN="${MAX_LLM_CALL_PER_RUN:-10000}" MAX_TOOL_CALLS_PER_RUN="${MAX_TOOL_CALLS_PER_RUN:-0}"
export TASK_TIMEOUT_SEC="${TASK_TIMEOUT_SEC:-9000}" OPENAI_REQUEST_TIMEOUT="${OPENAI_REQUEST_TIMEOUT:-1800}"
export QWEN_API_MAX_OUTPUT_TOKENS="$MAX_NEW_TOKENS" SCIENCE_TRAJECTORY_MAX_TOKENS="$MAX_LEN"
export AGENT_EVAL_ENABLE_MODEL_IMAGE_INPUTS="${AGENT_EVAL_ENABLE_MODEL_IMAGE_INPUTS:-1}"
export MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-1}"
export SCIENCE_AGENT_SEARCH_MAX_RESULTS="${SCIENCE_AGENT_SEARCH_MAX_RESULTS:-10}"
export SCIENCE_AGENT_SEARCH_SNIPPET_CHARS="${SCIENCE_AGENT_SEARCH_SNIPPET_CHARS:-0}"
export SCIENCE_AGENT_SCHOLAR_MAX_RESULTS="${SCIENCE_AGENT_SCHOLAR_MAX_RESULTS:-0}"
export SCIENCE_AGENT_SCHOLAR_SNIPPET_CHARS="${SCIENCE_AGENT_SCHOLAR_SNIPPET_CHARS:-0}"
# Keep Python, web requests, and judging in the same task-owned sandbox.
export PYTHONPATH="$REPO_ROOT:$SCIENCE_AGENT_ROOT:$SCIENCE_AGENT_ROOT/inference${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_SESSION_BACKEND=gateway_sandbox SCIENCE_AGENT_TOOL_HTTP_BACKEND=gateway_sandbox
export SCIENCE_AGENT_SANDBOX_HTTP_STRICT=true
export SCIENCE_AGENT_SANDBOX_IMAGE="${SANDBOX_IMAGE:-registry.h.pjlab.org.cn/ailab-ai4sgi-ailab_ai4sgi_pool/yqy:science-agent-cpu-20261010}"
export SCIENCE_AGENT_SANDBOX_CPU="${SANDBOX_CPU:-2}" SCIENCE_AGENT_SANDBOX_MEMORY="${SANDBOX_MEMORY:-4Gi}"
export SCIENCE_AGENT_SANDBOX_TIMEOUT="${SANDBOX_TIMEOUT:-4500}"
export SCIENCE_AGENT_READ_PAGE_MODE=file
export PYTHON_TOOL_RETURN_MAX_CHARS="${PYTHON_TOOL_RETURN_MAX_CHARS:-6000}"
export SCIENCE_AGENT_READ_PAGE_MAX_CHARS="${SCIENCE_AGENT_READ_PAGE_MAX_CHARS:-6000}"
export SCIENCE_AGENT_READ_PAGE_MAX_EVIDENCE_BLOCKS="${SCIENCE_AGENT_READ_PAGE_MAX_EVIDENCE_BLOCKS:-4}"
export SCIENCE_AGENT_TRACE_FILE="${SCIENCE_AGENT_TRACE_FILE:-$LOG_ROOT/sandbox_trace.jsonl}"
test -f "$MODEL_PATH/config.json" || { echo "Missing model: $MODEL_PATH" >&2; exit 1; }
test -f "$DATASET" || { echo "Missing dataset: $DATASET" >&2; exit 1; }
test -f "$SCIENCE_AGENT_ROOT/inference/sandbox_gateway.py" || { echo "Missing sandbox gateway" >&2; exit 1; }
test -f "$SCIENCE_AGENT_ROOT/inference/react_agent_openai_api_tools.py" || { echo "Missing Legacy ReAct harness" >&2; exit 1; }
if [[ "${NODE_RANK:-0}" == "0" ]]; then
  python3 -u "$SCIENCE_AGENT_ROOT/sandbox_smoke.py" --search --judge --page-file \
    --http-url https://r.jina.ai/http://example.com 2>&1 | tee "$LOG_ROOT/sandbox_preflight.log"
fi

export WANDB_DISABLED=true RAY_USAGE_STATS_ENABLED=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOLT_MOE_RESHARD_AFTER_FWD=0
export MOLT_MOE_DISPATCHER=hybridep
export MOLT_DEFER_GRAD_SYNC=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

RAY_BIN="${RAY_BIN:-/usr/local/bin/ray}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-${MASTER_SOCKET_ADDR:-127.0.0.1}}"
NODE_LOG_DIR="$LOG_ROOT/nodes/$NODE_RANK"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/molt_ray_${RUN_NAME##*_}_${NODE_RANK}}"
mkdir -p "$NODE_LOG_DIR" "$RAY_TEMP_DIR"
{
  date -Ins
  printf 'hostname='; hostname -f 2>/dev/null || hostname
  printf 'NODE_RANK=%s NODE_COUNT=%s PROC_PER_NODE=%s MASTER_ADDR=%s RAY_ADDRESS=%s\n' \
    "$NODE_RANK" "${NODE_COUNT:-unknown}" "${PROC_PER_NODE:-unknown}" "$MASTER_ADDR" "${RAY_ADDRESS:-unset}"
  printf 'layout=rollout4+policy4 policy_world=4 tp=1 cp=%s ep=%s vllm_tp=%s dispatcher=%s max_len=%s\n' \
    "$FSDP_CP_SIZE" "$FSDP_EP_SIZE" "$VLLM_TP_SIZE" "$MOLT_MOE_DISPATCHER" "$MAX_LEN"
  env | sort | grep -E '^(RANK|LOCAL_RANK|WORLD_SIZE|MASTER_|NODE_|PROC_PER_NODE|NCCL_|TORCH_NCCL_|TORCH_DISTRIBUTED_DEBUG|MOLT_MOE_|MOLT_DIST_)' || true
  nvidia-smi -q || true
} >"$NODE_LOG_DIR/node-info.log" 2>&1

if command -v nvidia-smi >/dev/null 2>&1; then
  (while true; do
    date -Ins
    nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv
    sleep 15
  done) >"$NODE_LOG_DIR/gpu-monitor.log" 2>&1 &
  GPU_MONITOR_PID=$!
fi

persist_ray_logs() {
  local session target
  for session in "$RAY_TEMP_DIR"/session_*; do
    [[ -d "$session/logs" && ! -L "$session" ]] || continue
    target="$LOG_ROOT/ray-runtime/node-$NODE_RANK/${session##*/}"
    mkdir -p "$target" && cp -a "$session/logs" "$target/" 2>>"$NODE_LOG_DIR/ray-runtime-copy.log" || true
  done
}
cleanup() {
  local status=$?
  [[ -n "${GPU_MONITOR_PID:-}" ]] && kill "$GPU_MONITOR_PID" 2>/dev/null || true
  [[ -n "${MEMORY_MONITOR_PID:-}" ]] && kill "$MEMORY_MONITOR_PID" 2>/dev/null || true
  [[ -n "${RAY_LOG_MONITOR_PID:-}" ]] && kill "$RAY_LOG_MONITOR_PID" 2>/dev/null || true
  dmesg -T >"$NODE_LOG_DIR/kernel-exit.log" 2>&1 || true
  persist_ray_logs
  "$RAY_BIN" stop --force >>"$NODE_LOG_DIR/ray-stop.log" 2>&1 || true
  persist_ray_logs
  printf 'entrypoint_exit_status=%s node_rank=%s time=%s\n' "$status" "$NODE_RANK" "$(date -Ins)" >>"$LOG_ROOT/exit-status.log" || true
  return "$status"
}
trap cleanup EXIT

(while true; do
  date -Ins
  for counter in /sys/fs/cgroup/memory.events /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.max \
    /sys/fs/cgroup/memory/memory.oom_control /sys/fs/cgroup/memory/memory.usage_in_bytes \
    /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    if [[ -r "$counter" ]]; then printf '%s\n' "$counter"; cat "$counter" || true; fi
  done
  sleep 2
done) >>"$NODE_LOG_DIR/host-memory.log" 2>&1 &
MEMORY_MONITOR_PID=$!
(while true; do persist_ray_logs; sleep 10; done) &
RAY_LOG_MONITOR_PID=$!

if [[ "$NODE_RANK" == "0" ]]; then
  "$RAY_BIN" stop --force >/dev/null 2>&1 || true
  "$RAY_BIN" start --head --node-ip-address="$MASTER_ADDR" --port="$RAY_HEAD_PORT" --num-gpus=8 \
    --temp-dir="$RAY_TEMP_DIR" --disable-usage-stats >>"$NODE_LOG_DIR/ray-start.log" 2>&1
  export RAY_ADDRESS="$MASTER_ADDR:$RAY_HEAD_PORT"
else
  export RAY_ADDRESS="$MASTER_ADDR:$RAY_HEAD_PORT"
  "$RAY_BIN" stop --force >/dev/null 2>&1 || true
  joined=0
  for _ in $(seq 1 60); do
    if "$RAY_BIN" start --address="$RAY_ADDRESS" --num-gpus=8 --temp-dir="$RAY_TEMP_DIR" --disable-usage-stats \
      >>"$NODE_LOG_DIR/ray-start.log" 2>&1; then joined=1; break; fi
    sleep 2
  done
  (( joined == 1 )) || { echo "Could not join Ray head at $RAY_ADDRESS" >&2; exit 1; }
  RAY_WORKER_NODE=1
fi


if [[ "${RAY_WORKER_NODE:-0}" == "1" ]]; then
  echo "Ray worker joined $RAY_ADDRESS; waiting for rank-0 training driver."
  while "$RAY_BIN" status --address="$RAY_ADDRESS" >/dev/null 2>&1; do sleep 30; done
  exit 0
fi

cd "$REPO_ROOT"
echo "run=$RUN_NAME sandbox_image=$SCIENCE_AGENT_SANDBOX_IMAGE sandbox_cpu=$SCIENCE_AGENT_SANDBOX_CPU sandbox_memory=$SCIENCE_AGENT_SANDBOX_MEMORY max_new_tokens=$MAX_NEW_TOKENS layout=4+4 max_len=$MAX_LEN max_samples=$MAX_SAMPLES rollout_batch_size=$ROLLOUT_BATCH_SIZE train_batch_size=$TRAIN_BATCH_SIZE train_max_tokens_per_gpu=$TRAIN_MAX_TOKENS_PER_GPU dispatcher=$MOLT_MOE_DISPATCHER routing_replay=0 partial_rollout=1 policy_gpus=4 vllm_tp=4 fsdp_ep=4 fsdp_cp=$FSDP_CP_SIZE"
"$RAY_BIN" status --address="$RAY_ADDRESS" >"$LOG_ROOT/ray-cluster-status.txt" 2>&1 || true
env | sort | grep -E '^(RANK|LOCAL_RANK|WORLD_SIZE|MASTER_|NODE_|PROC_PER_NODE|NCCL_|TORCH_NCCL_|TORCH_DISTRIBUTED_DEBUG|MOLT_MOE_|MOLT_DIST_)' >"$LOG_ROOT/distributed-environment.log" || true
set +e
python3 -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$MODEL_PATH" \
  --data.prompt_dataset "$DATASET" --data.input_key prompt --data.label_key science_item --data.apply_chat_template \
  --data.max_samples "$MAX_SAMPLES" --data.max_len "$MAX_LEN" \
  --data.max_images_per_prompt "$MAX_IMAGES_PER_PROMPT" \
  --rollout.batch_size "$ROLLOUT_BATCH_SIZE" --rollout.vllm_generate_batch_size "$ROLLOUT_VLLM_GENERATE_BATCH_SIZE" --rollout.micro_batch_size 1 \
  --rollout.n_samples_per_prompt "$N_SAMPLES_PER_PROMPT" --rollout.max_new_tokens "$MAX_NEW_TOKENS" --rollout.temperature 1.0 --rollout.top_p 1.0 \
  --train.batch_size "$TRAIN_BATCH_SIZE" --train.micro_batch_size 1 --train.max_tokens_per_gpu "$TRAIN_MAX_TOKENS_PER_GPU" --train.max_epochs 1 \
  --train.num_episodes 1 --train.async_queue_size 2 --train.colocate_fsdp_models --train.partial_rollout_enable \
  --actor.num_nodes 1 --actor.num_gpus_per_node 4 \
  --vllm.num_engines 1 --vllm.tensor_parallel_size 4 --vllm.sync_backend nccl --vllm.enable_auto_tool_choice --vllm.tool_call_parser qwen3_coder \
  --vllm.gpu_memory_utilization 0.8 --vllm.distributed_executor_backend mp --vllm.enable_prefix_caching \
  --fsdp.param_dtype bf16 --fsdp.attn_implementation te --fsdp.tp_size 1 --fsdp.ep_size 4 --fsdp.cp_size 4 --fsdp.offload "$FSDP_OFFLOAD" \
  --actor.gradient_checkpoint full --actor.adam.lr 1e-6 --actor.entropy_coef 0 --actor.loss_mode ppo --actor.eps_clip_low_high 0.2 0.27 \
  --algo.advantage.estimator reinforce_baseline --algo.advantage.is_correction_level geo --train.force_on_policy \
  "${dynamic_filter_args[@]}" --algo.kl.estimator k1 --algo.kl.init_coef 0 --reward.clip_range -10 10 \
  --reward.group_length_penalty.enable --reward.group_length_penalty.min_pass_rate 0.5 \
  --reward.group_length_penalty.quantile 0.5 --reward.group_length_penalty.max_penalty 0.1 \
  --reward.group_length_penalty.tolerance 0.2 --reward.group_length_penalty.saturation 1.0 \
  --train.agent_path "$REPO_ROOT/examples/python/agents/science_legacy_react_sandbox.py" \
  --ckpt.output_dir "$SAVE_ROOT/hf" --ckpt.path "$SAVE_ROOT/state" --ckpt.save_steps "$CKPT_SAVE_STEPS" --ckpt.max_num 0 \
  --no-ckpt.save_optimizer --logger.logging_steps 1 --logger.tensorboard_dir "$RUN_ROOT/tensorboard" \
  "$@" \
  2>&1 | tee -a "$LOG_ROOT/train.log"
status=${PIPESTATUS[0]}
set -e
printf 'train_exit_status=%s time=%s\n' "$status" "$(date -Ins)" | tee -a "$LOG_ROOT/exit-status.log"
exit "$status"
