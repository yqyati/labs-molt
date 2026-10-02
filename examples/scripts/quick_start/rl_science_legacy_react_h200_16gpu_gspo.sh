#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
GPFS2_ROOT="${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}"
SCI_RESEARCH_ROOT="${SCI_RESEARCH_ROOT:-$GPFS2_ROOT/scientific_reasoning_and_research}"
MODEL_PATH="${MODEL_PATH:-$SCI_RESEARCH_ROOT/science_model/tool_models/hf_models/p1_rollout1_kimi2.6_subset_kimi_prompt_sft_qwen3.5_35b-0511-201941}"
DATASET="${DATASET:-$SCI_RESEARCH_ROOT/yangqingyu/rft/molt_science_legacy/train.jsonl}"
RUN_NAME="${RUN_NAME:-science_agent_async_training_gspo_16gpu_$(date +%Y%m%d_%H%M%S)}"
SAVE_ROOT="${SAVE_ROOT:-/mnt/shared-storage-user/ai4sgi-gpfs1/yangqingyu/ckp/$RUN_NAME}"
MAX_SAMPLES="${MAX_SAMPLES:-3000}"

# Two-node layout: one node owns the 8-GPU vLLM engine; the other owns the
# 8-GPU policy. The known-working baseline uses EP=8 x CP=1 x DP=1.
# Context parallelism is intentionally disabled until its MoE collective order
# is validated for this Qwen3.5 + NeMo AutoModel path.
RAY_NUM_GPUS="${RAY_NUM_GPUS:-16}"
ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-8}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-8}"
FSDP_EP_SIZE="${FSDP_EP_SIZE:-8}"
FSDP_CP_SIZE="${FSDP_CP_SIZE:-1}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
ROLLOUT_VLLM_GENERATE_BATCH_SIZE="${ROLLOUT_VLLM_GENERATE_BATCH_SIZE:-$ROLLOUT_BATCH_SIZE}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
TRAIN_MAX_TOKENS_PER_GPU="${TRAIN_MAX_TOKENS_PER_GPU:-16192}"
# A nonzero entropy coefficient materializes the full fp32 [B,S,V] logits
# tensor in PolicyActor.forward. With V=248320, long trajectories can add
# tens of GiB; keep entropy disabled for this memory-bound GSPO run.
ACTOR_ENTROPY_COEF="${ACTOR_ENTROPY_COEF:-0}"
if [[ "$ACTOR_ENTROPY_COEF" != "0" && "$ACTOR_ENTROPY_COEF" != "0.0" ]]; then
  echo "This long-context memory-bound run requires ACTOR_ENTROPY_COEF=0; refusing a full-vocab entropy allocation." >&2
  exit 1
fi
MAX_LEN="${MAX_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
FSDP_OFFLOAD="${FSDP_OFFLOAD:-optimizer}"
CKPT_SAVE_STEPS="${CKPT_SAVE_STEPS:-100}"

export SCIENCE_AGENT_MODEL_PATH="$MODEL_PATH"
export SCIENCE_AGENT_ROOT="${SCIENCE_AGENT_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/harness/science-agent}"
export SCIENCE_AGENT_GATEWAY_URL="${SCIENCE_AGENT_GATEWAY_URL:-http://100.102.21.127:8765}"
export SCIENCE_AGENT_JUDGE_BASE_URL="${SCIENCE_AGENT_JUDGE_BASE_URL:-${SCIENCE_AGENT_GATEWAY_URL%/}/v1/judge}"
export SCIENCE_AGENT_JUDGE_MODEL="${SCIENCE_AGENT_JUDGE_MODEL:-o3-mini}"
export SCIENCE_AGENT_TOOLS="${SCIENCE_AGENT_TOOLS:-search,visit,python,google_scholar}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}100.102.21.127"
export no_proxy="$NO_PROXY"
export OPENAI_COMPAT_USE_SYSTEM_PROMPT=1 AGENT_TOOL_BACKEND=remote AGENT_EVAL_FORCE_FINALIZE=1
export LLM_RETRY_INCOMPLETE_OUTPUT=0
export MAX_LLM_CALL_PER_RUN="${MAX_LLM_CALL_PER_RUN:-16}" MAX_TOOL_CALLS_PER_RUN="${MAX_TOOL_CALLS_PER_RUN:-0}"
export TASK_TIMEOUT_SEC="${TASK_TIMEOUT_SEC:-3600}" OPENAI_REQUEST_TIMEOUT="${OPENAI_REQUEST_TIMEOUT:-1800}"
export QWEN_API_MAX_OUTPUT_TOKENS="$MAX_NEW_TOKENS" SCIENCE_TRAJECTORY_MAX_TOKENS="$MAX_LEN"
export SCIENCE_AGENT_SEARCH_MAX_RESULTS="${SCIENCE_AGENT_SEARCH_MAX_RESULTS:-6}"
export SCIENCE_AGENT_SEARCH_SNIPPET_CHARS="${SCIENCE_AGENT_SEARCH_SNIPPET_CHARS:-600}"
export SCIENCE_AGENT_SCHOLAR_MAX_RESULTS="${SCIENCE_AGENT_SCHOLAR_MAX_RESULTS:-6}"
export SCIENCE_AGENT_SCHOLAR_SNIPPET_CHARS="${SCIENCE_AGENT_SCHOLAR_SNIPPET_CHARS:-700}"
export SCIENCE_AGENT_READ_PAGE_MAX_EVIDENCE_BLOCKS="${SCIENCE_AGENT_READ_PAGE_MAX_EVIDENCE_BLOCKS:-4}"
export SCIENCE_AGENT_READ_PAGE_MODE="${SCIENCE_AGENT_READ_PAGE_MODE:-extractive}"
export SCIENCE_AGENT_READ_PAGE_MAX_CHARS="${SCIENCE_AGENT_READ_PAGE_MAX_CHARS:-6000}"
export WANDB_DISABLED=true RAY_USAGE_STATS_ENABLED=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOLT_MOE_RESHARD_AFTER_FWD="${MOLT_MOE_RESHARD_AFTER_FWD:-1}"

# RJob -P 2 starts this entrypoint once per node. Build one Ray cluster across
# the two 8-GPU nodes; without this, ray.init() creates a local 8-GPU head on
# the first node and Molt rejects the required 16-GPU placement.
RAY_BIN="${RAY_BIN:-/usr/local/bin/ray}"
RAY_NODE_GPUS="${RAY_NODE_GPUS:-8}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-${MASTER_SOCKET_ADDR:-127.0.0.1}}"
if [[ "$NODE_RANK" == "0" ]]; then
  "$RAY_BIN" stop --force >/dev/null 2>&1 || true
  "$RAY_BIN" start --head --node-ip-address="$MASTER_ADDR" --port="$RAY_HEAD_PORT" \
    --num-gpus="$RAY_NODE_GPUS" --disable-usage-stats >/dev/null
  export RAY_ADDRESS="$MASTER_ADDR:$RAY_HEAD_PORT"
  STARTED_RAY_HEAD=1
else
  export RAY_ADDRESS="$MASTER_ADDR:$RAY_HEAD_PORT"
  "$RAY_BIN" stop --force >/dev/null 2>&1 || true
  joined=0
  for _ in $(seq 1 60); do
    if "$RAY_BIN" start --address="$RAY_ADDRESS" --num-gpus="$RAY_NODE_GPUS" \
      --disable-usage-stats >/dev/null 2>&1; then
      joined=1
      break
    fi
    sleep 2
  done
  (( joined == 1 )) || { echo "Could not join Ray head at $RAY_ADDRESS" >&2; exit 1; }
  RAY_WORKER_NODE=1
fi
trap '[ "${STARTED_RAY_HEAD:-0}" = "1" ] && "$RAY_BIN" stop --force >/dev/null 2>&1 || true' EXIT

test -f "$MODEL_PATH/config.json" || { echo "Missing model: $MODEL_PATH" >&2; exit 1; }
test -f "$DATASET" || { echo "Missing dataset: $DATASET" >&2; exit 1; }
test -f "$SCIENCE_AGENT_ROOT/inference/react_agent_openai_api_tools.py" || {
  echo "Missing Legacy ReAct harness under $SCIENCE_AGENT_ROOT" >&2
  exit 1
}
(( N_SAMPLES_PER_PROMPT >= 2 )) || { echo "reinforce_baseline requires N_SAMPLES_PER_PROMPT >= 2" >&2; exit 1; }

if [[ "${RAY_WORKER_NODE:-0}" == "1" ]]; then
  echo "Ray worker joined $RAY_ADDRESS; waiting for the rank-0 training driver."
  while "$RAY_BIN" status --address="$RAY_ADDRESS" >/dev/null 2>&1; do
    sleep 30
  done
  exit 0
fi

cd "$REPO_ROOT"
echo "run=$RUN_NAME model=$MODEL_PATH dataset=$DATASET max_samples=$MAX_SAMPLES rollout_batch_size=$ROLLOUT_BATCH_SIZE vllm_generate_batch_size=$ROLLOUT_VLLM_GENERATE_BATCH_SIZE samples_per_prompt=$N_SAMPLES_PER_PROMPT train_batch_size=$TRAIN_BATCH_SIZE train_max_tokens_per_gpu=$TRAIN_MAX_TOKENS_PER_GPU routing_replay=0 partial_rollout=1 loss_mode=gspo is_correction=geo policy_gpus=$ACTOR_GPUS_PER_NODE vllm_tp=$VLLM_TP_SIZE fsdp_ep=$FSDP_EP_SIZE fsdp_cp=$FSDP_CP_SIZE max_len=$MAX_LEN max_new_tokens=$MAX_NEW_TOKENS save_root=$SAVE_ROOT"
python3 -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$MODEL_PATH" \
  --data.prompt_dataset "$DATASET" --data.input_key prompt --data.label_key science_item \
  --data.apply_chat_template --data.max_samples "$MAX_SAMPLES" --data.max_len "$MAX_LEN" \
  --rollout.batch_size "$ROLLOUT_BATCH_SIZE" --rollout.vllm_generate_batch_size "$ROLLOUT_VLLM_GENERATE_BATCH_SIZE" --rollout.micro_batch_size 1 \
  --rollout.n_samples_per_prompt "$N_SAMPLES_PER_PROMPT" --rollout.max_new_tokens "$MAX_NEW_TOKENS" \
  --rollout.temperature 1.0 --rollout.top_p 1.0 --train.batch_size "$TRAIN_BATCH_SIZE" --train.micro_batch_size 1 --train.max_tokens_per_gpu "$TRAIN_MAX_TOKENS_PER_GPU" --train.max_epochs 1 \
  --train.num_episodes 1 --train.async_queue_size 2 --train.colocate_fsdp_models --train.partial_rollout_enable \
  --actor.num_nodes 1 --actor.num_gpus_per_node "$ACTOR_GPUS_PER_NODE" \
  --vllm.num_engines 1 --vllm.tensor_parallel_size "$VLLM_TP_SIZE" --vllm.sync_backend nccl \
  --vllm.enable_auto_tool_choice --vllm.tool_call_parser "${VLLM_TOOL_CALL_PARSER:-qwen3_coder}" \
  --vllm.gpu_memory_utilization 0.8 --vllm.distributed_executor_backend mp \
  --fsdp.param_dtype bf16 --fsdp.attn_implementation te --fsdp.tp_size 1 --fsdp.ep_size "$FSDP_EP_SIZE" --fsdp.cp_size "$FSDP_CP_SIZE" \
  --fsdp.offload "$FSDP_OFFLOAD" \
  --actor.gradient_checkpoint full --actor.freeze_moe_router --actor.adam.lr 1e-6 --actor.entropy_coef "$ACTOR_ENTROPY_COEF" --actor.loss_mode gspo --actor.eps_clip_low_high 0.2 0.27 \
  --algo.advantage.estimator reinforce_baseline --algo.advantage.is_correction_level geo \
  --train.force_on_policy --algo.dynamic_filtering_enable --algo.dynamic_filtering_range -0.1 1.1 \
  --algo.kl.estimator k1 --algo.kl.init_coef 0 --reward.clip_range -10 10 \
  --train.agent_path "$REPO_ROOT/examples/python/agents/science_legacy_react.py" \
  --ckpt.output_dir "$SAVE_ROOT/hf" --ckpt.path "$SAVE_ROOT/state" \
  --ckpt.save_steps "$CKPT_SAVE_STEPS" --ckpt.max_num 0 --logger.logging_steps 1 \
  --no-ckpt.save_optimizer --logger.tensorboard_dir "$SAVE_ROOT/tensorboard"
