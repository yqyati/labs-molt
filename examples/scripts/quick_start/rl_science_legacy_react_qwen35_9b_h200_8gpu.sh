#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
GPFS2_ROOT="${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}"
SCI_RESEARCH_ROOT="${SCI_RESEARCH_ROOT:-$GPFS2_ROOT/scientific_reasoning_and_research}"
MODEL_PATH="${MODEL_PATH:-$GPFS2_ROOT/yangqingyu/model/Qwen3.5-9B}"
DATASET="${DATASET:-$SCI_RESEARCH_ROOT/yangqingyu/rft/molt_science_legacy/train.jsonl}"
RUN_NAME="${RUN_NAME:-science_agent_qwen35_9b_rl_$(date +%Y%m%d_%H%M%S)}"
SAVE_ROOT="${SAVE_ROOT:-/mnt/shared-storage-user/ai4sgi-gpfs1/yangqingyu/ckp/$RUN_NAME}"
MAX_SAMPLES="${MAX_SAMPLES:-3000}"
# Eight-H200 layout without a reference model: four GPUs for the vLLM rollout
# engine and four GPUs for the FSDP policy (dense Qwen3.5-9B, EP=1, DP=4).
RAY_NUM_GPUS="${RAY_NUM_GPUS:-8}"
ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-4}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-4}"
FSDP_EP_SIZE="${FSDP_EP_SIZE:-1}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
ROLLOUT_VLLM_GENERATE_BATCH_SIZE="${ROLLOUT_VLLM_GENERATE_BATCH_SIZE:-$ROLLOUT_BATCH_SIZE}"
# In this AutoModel/FSDP2 mesh, EP is a separate expert mesh and the training
# batch validator uses the four-rank FSDP data group. Use a batch divisible by
# dp_size=4; with micro-batch 1 this gives two accumulation steps.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
# A nonzero entropy coefficient materializes the full fp32 [B,S,V] logits
# tensor in PolicyActor.forward. With V=248320, long trajectories can add
# tens of GiB; keep entropy disabled for this memory-bound RL run.
ACTOR_ENTROPY_COEF="${ACTOR_ENTROPY_COEF:-0}"
if [[ "$ACTOR_ENTROPY_COEF" != "0" && "$ACTOR_ENTROPY_COEF" != "0.0" ]]; then
  echo "This long-context memory-bound run requires ACTOR_ENTROPY_COEF=0; refusing a full-vocab entropy allocation." >&2
  exit 1
fi
export SCIENCE_AGENT_MODEL_PATH="$MODEL_PATH"
export SCIENCE_AGENT_ROOT="${SCIENCE_AGENT_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/harness/science-agent}"
export SCIENCE_AGENT_GATEWAY_URL="${SCIENCE_AGENT_GATEWAY_URL:-http://100.102.21.127:8765}"
export SCIENCE_AGENT_JUDGE_BASE_URL="${SCIENCE_AGENT_JUDGE_BASE_URL:-${SCIENCE_AGENT_GATEWAY_URL%/}/v1/judge}"
export SCIENCE_AGENT_JUDGE_MODEL="${SCIENCE_AGENT_JUDGE_MODEL:-o3-mini}"
export SCIENCE_AGENT_TOOLS="${SCIENCE_AGENT_TOOLS:-search,visit,python,google_scholar}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}100.102.21.127"
export no_proxy="$NO_PROXY"
export OPENAI_COMPAT_USE_SYSTEM_PROMPT=1 AGENT_TOOL_BACKEND=remote AGENT_EVAL_FORCE_FINALIZE=1
# Do not retry deterministic finish_reason=length responses; keep API/network
# retries enabled. A truncated rollout is handled by Molt as truncated/dropped.
export LLM_RETRY_INCOMPLETE_OUTPUT=0
export MAX_LLM_CALL_PER_RUN="${MAX_LLM_CALL_PER_RUN:-16}" MAX_TOOL_CALLS_PER_RUN="${MAX_TOOL_CALLS_PER_RUN:-0}"
export TASK_TIMEOUT_SEC="${TASK_TIMEOUT_SEC:-3600}" OPENAI_REQUEST_TIMEOUT="${OPENAI_REQUEST_TIMEOUT:-1800}"
export WANDB_DISABLED=true RAY_USAGE_STATS_ENABLED=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOLT_MOE_RESHARD_AFTER_FWD="${MOLT_MOE_RESHARD_AFTER_FWD:-1}"
# The 131K context used by the first run let one long multi-turn trajectory
# exhaust the policy GPU during backward. Keep the full rollout contract, but
# bound this smoke to a trainable context and move Adam moments/step to CPU.
MAX_LEN="${MAX_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
export QWEN_API_MAX_OUTPUT_TOKENS="${QWEN_API_MAX_OUTPUT_TOKENS:-4096}"
export SCIENCE_TRAJECTORY_MAX_TOKENS="$MAX_LEN"
export SCIENCE_AGENT_SEARCH_MAX_RESULTS="${SCIENCE_AGENT_SEARCH_MAX_RESULTS:-6}"
export SCIENCE_AGENT_SEARCH_SNIPPET_CHARS="${SCIENCE_AGENT_SEARCH_SNIPPET_CHARS:-600}"
export SCIENCE_AGENT_SCHOLAR_MAX_RESULTS="${SCIENCE_AGENT_SCHOLAR_MAX_RESULTS:-6}"
export SCIENCE_AGENT_SCHOLAR_SNIPPET_CHARS="${SCIENCE_AGENT_SCHOLAR_SNIPPET_CHARS:-700}"
export SCIENCE_AGENT_READ_PAGE_MAX_EVIDENCE_BLOCKS="${SCIENCE_AGENT_READ_PAGE_MAX_EVIDENCE_BLOCKS:-4}"
export SCIENCE_AGENT_READ_PAGE_MODE="${SCIENCE_AGENT_READ_PAGE_MODE:-extractive}"
export SCIENCE_AGENT_READ_PAGE_MAX_CHARS="${SCIENCE_AGENT_READ_PAGE_MAX_CHARS:-6000}"
FSDP_OFFLOAD="${FSDP_OFFLOAD:-optimizer}"
CKPT_SAVE_STEPS="${CKPT_SAVE_STEPS:-100}"

test -f "$MODEL_PATH/config.json" || { echo "Missing model: $MODEL_PATH" >&2; exit 1; }
test -f "$DATASET" || { echo "Missing dataset: $DATASET" >&2; exit 1; }
test -f "$SCIENCE_AGENT_ROOT/inference/react_agent_openai_api_tools.py" || { echo "Missing Legacy ReAct harness under $SCIENCE_AGENT_ROOT" >&2; exit 1; }
if (( N_SAMPLES_PER_PROMPT < 2 )); then
  echo "reinforce_baseline requires N_SAMPLES_PER_PROMPT >= 2" >&2
  exit 1
fi
python3 -c 'import qwen_agent' || { echo "qwen-agent is required in the selected Molt environment" >&2; exit 1; }
python3 - "$DATASET" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
if not rows or not rows[0].get("prompt") or not isinstance(rows[0].get("science_item"), dict):
    raise SystemExit("dataset must contain prompt and science_item fields")
print(f"dataset rows={len(rows)} first_id={rows[0]['science_item'].get('question_id', 'unknown')}")
PY
python3 - "$MODEL_PATH/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    config = json.load(stream)
if config.get("model_type") != "qwen3_5":
    raise SystemExit(f"expected Qwen3.5 config, got model_type={config.get('model_type')!r}")
text_config = config.get("text_config", {})
if text_config.get("num_experts") or text_config.get("num_local_experts"):
    raise SystemExit("this 9B recipe expects the dense Qwen3.5 text config, but experts were found")
print("model config ok: qwen3_5 dense text backbone")
PY
python3 - "$SCIENCE_AGENT_GATEWAY_URL" <<'PY'
import json
import sys
from urllib.request import ProxyHandler, build_opener

url = sys.argv[1].rstrip("/") + "/v1/capabilities"
with build_opener(ProxyHandler({})).open(url, timeout=10) as response:
    capabilities = json.load(response)
if not capabilities.get("ok") or not capabilities.get("judge_proxy"):
    raise SystemExit(f"Gateway judge proxy unavailable: {capabilities}")
print("gateway reachable; judge proxy advertised")
PY
RAY_BIN="${RAY_BIN:-/usr/local/bin/ray}"
if ! "$RAY_BIN" status >/dev/null 2>&1; then
  "$RAY_BIN" start --head --num-gpus="$RAY_NUM_GPUS" --disable-usage-stats >/dev/null
  STARTED_RAY=1
else
  STARTED_RAY=0
fi
trap '[ "$STARTED_RAY" = "1" ] && "$RAY_BIN" stop --force >/dev/null 2>&1 || true' EXIT

cd "$REPO_ROOT"
echo "run=$RUN_NAME model=$MODEL_PATH dataset=$DATASET max_samples=$MAX_SAMPLES rollout_batch_size=$ROLLOUT_BATCH_SIZE rollout_vllm_generate_batch_size=$ROLLOUT_VLLM_GENERATE_BATCH_SIZE samples_per_prompt=$N_SAMPLES_PER_PROMPT train_batch_size=$TRAIN_BATCH_SIZE async_queue_size=2 routing_replay=0 partial_rollout=1 freeze_moe_router=1 reshard_after_fwd=$MOLT_MOE_RESHARD_AFTER_FWD ray_gpus=$RAY_NUM_GPUS actor_gpus=$ACTOR_GPUS_PER_NODE policy_dp=4 vllm_tp=$VLLM_TP_SIZE fsdp_ep=$FSDP_EP_SIZE ckpt_save_steps=$CKPT_SAVE_STEPS save_root=$SAVE_ROOT"
python3 -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$MODEL_PATH" \
  --data.prompt_dataset "$DATASET" --data.input_key prompt --data.label_key science_item \
  --data.apply_chat_template --data.max_samples "$MAX_SAMPLES" --data.max_len "$MAX_LEN" \
  --rollout.batch_size "$ROLLOUT_BATCH_SIZE" --rollout.vllm_generate_batch_size "$ROLLOUT_VLLM_GENERATE_BATCH_SIZE" --rollout.micro_batch_size 1 \
  --rollout.n_samples_per_prompt "$N_SAMPLES_PER_PROMPT" --rollout.max_new_tokens "$MAX_NEW_TOKENS" \
  --rollout.temperature 1.0 --rollout.top_p 1.0 --train.batch_size "$TRAIN_BATCH_SIZE" --train.micro_batch_size 1 --train.max_epochs 1 \
  --train.num_episodes 1 --train.async_queue_size 2 --train.colocate_fsdp_models --train.partial_rollout_enable \
  --actor.num_nodes 1 --actor.num_gpus_per_node "$ACTOR_GPUS_PER_NODE" \
  --vllm.num_engines 1 --vllm.tensor_parallel_size "$VLLM_TP_SIZE" --vllm.sync_backend nccl \
  --vllm.enable_auto_tool_choice --vllm.tool_call_parser "${VLLM_TOOL_CALL_PARSER:-qwen3_coder}" \
  --vllm.gpu_memory_utilization 0.8 --vllm.distributed_executor_backend mp \
  --fsdp.param_dtype bf16 --fsdp.attn_implementation te --fsdp.tp_size 1 --fsdp.ep_size "$FSDP_EP_SIZE" --fsdp.cp_size 1 \
  --fsdp.offload "$FSDP_OFFLOAD" \
  --actor.gradient_checkpoint full --actor.freeze_moe_router --actor.adam.lr 1e-6 --actor.entropy_coef "$ACTOR_ENTROPY_COEF" --actor.eps_clip_low_high 0.2 0.27 --actor.dual_clip 10.0 \
  --algo.advantage.estimator reinforce_baseline --algo.advantage.is_correction_level geo \
  --train.force_on_policy \
  --algo.dynamic_filtering_enable --algo.dynamic_filtering_range -0.1 1.1 \
  --algo.kl.estimator k1 --algo.kl.init_coef 0 \
  --reward.clip_range -10 10 \
  --train.agent_path "$REPO_ROOT/examples/python/agents/science_legacy_react.py" \
  --ckpt.output_dir "$SAVE_ROOT/hf" \
  --ckpt.path "$SAVE_ROOT/state" \
  --ckpt.save_steps "$CKPT_SAVE_STEPS" --ckpt.max_num 0 --logger.logging_steps 1 \
  --no-ckpt.save_optimizer \
  --logger.tensorboard_dir "$SAVE_ROOT/tensorboard"
