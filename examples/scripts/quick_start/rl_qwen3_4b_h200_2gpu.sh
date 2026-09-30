#!/usr/bin/env bash
# Two-GPU H200 smoke/full-data launcher for Qwen3-4B dense text RL.
#
# The dataset is staged offline by prepare_dapo.py into:
#   .tmp/qwen3_4b_sanity/{train,eval}
# It contains 1,460 train rows and 60 eval rows with Molt's prompt/reward_model
# schema. This script does not configure a W&B key, so W&B is disabled.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/model/Qwen3-4B-Base}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/.tmp/qwen3_4b_sanity}"
PROMPT_DATASET="${PROMPT_DATASET:-$DATA_DIR/train}"
EVAL_DATASET="${EVAL_DATASET:-$DATA_DIR/eval}"
SAVE_ROOT="${SAVE_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/labs-molt/outputs/qwen3_4b_h200_2gpu}"

test -f "$MODEL_PATH/config.json" || { echo "MODEL_PATH missing or not a Qwen checkpoint: $MODEL_PATH" >&2; exit 1; }
test -f "$PROMPT_DATASET/dataset_info.json" || { echo "PROMPT_DATASET missing: $PROMPT_DATASET" >&2; exit 1; }
test -f "$EVAL_DATASET/dataset_info.json" || { echo "EVAL_DATASET missing: $EVAL_DATASET" >&2; exit 1; }

GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
ACTOR_GPUS="${ACTOR_GPUS:-1}"
VLLM_TP="${VLLM_TP:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-1460}"
NUM_EPISODES="${NUM_EPISODES:-1}"

export HF_HOME="${HF_HOME:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/hf-cache}"
export TOKENIZERS_PARALLELISM=true
export RAY_USAGE_STATS_ENABLED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
export MAX_AGENT_TURNS=1
RAY_BIN="${RAY_BIN:-/usr/local/bin/ray}"
test -x "$RAY_BIN" || { echo "Ray CLI not executable: $RAY_BIN" >&2; exit 1; }

if ! "$RAY_BIN" status >/dev/null 2>&1; then
  "$RAY_BIN" start --head --num-gpus="$GPUS_PER_NODE" --disable-usage-stats >/dev/null
  STARTED_RAY=1
else
  STARTED_RAY=0
fi
trap '[ "$STARTED_RAY" = "1" ] && "$RAY_BIN" stop --force >/dev/null 2>&1 || true' EXIT

cd "$REPO_ROOT"
python3 -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$MODEL_PATH" \
  --data.prompt_dataset "$PROMPT_DATASET" \
  --data.input_key prompt \
  --data.label_key reward_model \
  --data.apply_chat_template \
  --data.max_samples "$MAX_SAMPLES" \
  --data.max_len 8192 \
  --rollout.batch_size 2 \
  --rollout.vllm_generate_batch_size 2 \
  --rollout.micro_batch_size 1 \
  --rollout.n_samples_per_prompt 2 \
  --rollout.max_new_tokens 4096 \
  --rollout.temperature 1.0 \
  --train.batch_size 2 \
  --train.micro_batch_size 1 \
  --train.max_epochs 1 \
  --train.num_episodes "$NUM_EPISODES" \
  --train.async_queue_size 1 \
  --train.colocate_fsdp_models \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node "$ACTOR_GPUS" \
  --ref.num_nodes 1 \
  --ref.num_gpus_per_node "$ACTOR_GPUS" \
  --vllm.num_engines 1 \
  --vllm.tensor_parallel_size "$VLLM_TP" \
  --vllm.sync_backend nccl \
  --vllm.gpu_memory_utilization 0.8 \
  --vllm.distributed_executor_backend mp \
  --fsdp.param_dtype bf16 \
  --fsdp.attn_implementation te \
  --fsdp.tp_size 1 \
  --fsdp.ep_size 1 \
  --fsdp.cp_size 1 \
  --fsdp.packing_samples \
  --actor.gradient_checkpoint full \
  --actor.adam.lr 1e-6 \
  --actor.eps_clip_low_high 0.2 0.27 \
  --actor.dual_clip 10.0 \
  --algo.advantage.estimator reinforce_baseline \
  --algo.advantage.is_correction_level geo \
  --algo.advantage.is_correction_threshold 0.99 1.01 \
  --algo.kl.use_loss \
  --algo.kl.estimator k2 \
  --algo.kl.init_coef 0.001 \
  --reward.clip_range -10 10 \
  --train.agent_path "$REPO_ROOT/examples/python/agents/math.py" \
  --eval.dataset "$EVAL_DATASET" \
  --eval.steps 5 \
  --eval.n_samples_per_prompt 1 \
  --ckpt.output_dir "$SAVE_ROOT/hf" \
  --ckpt.path "$SAVE_ROOT/state" \
  --ckpt.save_steps 1 \
  --logger.logging_steps 1
