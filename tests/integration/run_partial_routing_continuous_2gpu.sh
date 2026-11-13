#!/usr/bin/env bash
# Two-GPU RL smoke: local-expert actor, one vLLM; preserve async partial rollout.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/science_model/tool_models/hf_models/p1_rollout1_kimi2.6_subset_kimi_prompt_sft_qwen3.5_35b-0511-201941}"
export RUN_DIR="${RUN_DIR:-$REPO_ROOT/outputs/partial-routing-continuous-2gpu-$(date +%Y%m%d-%H%M%S)}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_USAGE_STATS_ENABLED=0
export WANDB_MODE="${WANDB_MODE:-disabled}"
export MOLT_AUDIT_PARTIAL_ROUTING=1
export MOLT_TEST_WEIGHT_VALUE_AUDIT=1
export PYTHONPATH="$REPO_ROOT/tests/integration:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export RAY_DEDUP_LOGS=0
export MAX_AGENT_TURNS=1
# Portable local experts: no grouped_gemm extension or EP dispatcher required.
export MOLT_MOE_EXPERTS=torch
export MOLT_LINEAR_BACKEND=torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - <<'PY'
import os
from pathlib import Path
import torch
assert torch.cuda.device_count() >= 2, 'This MoE test needs two visible GPUs (one actor + one vLLM).'
assert (Path(os.environ['MODEL_PATH']) / 'config.json').is_file(), 'MODEL_PATH is not accessible.'
for i in range(2):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB')
PY
mkdir -p "$RUN_DIR"
python -u "$REPO_ROOT/tests/integration/watch_partial_routing.py" "$RUN_DIR/audit.json" -- python -u "$REPO_ROOT/tests/integration/launch_partial_routing_audit.py" \
  --actor.model_name_or_path "$MODEL_PATH" \
  --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
  --actor.lora_dim 8 --actor.lora_alpha 16 \
  --actor.lora_target_modules '*.q_proj' '*.v_proj' \
  --actor.freeze_visual_encoder --actor.freeze_moe_router \
  --actor.gradient_checkpoint full --actor.adam.lr 1e-5 \
  --data.prompt_dataset "$REPO_ROOT/tests/integration/data/partial_routing_long.jsonl" \
  --data.input_key prompt --data.label_key label --data.apply_chat_template \
  --data.max_samples 128 --data.max_len 32768 \
  --rollout.batch_size 4 --rollout.n_samples_per_prompt 4 \
  --rollout.num_runners 16 --rollout.vllm_generate_batch_size 16 --rollout.micro_batch_size 1 \
  --rollout.max_new_tokens 24576 --rollout.temperature 1.0 --rollout.top_p 1.0 \
  --train.batch_size 16 --train.micro_batch_size 1 \
  --train.max_epochs 1 --train.num_episodes 1 --train.async_queue_size 2 \
  --train.partial_rollout_enable --train.routing_replay \
  --train.agent_path "$REPO_ROOT/tests/integration/partial_routing_continuous_agent.py" \
  --train.check_weight_update_equal --train.rollout_dump_dir "$RUN_DIR/rollouts" \
  --algo.advantage.estimator reinforce_baseline \
  --algo.advantage.is_correction_level geo \
  --algo.advantage.is_correction_gating ratio --algo.advantage.is_correction_mode mask \
  --algo.advantage.is_correction_threshold 0.5 5.0 \
  --algo.kl.init_coef 0 \
  --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
  --vllm.distributed_executor_backend mp --vllm.sync_backend nccl \
  --vllm.gpu_memory_utilization 0.8 --vllm.enable_prefix_caching \
  --fsdp.param_dtype bf16 --fsdp.attn_implementation flash_attention_2 \
  --fsdp.tp_size 1 --fsdp.ep_size 1 --fsdp.cp_size 1 \
  --fsdp.offload optimizer \
  --ckpt.path "$RUN_DIR/checkpoints" --ckpt.disable_final_save \
  --ckpt.save_steps -1 --logger.logging_steps 1 \
  "$@" 2>&1 | tee "$RUN_DIR/train.log"
printf 'Training finished. Inspect gradients, rewards, refit and preemption logs: %s\n' "$RUN_DIR/train.log"
printf 'Rollouts: %s\n' "$RUN_DIR/rollouts"
