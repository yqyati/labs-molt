#!/usr/bin/env bash
# Group-normalized GRPO with mixed-outcome filtering and token-mean PPO loss.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export N_SAMPLES_PER_PROMPT=8
export ROLLOUT_BATCH_SIZE=8
export TRAIN_BATCH_SIZE=64
export DYNAMIC_FILTERING_ENABLE=1
export RJOB_NAME="${RJOB_NAME:-science-agent-mix1200-grpo-filter-94k-f84k-bs8x8}"
export RUN_NAME="${RUN_NAME:-science_agent_mix1200_grpo_dynamic_filter_batch8x8_94k_forcefinal84k_8gpu_4plus4_$(date +%Y%m%d_%H%M%S)}"

exec bash "$SCRIPT_DIR/rjob_entry_science_mix1200_sandbox_batch8x8_96k_forcefinal86k_h200_8gpu_4plus4.sh" \
  --actor.adam.lr 1e-6 \
  --algo.advantage.estimator grpo \
  --algo.advantage.no_whiten \
  --algo.dynamic_filtering_enable --algo.dynamic_filtering_range 0 1 \
  --actor.loss_agg_mode token-mean --actor.eps_clip_low_high 0.2 0.27 \
  "$@"
