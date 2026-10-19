#!/usr/bin/env bash
set -euo pipefail

# Single-rollout FlashREINFORCE with the science recipe's model, tools and GPU layout.
# The batch baseline and sequence weighting assume unsegmented training trajectories.
# Usage: bash examples/scripts/quick_start/rl_science_flash_reinforce_sandbox_h200_8gpu_4plus4.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-science_agent_flash_reinforce_8gpu_4plus4_$(date +%Y%m%d_%H%M%S)}"
export N_SAMPLES_PER_PROMPT=1
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export ROLLOUT_VLLM_GENERATE_BATCH_SIZE="${ROLLOUT_VLLM_GENERATE_BATCH_SIZE:-16}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
# Group-based dynamic filtering requires multiple samples per prompt.
export DYNAMIC_FILTERING_ENABLE=0

# Trailing CLI arguments override the shared recipe's algorithm defaults.
exec bash "$SCRIPT_DIR/rl_science_legacy_react_sandbox_h200_8gpu_4plus4.sh" \
  --algo.advantage.estimator flash_reinforce \
  --algo.advantage.is_correction_level seq \
  --algo.advantage.is_correction_gating binary_kl \
  --algo.advantage.is_correction_threshold "${TRUST_REGION_DELTA:-5e-3}" \
  --actor.loss_agg_mode seq-mean-token-mean \
  "$@"
