#!/usr/bin/env bash
# Temporarily route this entrypoint to the 96k/86k Vendor FlashREINFORCE recipe.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export N_SAMPLES_PER_PROMPT=1
export ROLLOUT_BATCH_SIZE=64
export ROLLOUT_VLLM_GENERATE_BATCH_SIZE=48
export TRAIN_BATCH_SIZE=64
export DYNAMIC_FILTERING_ENABLE=0
export RJOB_NAME="${RJOB_NAME:-science-agent-vendor3443-flash64x1-96k-f86k}"
export RUN_NAME="${RUN_NAME:-science_agent_vendor3443_math40_flash_reinforce_batch64x1_96k_forcefinal86k_8gpu_4plus4_$(date +%Y%m%d_%H%M%S)}"
exec bash "$SCRIPT_DIR/rjob_entry_science_vendor6000_sandbox_batch8x8_length_penalty_96k_forcefinal86k_h200_8gpu_4plus4.sh" \
  --algo.advantage.estimator flash_reinforce \
  --algo.advantage.is_correction_level seq \
  --algo.advantage.is_correction_gating binary_kl \
  --algo.advantage.is_correction_threshold "${TRUST_REGION_DELTA:-5e-3}" \
  --actor.loss_agg_mode seq-mean-token-mean \
  "$@"
