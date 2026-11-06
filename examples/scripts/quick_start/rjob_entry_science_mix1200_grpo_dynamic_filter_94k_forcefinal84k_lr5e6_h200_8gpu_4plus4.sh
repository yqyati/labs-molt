#!/usr/bin/env bash
# GRPO mixed-outcome filtering recipe with learning rate 5e-6.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RJOB_NAME="${RJOB_NAME:-science-agent-mix1200-grpo-filter-94k-f84k-lr5e6-bs8x8}"
export RUN_NAME="${RUN_NAME:-science_agent_mix1200_grpo_dynamic_filter_batch8x8_94k_forcefinal84k_lr5e6_8gpu_4plus4_$(date +%Y%m%d_%H%M%S)}"

exec bash "$SCRIPT_DIR/rjob_entry_science_mix1200_grpo_dynamic_filter_94k_forcefinal84k_h200_8gpu_4plus4.sh" \
  --actor.adam.lr 5e-6 "$@"
