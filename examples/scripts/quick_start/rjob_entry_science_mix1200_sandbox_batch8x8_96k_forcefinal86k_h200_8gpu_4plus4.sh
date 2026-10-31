#!/usr/bin/env bash
# Use the mixed science/math dataset with the existing 96k training recipe.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPFS2_ROOT="${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}"
export DATASET="${DATASET:-$GPFS2_ROOT/yangqingyu/workspace/science_training_reports/science_mix1200_20261031/train.jsonl}"
export MAX_SAMPLES=1200
export RJOB_NAME="${RJOB_NAME:-science-agent-mix1200-96k-f86k-bs8x8}"
export RUN_NAME="${RUN_NAME:-science_agent_mix1200_batch8x8_96k_forcefinal86k_8gpu_4plus4_$(date +%Y%m%d_%H%M%S)}"

exec bash "$SCRIPT_DIR/rjob_entry_science_vendor6000_sandbox_batch8x8_length_penalty_96k_forcefinal86k_h200_8gpu_4plus4.sh" "$@"
