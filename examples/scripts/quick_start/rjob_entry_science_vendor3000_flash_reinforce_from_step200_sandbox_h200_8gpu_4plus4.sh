#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/eval/molt_hle360_sandbox/hle360-rl15993171-step200-nomask-t06-120k-20261020-092840/hf_model/model}"
export DATASET="${DATASET:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/science_training_reports/vendor3000-grpo8-from-rl15993171-step200/train.jsonl}"
export TRAIN_SCRIPT="$SCRIPT_DIR/rl_science_vendor3000_flash_reinforce_from_step200_sandbox_h200_8gpu_4plus4.sh"
export RJOB_NAME="${RJOB_NAME:-science-agent-vendor3000-flashrl8-from-step200}"
export RUN_NAME="${RUN_NAME:-science_agent_vendor3000_flashrl8_from_step200_$(date +%Y%m%d_%H%M%S)}"
exec bash "$SCRIPT_DIR/rjob_entry_science_legacy_react_sandbox_h200_8gpu_4plus4.sh" "$@"
