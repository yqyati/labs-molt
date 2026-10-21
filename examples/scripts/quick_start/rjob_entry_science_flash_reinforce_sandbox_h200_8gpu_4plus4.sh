#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SCRIPT_DIR/rl_science_flash_reinforce_sandbox_h200_8gpu_4plus4.sh}"
export RJOB_NAME="${RJOB_NAME:-science-agent-flash-reinforce-8gpu-4plus4}"
export RUN_NAME="${RUN_NAME:-science_agent_flash_reinforce_8gpu_4plus4_$(date +%Y%m%d_%H%M%S)}"
exec bash "$SCRIPT_DIR/rjob_entry_science_legacy_react_sandbox_h200_8gpu_4plus4.sh" "$@"
