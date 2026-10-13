#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export SCIENCE_AGENT_ROOT="${SCIENCE_AGENT_ROOT:-${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}/yangqingyu/workspace/harness/science-agent-sandbox}"
export TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SCRIPT_DIR/rl_science_legacy_react_sandbox_h200_8gpu_4plus4.sh}"
export RJOB_NAME="${RJOB_NAME:-science-agent-sandbox-training-8gpu-4plus4}"
export RUN_NAME="${RUN_NAME:-science_agent_sandbox_training_8gpu_4plus4_$(date +%Y%m%d_%H%M%S)}"
exec bash "$REPO_ROOT/examples/scripts/quick_start/rjob_entry_science_legacy_react_h200_8gpu_4plus4.sh" "$@"
