#!/usr/bin/env bash
# Keep the Vendor 8x8 recipe, but finalize at 56k without context compaction.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AGENT_CONTEXT_COMPACTION=0
export AGENT_EVAL_FORCE_FINALIZE=1
export AGENT_CONTEXT_COMPACT_THRESHOLD_TOKENS=57344
export RJOB_NAME="${RJOB_NAME:-science-agent-vendor3443-grpo8x8-lenpen-forcefinal56k}"
export RUN_NAME="${RUN_NAME:-science_agent_vendor3443_math40_grpo_batch8x8_length_penalty_forcefinal56k_8gpu_4plus4_$(date +%Y%m%d_%H%M%S)}"
exec bash "$SCRIPT_DIR/rjob_entry_science_vendor6000_sandbox_batch8x8_length_penalty_h200_8gpu_4plus4.sh" "$@"
