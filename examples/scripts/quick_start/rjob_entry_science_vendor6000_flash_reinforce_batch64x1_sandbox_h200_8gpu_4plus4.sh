#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export SCIENCE_AGENT_ROOT="${SCIENCE_AGENT_ROOT:-${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}/yangqingyu/workspace/harness/science-agent-sandbox}"
export MODEL_PATH="${MODEL_PATH:-${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}/scientific_reasoning_and_research/science_model/tool_models/hf_models/p1_rollout1_kimi2.6_subset_kimi_prompt_sft_qwen3.5_35b-0511-201941}"
export DATASET="${DATASET:-${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}/yangqingyu/workspace/science_training_reports/vendor_balanced_math40_difficulty6_7_20261023/train.jsonl}"
export MAX_SAMPLES="${MAX_SAMPLES:-3443}"
export TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SCRIPT_DIR/rl_science_vendor6000_flash_reinforce_batch64x1_sandbox_h200_8gpu_4plus4.sh}"
export RJOB_NAME="${RJOB_NAME:-science-agent-vendor3443-math40-flashrl-bs64x1-w48}"
export RUN_NAME="${RUN_NAME:-science_agent_vendor3443_math40_flash_reinforce_batch64x1_8gpu_4plus4_$(date +%Y%m%d_%H%M%S)}"
exec bash "$REPO_ROOT/examples/scripts/quick_start/rjob_entry_science_legacy_react_h200_8gpu_4plus4.sh" "$@"
