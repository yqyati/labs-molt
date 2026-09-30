#!/usr/bin/env bash
# In-container RJob entrypoint for the 16-GPU async science RL run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPFS2_ROOT="${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}"
GPFS1_ROOT="${GPFS1_ROOT:-/mnt/shared-storage-user/ai4sgi-gpfs1}"
YANGQINGYU_MOUNT_ROOT="${YANGQINGYU_MOUNT_ROOT:-$GPFS2_ROOT/yangqingyu}"
GPFS1_USER_ROOT="${GPFS1_USER_ROOT:-$GPFS1_ROOT/yangqingyu}"
SAVE_ROOT="${SAVE_ROOT:-$GPFS1_USER_ROOT/ckp}"
SCI_RESEARCH_ROOT="${SCI_RESEARCH_ROOT:-$GPFS2_ROOT/scientific_reasoning_and_research}"
WHEEL_DIR="${QWEN_AGENT_WHEEL_DIR:-$SCI_RESEARCH_ROOT/yangqingyu/rft/offline_wheels/qwen_agent_py312_cp312_clean}"
INSTALLER="${QWEN_AGENT_INSTALLER:-$SCI_RESEARCH_ROOT/yangqingyu/rft/offline_wheels/install_qwen_agent_offline.sh}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/examples/scripts/quick_start/rl_science_legacy_react_h200_16gpu.sh}"
SCIENCE_AGENT_ROOT="${SCIENCE_AGENT_ROOT:-$SCI_RESEARCH_ROOT/liuzhuo/science-agent}"
RJOB_NAME="${RJOB_NAME:-science-agent-async-training-test-16gpu}"
MODEL_PATH="${MODEL_PATH:-$SCI_RESEARCH_ROOT/science_model/tool_models/hf_models/p1_rollout1_kimi2.6_subset_kimi_prompt_sft_qwen3.5_35b-0511-201941}"
DATASET="${DATASET:-$SCI_RESEARCH_ROOT/yangqingyu/rft/molt_science_legacy/train.jsonl}"

MISSING=0
require_path() {
  local kind="$1" path="$2"
  if [[ ! -e "$path" ]]; then
    printf 'MISSING %-8s %s\n' "$kind" "$path" >&2
    MISSING=1
  else
    printf 'FOUND   %-8s %s\n' "$kind" "$path"
  fi
}

echo "Required RJob mounts:"
echo "  gpfs://gpfs2/ai4sgi-gpfs2/yangqingyu:$YANGQINGYU_MOUNT_ROOT"
echo "  gpfs://gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research:$SCI_RESEARCH_ROOT"
echo "  gpfs://gpfs1/ai4sgi-gpfs1/yangqingyu:$GPFS1_USER_ROOT"
require_path workspace_mount "$YANGQINGYU_MOUNT_ROOT/workspace/labs-molt"
require_path research_mount "$SCI_RESEARCH_ROOT"
require_path checkpoint_mount "$GPFS1_USER_ROOT"
require_path repository "$REPO_ROOT"
require_path qwen_wheels "$WHEEL_DIR"
require_path qwen_installer "$INSTALLER"
require_path model_config "$MODEL_PATH/config.json"
require_path dataset "$DATASET"
require_path legacy_harness "$SCIENCE_AGENT_ROOT/inference/react_agent_openai_api_tools.py"
require_path train_script "$TRAIN_SCRIPT"
if (( MISSING )); then
  echo "RJob preflight failed. Add the GPFS1/GPFS2 mounts above; training was not started." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MOLT_PATH="$REPO_ROOT" SCIENCE_AGENT_ROOT MODEL_PATH DATASET GPFS2_ROOT GPFS1_ROOT YANGQINGYU_MOUNT_ROOT GPFS1_USER_ROOT SCI_RESEARCH_ROOT SAVE_ROOT
export PYTHONPATH="$REPO_ROOT:$SCIENCE_AGENT_ROOT:$SCIENCE_AGENT_ROOT/inference${PYTHONPATH:+:$PYTHONPATH}"

echo "Installing qwen-agent from local wheels only: $WHEEL_DIR"
PYTHON_BIN="$PYTHON_BIN" bash "$INSTALLER" "$WHEEL_DIR"
"$PYTHON_BIN" - <<'PY'
import qwen_agent
from react_agent_openai_api_tools import MultiTurnReactAgent
print(f"qwen-agent import ok: {qwen_agent.__file__}")
print(f"Legacy ReAct import ok: {MultiTurnReactAgent.__module__}")
PY

cd "$REPO_ROOT"
export RUN_NAME="${RUN_NAME:-science_agent_async_training_test_16gpu_$(date +%Y%m%d_%H%M%S)}"
echo "rjob_name=$RJOB_NAME run_name=$RUN_NAME"
echo "Starting training entrypoint: $TRAIN_SCRIPT"
exec bash "$TRAIN_SCRIPT"
