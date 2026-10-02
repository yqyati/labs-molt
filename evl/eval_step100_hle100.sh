#!/usr/bin/env bash
# Evaluate the Molt actor checkpoint at global step 100 through Molt rollout.
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/shared-storage-user/ai4sgi-gpfs1/yangqingyu/ckp}"
MODEL_PATH="${MODEL_PATH:-$CKPT_ROOT/state/_actor/global_step100/model}"
MODEL_NAME="${MODEL_NAME:-step100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/shared-storage-user/ai4sgi-gpfs1/yangqingyu/eval/molt_hle100/$MODEL_NAME}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing step-100 model directory: $MODEL_PATH" >&2
  echo "Available actor checkpoints:" >&2
  find "$CKPT_ROOT/state/_actor" -mindepth 1 -maxdepth 1 -type d -name 'global_step*' -printf '  %f\n' 2>/dev/null | sort -V >&2 || true
  exit 1
fi

exec env \
  MODEL_PATH="$MODEL_PATH" \
  MODEL_NAME="$MODEL_NAME" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  bash "$WORKSPACE_ROOT/labs-molt/evl/eval_molt_hle100_rollout.sh"
