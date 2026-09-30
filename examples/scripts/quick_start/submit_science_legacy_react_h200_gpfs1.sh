#!/usr/bin/env bash
set -euo pipefail

# Submit the science-agent run with GPFS2 for code/model/data inputs and GPFS1
# for all durable run outputs (checkpoint state, HF weights, TensorBoard).
source /etc/profile.d/ssh-init.sh

GPU_COUNT="${GPU_COUNT:-8}"
ENTRYPOINT="${ENTRYPOINT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/labs-molt/examples/scripts/quick_start/rjob_entry_science_legacy_react_h200_8gpu.sh}"
RJOB_NAME="${RJOB_NAME:-science-agent-async-training-test}"

rjob submit \
  --name="$RJOB_NAME" \
  --charged-group=ailab_ai4sgi_pool \
  --private-machine=group \
  --priority=9 \
  --namespace=ailab-ai4sgi \
  --gpu "$GPU_COUNT" \
  --memory 768000 \
  --cpu 48 \
  -P 1 \
  -e DISTRIBUTED_JOB=true \
  -e GPFS1_ROOT=/mnt/shared-storage-user/ai4sgi-gpfs1 \
  -e GPFS1_USER_ROOT=/mnt/shared-storage-user/ai4sgi-gpfs1/yangqingyu \
  -e SAVE_ROOT=/mnt/shared-storage-user/ai4sgi-gpfs1/yangqingyu/ckp \
  --image=registry.h.pjlab.org.cn/ailab-ai4sgi-ailab_ai4sgi_pool/yqy:molt-cu13-runtime \
  --mount=gpfs://gpfs2/ai4sgi-gpfs2:/mnt/shared-storage-gpfs2/ai4sgi-gpfs2 \
  --mount=gpfs://gpfs1/ai4sgi-gpfs1:/mnt/shared-storage-user/ai4sgi-gpfs1 \
  -- bash "$ENTRYPOINT"
