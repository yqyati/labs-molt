#!/usr/bin/env bash
set -euo pipefail

source /etc/profile.d/ssh-init.sh

GPU_COUNT=4
ENTRYPOINT="/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/labs-molt/examples/scripts/rjob/rjob_entry_hle100_base_observation_mask.sh"
RJOB_NAME="${RJOB_NAME:-hle100-base-observation-mask-20261008}"

rjob submit \
  --name="$RJOB_NAME" \
  --charged-group=ailab_ai4sgi_pool \
  --private-machine=group \
  --priority=9 \
  --namespace=ailab-ai4sgi \
  --gpu "$GPU_COUNT" \
  --memory 192000 \
  --cpu 16 \
  -P 1 \
  --image=registry.h.pjlab.org.cn/ailab-ai4sgi-ailab_ai4sgi_pool/yqy:molt-cu13-runtime \
  --mount=gpfs://gpfs2/ai4sgi-gpfs2:/mnt/shared-storage-gpfs2/ai4sgi-gpfs2 \
  --mount=gpfs://gpfs1/ai4sgi-gpfs1:/mnt/shared-storage-user/ai4sgi-gpfs1 \
  -- bash "$ENTRYPOINT"
