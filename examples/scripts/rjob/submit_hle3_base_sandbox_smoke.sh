#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT=/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/harness/science-agent-sandbox
CONFIG_FILE="${SANDBOX_LOCAL_ENV_FILE:-$HARNESS_ROOT/sandbox.local.env}"
test -f "$CONFIG_FILE" || { echo "Missing local Sandbox config: $CONFIG_FILE" >&2; exit 2; }
set -a
source "$CONFIG_FILE"
set +a
for name in OPEN_SANDBOX_GATEWAY_TOKEN API_KEY SUMMARY_MODEL_NAME SERPER_API_KEY; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing $name in $CONFIG_FILE; configure it before submitting the four-tool smoke." >&2
    exit 2
  fi
done

ENTRYPOINT=/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/labs-molt/examples/scripts/rjob/rjob_entry_hle3_base_sandbox_smoke.sh
RJOB_NAME="${RJOB_NAME:-hle3-base-sandbox-smoke-$(date +%Y%m%d-%H%M%S)}"
rjob submit \
  --name="$RJOB_NAME" \
  --charged-group=ailab_ai4sgi_pool --private-machine=group --priority=9 \
  --namespace=ailab-ai4sgi --gpu 4 --memory 192000 --cpu 16 -P 1 \
  --image=registry.h.pjlab.org.cn/ailab-ai4sgi-ailab_ai4sgi_pool/yqy:molt-cu13-runtime \
  --mount=gpfs://gpfs2/ai4sgi-gpfs2:/mnt/shared-storage-gpfs2/ai4sgi-gpfs2 \
  --mount=gpfs://gpfs1/ai4sgi-gpfs1:/mnt/shared-storage-user/ai4sgi-gpfs1 \
  -- bash "$ENTRYPOINT"
