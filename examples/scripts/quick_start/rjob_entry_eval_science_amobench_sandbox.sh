#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCIENCE_AGENT_ROOT="${SCIENCE_AGENT_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/harness/science-agent-sandbox}"
# Apply evaluation settings after loading credentials, which may include legacy defaults.
exec bash "$SCIENCE_AGENT_ROOT/run_sandbox_harness.sh" env \
  SCIENCE_AGENT_JUDGE_MODEL="${SCIENCE_AGENT_JUDGE_MODEL:-bailian/deepseek-v4.1-flash}" \
  SCIENCE_AGENT_JUDGE_THINKING="${SCIENCE_AGENT_JUDGE_THINKING:-enabled}" \
  SCIENCE_AGENT_JUDGE_MAX_TOKENS="${SCIENCE_AGENT_JUDGE_MAX_TOKENS:-6000}" \
  bash "$SCRIPT_DIR/eval_science_amobench_sandbox.sh" "$@"
