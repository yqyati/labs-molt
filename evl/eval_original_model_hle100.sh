#!/usr/bin/env bash
# Evaluate the original science policy on HLE-100 through Molt rollout.
# No checkpoint consolidation is performed by this entrypoint.
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace}"
MOLT_REPO="${MOLT_REPO:-$WORKSPACE_ROOT/labs-molt}"
HARNESS_ROOT="${HARNESS_ROOT:-$WORKSPACE_ROOT/harness/science-agent}"
GPFS2_ROOT="${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}"
SCI_RESEARCH_ROOT="${SCI_RESEARCH_ROOT:-$GPFS2_ROOT/scientific_reasoning_and_research}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WHEEL_DIR="${QWEN_AGENT_WHEEL_DIR:-$SCI_RESEARCH_ROOT/yangqingyu/rft/offline_wheels/qwen_agent_py312_cp312_clean}"
INSTALLER="${QWEN_AGENT_INSTALLER:-$SCI_RESEARCH_ROOT/yangqingyu/rft/offline_wheels/install_qwen_agent_offline.sh}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/science_model/tool_models/hf_models/p1_rollout1_kimi2.6_subset_kimi_prompt_sft_qwen3.5_35b-0511-201941}"
HLE_SOURCE="${HLE_SOURCE:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/yangqingyu/science-agent-next/data/hle_subsets/HLE_all_short_100.jsonl}"
MODEL_NAME="${MODEL_NAME:-original}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/shared-storage-user/ai4sgi-gpfs1/yangqingyu/eval/molt_hle100/$MODEL_NAME}"
EVAL_DATASET="${EVAL_DATASET:-$OUTPUT_ROOT/hle100_molt.jsonl}"

GATEWAY_URL="${GATEWAY_URL:-http://100.102.21.127:8765}"
JUDGE_MODEL="${JUDGE_MODEL:-o3-mini}"
TP="${TP:-4}"
MAX_WORKERS="${MAX_WORKERS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-98304}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"

test -d "$MOLT_REPO" || { echo "Missing Molt repository: $MOLT_REPO" >&2; exit 1; }
test -f "$MOLT_REPO/molt/cli/train_rl_ray.py" || { echo "Missing Molt CLI" >&2; exit 1; }
test -f "$HARNESS_ROOT/inference/react_agent_openai_api_tools.py" || {
  echo "Missing pinned Legacy ReAct harness: $HARNESS_ROOT" >&2; exit 1;
}
test -d "$WHEEL_DIR" || { echo "Missing offline qwen-agent wheels: $WHEEL_DIR" >&2; exit 1; }
test -f "$INSTALLER" || { echo "Missing qwen-agent installer: $INSTALLER" >&2; exit 1; }
test -f "$HLE_SOURCE" || { echo "Missing HLE-100 dataset: $HLE_SOURCE" >&2; exit 1; }
test -f "$MODEL_PATH/config.json" || {
  echo "Original model does not contain config.json: $MODEL_PATH" >&2
  exit 1
}

mkdir -p "$OUTPUT_ROOT"

export SCIENCE_AGENT_ROOT="$HARNESS_ROOT"
export PYTHONPATH="$MOLT_REPO:$HARNESS_ROOT:$HARNESS_ROOT/inference${PYTHONPATH:+:$PYTHONPATH}"
echo "Installing qwen-agent from local wheels only: $WHEEL_DIR" >&2
PYTHON_BIN="$PYTHON_BIN" bash "$INSTALLER" "$WHEEL_DIR"
"$PYTHON_BIN" - <<'PY'
import qwen_agent
from react_agent_openai_api_tools import MultiTurnReactAgent
print(f"qwen-agent import ok: {qwen_agent.__file__}")
print(f"Legacy ReAct import ok: {MultiTurnReactAgent.__module__}")
PY

python3 - "$HLE_SOURCE" "$EVAL_DATASET" <<'PY'
import json
import pathlib
import sys

source, target = map(pathlib.Path, sys.argv[1:])
count = 0
with source.open(encoding="utf-8") as inp, target.open("w", encoding="utf-8") as out:
    for line in inp:
        if not line.strip():
            continue
        row = json.loads(line)
        question = row.get("question") or row.get("problem") or row.get("prompt")
        if not question:
            raise SystemExit(f"HLE row has no question: {row.keys()}")
        item_id = row.get("id") or row.get("question_id") or f"hle100-{count:04d}"
        item = dict(row)
        item["question_id"] = item.get("question_id") or item_id
        item["answer"] = item.get("answer") or row.get("ground_truth") or row.get("target") or ""
        item["source"] = item.get("source") or "hle_all_mini_100"
        out.write(json.dumps({
            "prompt": [{"role": "user", "content": str(question)}],
            "answer": str(item["answer"]),
            "science_item": item,
            "id": str(item_id),
            "datasource": "hle_all_mini_100",
        }, ensure_ascii=False) + "\n")
        count += 1
if count != 100:
    raise SystemExit(f"Expected 100 HLE rows, found {count}")
print(f"prepared {count} rows: {target}")
PY

export SCIENCE_AGENT_MODEL_PATH="$MODEL_PATH"
export SCIENCE_AGENT_GATEWAY_URL="$GATEWAY_URL"
export SCIENCE_AGENT_JUDGE_BASE_URL="${SCIENCE_AGENT_JUDGE_BASE_URL:-${GATEWAY_URL%/}/v1/judge}"
export SCIENCE_AGENT_JUDGE_MODEL="$JUDGE_MODEL"
export SCIENCE_AGENT_JUDGE_MAX_TOKENS="${SCIENCE_AGENT_JUDGE_MAX_TOKENS:-6000}"
export SCIENCE_AGENT_NEXT_HLE_EVAL_PROTOCOL=1
export SCIENCE_AGENT_TOOLS="${SCIENCE_AGENT_TOOLS:-search,visit,python,google_scholar}"
# Keep evaluation on the same bounded tool-output protocol as training.
export SCIENCE_AGENT_READ_PAGE_MODE="${SCIENCE_AGENT_READ_PAGE_MODE:-extractive}"
export SCIENCE_AGENT_READ_PAGE_MAX_CHARS="${SCIENCE_AGENT_READ_PAGE_MAX_CHARS:-6000}"
export SCIENCE_AGENT_READ_PAGE_MAX_EVIDENCE_BLOCKS="${SCIENCE_AGENT_READ_PAGE_MAX_EVIDENCE_BLOCKS:-4}"
export SCIENCE_AGENT_SEARCH_MAX_RESULTS="${SCIENCE_AGENT_SEARCH_MAX_RESULTS:-6}"
export SCIENCE_AGENT_SEARCH_SNIPPET_CHARS="${SCIENCE_AGENT_SEARCH_SNIPPET_CHARS:-600}"
export SCIENCE_AGENT_SCHOLAR_MAX_RESULTS="${SCIENCE_AGENT_SCHOLAR_MAX_RESULTS:-6}"
export SCIENCE_AGENT_SCHOLAR_SNIPPET_CHARS="${SCIENCE_AGENT_SCHOLAR_SNIPPET_CHARS:-700}"
export OPENAI_COMPAT_USE_SYSTEM_PROMPT=1 AGENT_TOOL_BACKEND=remote AGENT_EVAL_FORCE_FINALIZE=1
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}100.102.21.127"
export no_proxy="$NO_PROXY"
export WANDB_DISABLED=true RAY_USAGE_STATS_ENABLED=0 VLLM_WORKER_MULTIPROC_METHOD=spawn

cd "$MOLT_REPO"
echo "Molt original-model HLE-100 eval: model=$MODEL_PATH output=$OUTPUT_ROOT tp=$TP workers=$MAX_WORKERS" >&2
python3 -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$MODEL_PATH" \
  --data.prompt_dataset "$EVAL_DATASET" \
  --data.input_key prompt \
  --data.label_key science_item \
  --data.apply_chat_template \
  --data.max_len "$MAX_MODEL_LEN" \
  --data.dataloader_num_workers 0 \
  --rollout.batch_size "$MAX_WORKERS" \
  --rollout.n_samples_per_prompt 1 \
  --rollout.max_new_tokens "$MAX_NEW_TOKENS" \
  --rollout.temperature "${TEMPERATURE:-1.0}" \
  --rollout.top_p "${TOP_P:-1.0}" \
  --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
  --vllm.num_engines 1 \
  --vllm.tensor_parallel_size "$TP" \
  --vllm.gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.8}" \
  --vllm.distributed_executor_backend mp \
  --vllm.enable_auto_tool_choice \
  --vllm.tool_call_parser "${VLLM_TOOL_CALL_PARSER:-qwen3_coder}" \
  --train.agent_path "$MOLT_REPO/examples/python/agents/science_legacy_react.py" \
  --train.rollout_dump_dir "$OUTPUT_ROOT/rollout_dump" \
  --eval.dataset "$EVAL_DATASET" \
  --eval.batch_size "$MAX_WORKERS" \
  --eval.n_samples_per_prompt 1 \
  --eval.max_new_tokens "$MAX_NEW_TOKENS" \
  --eval.eval_only \
  2>&1 | tee "$OUTPUT_ROOT/eval.log"
