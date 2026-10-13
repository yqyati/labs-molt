#!/usr/bin/env bash
# Run an isolated HLE subset Sandbox evaluation through Molt rollout.
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace}"
MOLT_REPO="${MOLT_REPO:-$WORKSPACE_ROOT/labs-molt}"
HARNESS_ROOT="${HARNESS_ROOT:-$WORKSPACE_ROOT/harness/science-agent-sandbox}"
GPFS2_ROOT="${GPFS2_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2}"
SCI_RESEARCH_ROOT="${SCI_RESEARCH_ROOT:-$GPFS2_ROOT/scientific_reasoning_and_research}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WHEEL_DIR="${QWEN_AGENT_WHEEL_DIR:-$SCI_RESEARCH_ROOT/yangqingyu/rft/offline_wheels/qwen_agent_py312_cp312_clean}"
INSTALLER="${QWEN_AGENT_INSTALLER:-$SCI_RESEARCH_ROOT/yangqingyu/rft/offline_wheels/install_qwen_agent_offline.sh}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to an HF model or consolidated Molt checkpoint directory}"
HLE_SOURCE="${HLE_SOURCE:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/yangqingyu/science-agent-next/data/hle_subsets/HLE_all_short_100.jsonl}"
HLE_EXPECTED_ROWS="${HLE_EXPECTED_ROWS:-100}"
HLE_DATASET_NAME="${HLE_DATASET_NAME:-hle_all_mini_100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/eval/molt_hle100_sandbox/$(date +%Y%m%d-%H%M%S)}"
EVAL_DATASET="${EVAL_DATASET:-$OUTPUT_ROOT/hle${HLE_EXPECTED_ROWS}_molt.jsonl}"
CONVERT_ROOT="${CONVERT_ROOT:-$OUTPUT_ROOT/hf_model}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/science_model/tool_models/hf_models/p1_rollout1_kimi2.6_subset_kimi_prompt_sft_qwen3.5_35b-0511-201941}"

GATEWAY_URL="${GATEWAY_URL:-http://100.102.21.127:8765}"
JUDGE_MODEL="${SCIENCE_AGENT_JUDGE_MODEL:-${JUDGE_MODEL:-o3-mini}}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH%/}")}"
TP="${TP:-4}"
MAX_WORKERS="${MAX_WORKERS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-98304}"
export SCIENCE_TRAJECTORY_MAX_TOKENS="${SCIENCE_TRAJECTORY_MAX_TOKENS:-65536}"
MAX_NEW_TOKENS="$SCIENCE_TRAJECTORY_MAX_TOKENS"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"

test -d "$MOLT_REPO" || { echo "Missing Molt repository: $MOLT_REPO" >&2; exit 1; }
test -f "$MOLT_REPO/molt/cli/train_rl_ray.py" || { echo "Missing Molt CLI" >&2; exit 1; }
test -f "$HARNESS_ROOT/inference/react_agent_openai_api_tools.py" || {
  echo "Missing pinned Legacy ReAct harness: $HARNESS_ROOT" >&2; exit 1;
}
test -d "$WHEEL_DIR" || { echo "Missing offline qwen-agent wheels: $WHEEL_DIR" >&2; exit 1; }
test -f "$INSTALLER" || { echo "Missing qwen-agent installer: $INSTALLER" >&2; exit 1; }
test -f "$HLE_SOURCE" || { echo "Missing HLE source: $HLE_SOURCE" >&2; exit 1; }
test -f "$MODEL_PATH/config.json" || test -f "$MODEL_PATH/.hf_metadata/config.json" || {
  echo "MODEL_PATH must be an HF model or Molt actor checkpoint: $MODEL_PATH" >&2
  exit 1
}

mkdir -p "$OUTPUT_ROOT" "$CONVERT_ROOT"

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

is_hf_model_dir() {
  [[ -f "$1/config.json" ]] && compgen -G "$1/*.safetensors" >/dev/null
}

prepare_model() {
  if is_hf_model_dir "$MODEL_PATH"; then
    printf '%s\n' "$MODEL_PATH"
    return
  fi
  local converted="$CONVERT_ROOT/$(basename "${MODEL_PATH%/}")"
  if is_hf_model_dir "$converted"; then
    printf '%s\n' "$converted"
    return
  fi
  test -f "$BASE_MODEL_PATH/config.json" || {
    echo "Missing base model: $BASE_MODEL_PATH" >&2
    exit 1
  }
  mkdir -p "$converted"
  echo "Consolidating Molt checkpoint $MODEL_PATH -> $converted" >&2
  if [[ -n "${CONSOLIDATION_TOOL:-}" ]]; then
    python3 "$CONSOLIDATION_TOOL" --backend gloo --num-threads "${CONSOLIDATION_THREADS:-8}" \
      --model-name "$BASE_MODEL_PATH" --input-dir "$MODEL_PATH" --output-dir "$converted" >&2
  else
    # Runtime images include the consolidation library without the repository CLI.
    python3 - "$MODEL_PATH" "$converted" "${CONSOLIDATION_THREADS:-8}" <<'PY' >&2
import json
import shutil
import sys
from pathlib import Path

import torch
from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import consolidate_safetensors_files

source, target = map(Path, sys.argv[1:3])
metadata = source / ".hf_metadata"
consolidate_safetensors_files(
    str(source), str(target),
    json.loads((metadata / "fqn_to_file_index_mapping.json").read_text()),
    num_threads=int(sys.argv[3]), cast_dtype=torch.bfloat16,
    fqn_to_dtype_mapping=json.loads((metadata / "fqn_to_dtype_mapping.json").read_text()),
)
config = json.loads((metadata / "config.json").read_text())
for section in (config, config.get("text_config", {}), config.get("vision_config", {})):
    section["dtype"] = "bfloat16"
    section.pop("torch_dtype", None)
if (metadata / "generation_config.json").exists():
    shutil.copy2(metadata / "generation_config.json", target)
# Publish config last so an interrupted conversion is not mistaken for an HF model.
(target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
print(f"Consolidated checkpoint: {target}")
PY
  fi
  local f
  for f in tokenizer.json tokenizer_config.json special_tokens_map.json \
           chat_template.jinja preprocessor_config.json processor_config.json; do
    if [[ -e "$BASE_MODEL_PATH/$f" && ! -e "$converted/$f" ]]; then
      ln -s "$BASE_MODEL_PATH/$f" "$converted/$f"
    fi
  done
  is_hf_model_dir "$converted" || {
    echo "Conversion did not produce a loadable HF model: $converted" >&2
    exit 1
  }
  printf '%s\n' "$converted"
}

export SCIENCE_AGENT_READ_PAGE_MODE="${SCIENCE_AGENT_READ_PAGE_MODE:-file}"
export PYTHON_TOOL_RETURN_MAX_CHARS="${PYTHON_TOOL_RETURN_MAX_CHARS:-6000}"
python3 -u "$HARNESS_ROOT/sandbox_smoke.py" --llm --search --judge --page-file --http-url https://r.jina.ai/http://example.com 2>&1 | tee "$OUTPUT_ROOT/sandbox_preflight.log"

SERVE_MODEL_PATH="$(prepare_model)"

# Convert HLE rows to the exact schema consumed by Molt's science adapter.
python3 - "$HLE_SOURCE" "$EVAL_DATASET" "$HLE_EXPECTED_ROWS" "$HLE_DATASET_NAME" <<'PY'
import json, pathlib, sys
src, dst = map(pathlib.Path, sys.argv[1:3])
expected_rows, dataset_name = int(sys.argv[3]), sys.argv[4]
count = 0
with src.open(encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as out:
    for line in inp:
        if not line.strip():
            continue
        row = json.loads(line)
        question = row.get("question") or row.get("problem") or row.get("prompt")
        answer = row.get("answer") or row.get("ground_truth") or row.get("target") or ""
        if not question:
            raise SystemExit(f"HLE row has no question: {row.keys()}")
        item_id = row.get("id") or row.get("question_id") or row.get("task_id") or f"hle{expected_rows}-{count:04d}"
        item = dict(row)
        item["question_id"] = item.get("question_id") or item_id
        item["answer"] = answer
        item["source"] = item.get("source") or dataset_name
        out.write(json.dumps({
            "prompt": [{"role": "user", "content": str(question)}],
            "answer": str(answer),
            "science_item": item,
            "id": str(item_id),
            "datasource": dataset_name,
        }, ensure_ascii=False) + "\n")
        count += 1
if count != expected_rows:
    raise SystemExit(f"Expected {expected_rows} Sandbox evaluation rows, found {count}")
print(f"prepared {count} rows: {dst}")
PY

export SCIENCE_AGENT_ROOT="$HARNESS_ROOT"
export SCIENCE_AGENT_MODEL_PATH="$SERVE_MODEL_PATH"
export QWEN_API_MAX_OUTPUT_TOKENS="$MAX_NEW_TOKENS"
export SCIENCE_AGENT_GATEWAY_URL="$GATEWAY_URL"
export SCIENCE_AGENT_JUDGE_BASE_URL="${SCIENCE_AGENT_JUDGE_BASE_URL:-${API_BASE%/}}"
export SCIENCE_AGENT_JUDGE_MODEL="$JUDGE_MODEL"
export SCIENCE_AGENT_TOOLS="${SCIENCE_AGENT_TOOLS:-search,visit,python,google_scholar}"
export SCIENCE_AGENT_JUDGE_MAX_TOKENS="${SCIENCE_AGENT_JUDGE_MAX_TOKENS:-6000}"
# Keep evaluation on the same bounded tool-output protocol as training.
export SCIENCE_AGENT_READ_PAGE_MODE="${SCIENCE_AGENT_READ_PAGE_MODE:-file}"
export PYTHON_TOOL_RETURN_MAX_CHARS="${PYTHON_TOOL_RETURN_MAX_CHARS:-6000}"
export SCIENCE_AGENT_READ_PAGE_MAX_CHARS="${SCIENCE_AGENT_READ_PAGE_MAX_CHARS:-6000}"
export SCIENCE_AGENT_READ_PAGE_MAX_EVIDENCE_BLOCKS="${SCIENCE_AGENT_READ_PAGE_MAX_EVIDENCE_BLOCKS:-4}"
export SCIENCE_AGENT_SEARCH_MAX_RESULTS="${SCIENCE_AGENT_SEARCH_MAX_RESULTS:-6}"
export SCIENCE_AGENT_SEARCH_SNIPPET_CHARS="${SCIENCE_AGENT_SEARCH_SNIPPET_CHARS:-600}"
export SCIENCE_AGENT_SCHOLAR_MAX_RESULTS="${SCIENCE_AGENT_SCHOLAR_MAX_RESULTS:-6}"
export SCIENCE_AGENT_SCHOLAR_SNIPPET_CHARS="${SCIENCE_AGENT_SCHOLAR_SNIPPET_CHARS:-700}"
# Evaluation-only OpenHands-style observation masking.  The harness keeps the
# full trajectory for diagnostics; this switch changes only the model-facing
# request View.  Set AGENT_OBSERVATION_MASKING=0 for the pre-mask baseline.
export AGENT_OBSERVATION_MASKING=0
export AGENT_OBSERVATION_MASKING_WINDOW="${AGENT_OBSERVATION_MASKING_WINDOW:-8}"
export SCIENCE_AGENT_NEXT_HLE_EVAL_PROTOCOL=1
export OPENAI_COMPAT_USE_SYSTEM_PROMPT=1 AGENT_TOOL_BACKEND=local AGENT_EVAL_FORCE_FINALIZE=0
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}100.102.21.127"
export no_proxy="$NO_PROXY"
export WANDB_DISABLED=true RAY_USAGE_STATS_ENABLED=0 VLLM_WORKER_MULTIPROC_METHOD=spawn

cd "$MOLT_REPO"
echo "Molt HLE eval: name=$HLE_DATASET_NAME rows=$HLE_EXPECTED_ROWS model=$SERVE_MODEL_PATH dataset=$EVAL_DATASET output=$OUTPUT_ROOT tp=$TP workers=$MAX_WORKERS" >&2
python3 -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$SERVE_MODEL_PATH" \
  --data.prompt_dataset "$EVAL_DATASET" \
  --data.input_key prompt \
  --data.label_key science_item \
  --data.apply_chat_template \
  --data.max_len "$MAX_MODEL_LEN" \
  --data.max_images_per_prompt "${MAX_IMAGES_PER_PROMPT:-0}" \
  --data.dataloader_num_workers 0 \
  --rollout.batch_size "$MAX_WORKERS" \
  --rollout.n_samples_per_prompt 1 \
  --rollout.max_new_tokens "$MAX_NEW_TOKENS" \
  --rollout.temperature "$TEMPERATURE" \
  --rollout.top_p "$TOP_P" \
  --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
  --vllm.num_engines 1 \
  --vllm.tensor_parallel_size "$TP" \
  --vllm.gpu_memory_utilization 0.8 \
  --vllm.distributed_executor_backend mp \
  --vllm.enable_auto_tool_choice \
  --vllm.tool_call_parser "${VLLM_TOOL_CALL_PARSER:-qwen3_coder}" \
  --train.agent_path "$MOLT_REPO/examples/python/agents/science_legacy_react_sandbox.py" \
  --train.rollout_dump_dir "$OUTPUT_ROOT/rollout_dump" \
  --eval.dataset "$EVAL_DATASET" \
  --eval.batch_size "$MAX_WORKERS" \
  --eval.n_samples_per_prompt 1 \
  --eval.max_new_tokens "$MAX_NEW_TOKENS" \
  --eval.temperature "$TEMPERATURE" \
  --eval.top_p "$TOP_P" \
  --eval.eval_only \
  --logger.tensorboard_dir "$OUTPUT_ROOT/tensorboard" \
  2>&1 | tee "$OUTPUT_ROOT/eval.log"
