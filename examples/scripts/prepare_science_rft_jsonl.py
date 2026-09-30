#!/usr/bin/env python3
"""Convert the three existing science-agent RFT shards to Molt chat rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SHARDS = [
    "/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/liuzhuo/science-agent/runs/rft_exomind_science_full_k8_ordered/rft/sources/01_Nemotron-RL-Science-v1/shards/00001/rft/artifacts/input/questions.jsonl",
    "/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/liuzhuo/science-agent/runs/rft_exomind_science_full_k8_ordered/rft/sources/01_Nemotron-RL-Science-v1/shards/00002/rft/artifacts/input/questions.jsonl",
    "/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/liuzhuo/science-agent/runs/rft_exomind_science_full_k8_ordered/rft/sources/01_Nemotron-RL-Science-v1/shards/00003/rft/artifacts/input/questions.jsonl",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", action="append", dest="shards")
    args = parser.parse_args()
    shards = args.shards or DEFAULT_SHARDS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as out:
        for shard in shards:
            with Path(shard).open(encoding="utf-8") as inp:
                for line in inp:
                    row = json.loads(line)
                    out_row = {
                        "prompt": [{"role": "user", "content": str(row["question"])}],
                        "answer": str(row.get("answer", "")),
                        # ChatAgent receives this complete object as ctx.label;
                        # it contains both the reference answer and stable ID.
                        "science_item": row,
                        "id": row.get("id") or row.get("question_id"),
                        "datasource": "Nemotron-RL-Science-v1",
                    }
                    out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    count += 1
                    if args.limit and count >= args.limit:
                        print(f"wrote {count} rows to {args.output}")
                        return
    print(f"wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
