#!/usr/bin/env python3
"""Build a symlinked HF checkpoint overlay with SFT text and base vision weights."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


def load_index(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    with index_path.open() as stream:
        return json.load(stream)


def link_file(source: Path, destination: Path) -> None:
    destination.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-model", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sft_dir = args.sft_model.resolve(strict=True)
    base_dir = args.base_model.resolve(strict=True)
    output = args.output.absolute()
    sft_index = load_index(sft_dir)
    base_index = load_index(base_dir)
    sft_weights = sft_index["weight_map"]
    base_weights = base_index["weight_map"]
    visual_weights = {
        key: shard
        for key, shard in base_weights.items()
        if key.startswith("model.visual.") or ".visual." in key
    }

    if not visual_weights:
        raise RuntimeError(f"No visual weights found in base checkpoint: {base_dir}")
    if any("visual" in key.lower() for key in sft_weights):
        raise RuntimeError(f"SFT checkpoint already has visual weights: {sft_dir}")
    overlap = set(sft_weights).intersection(visual_weights)
    if overlap:
        raise RuntimeError(f"SFT/base visual key collision: {sorted(overlap)[:5]}")

    sft_config = json.loads((sft_dir / "config.json").read_text())
    base_config = json.loads((base_dir / "config.json").read_text())
    for key in ("model_type", "architectures", "vision_config"):
        if sft_config.get(key) != base_config.get(key):
            raise RuntimeError(f"Checkpoint config mismatch at {key!r}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = load_index(output)
        existing_visual = [key for key in existing["weight_map"] if "visual" in key.lower()]
        if len(existing_visual) == len(visual_weights):
            print(f"HF visual overlay already exists: {output} ({len(existing_visual)} visual tensors)")
            return
        raise FileExistsError(f"Output exists but is not a valid visual overlay: {output}")

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for source in sft_dir.iterdir():
            if source.name == "model.safetensors.index.json":
                continue
            if source.name.endswith(".safetensors"):
                if source.name not in set(sft_weights.values()):
                    continue
            link_file(source, temp_dir / source.name)

        base_shards: dict[str, str] = {}
        for source_name in sorted(set(visual_weights.values())):
            alias = f"base-{source_name}"
            link_file(base_dir / source_name, temp_dir / alias)
            base_shards[source_name] = alias

        merged_weights = dict(sft_weights)
        merged_weights.update({key: base_shards[shard] for key, shard in visual_weights.items()})
        metadata = dict(sft_index.get("metadata", {}))
        merged_index = {"metadata": metadata, "weight_map": merged_weights}
        (temp_dir / "model.safetensors.index.json").write_text(
            json.dumps(merged_index, indent=2, sort_keys=True) + "\n"
        )
        os.rename(temp_dir, output)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(
        f"Created HF visual overlay: {output}\n"
        f"  SFT tensors: {len(sft_weights)}\n"
        f"  Base visual tensors: {len(visual_weights)}\n"
        f"  Linked base shards: {len(base_shards)}"
    )


if __name__ == "__main__":
    main()
