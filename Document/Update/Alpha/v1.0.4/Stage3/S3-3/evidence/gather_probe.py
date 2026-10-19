"""Where does an epoch's time actually go?  Gather-only microbenchmark.

The 4-epoch measurement (`throughput.py`) showed the de-duplication changing the
steady-state epoch by +9% at depth 1 (pure overhead, since depth 1 cannot repeat
a frame) and -1% at depth 3, i.e. it does *not* bring depth 3 under 60 s.  This
probe separates the data path from the model and explains why.

Each configuration is measured **twice**: the first pass is cold (the decoded
mmap page cache does not hold the 16.8 GB cache set), the second pass is warm.
Only the warm pass is comparable, and the first pass is reported so the cold-run
confound is visible instead of hidden.

    python Document/Update/Alpha/v1.0.4/Stage3/S3-3/evidence/gather_probe.py \
        --out Document/Update/Alpha/v1.0.4/Stage3/S3-3/evidence/gather-probe.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from touhou_rl import vision, vision_train
from touhou_rl.pipeline.manifest import DatasetManifest
from touhou_rl.vision_data import VisionDataset

DATASET = "third_party/th08/build/rl/datasets/pilot-ud0059-v2"
SIZES = {"playfield": (224, 192), "hud_score": (34, 216), "hud_status": (126, 216)}
BATCH = 96
TRAIN_BATCHES = 679  # 65172 frames at batch 96, drop_last
VAL_BATCHES = 76  # 7241 frames in 75 x 96 + 41


def naive_gather(dataset, indices, depth, region_sizes):
    """v1.0.3: decode every stack slot, duplicates included."""
    stack_ids = dataset.stack_indices(indices, depth).reshape(-1).tolist()
    regions = dataset.region_frames(stack_ids, region_sizes)
    stacked = {
        name: frames.reshape(
            len(indices), depth, region_sizes[name][0], region_sizes[name][1], vision.VISION_CHANNELS
        )
        for name, frames in regions.items()
    }
    center = dataset.labels(indices)
    labels = {
        "self_feat": center["self_feat"].astype(np.float32),
        "danger": center["danger"].astype(np.float32),
        "action": center["action"].astype(np.int64),
        "bullets": center["bullets"].astype(np.float32),
    }
    return {name: torch.as_tensor(value, dtype=torch.uint8) for name, value in stacked.items()}, labels


def measure(dataset, order, depth, gather, batches: int) -> dict:
    started = time.perf_counter()
    slots = 0
    unique_slots = 0
    for start in range(0, batches * BATCH, BATCH):
        indices = order[start : start + BATCH]
        gather(dataset, indices, depth, SIZES)
        stack = dataset.stack_indices(indices, depth).reshape(-1)
        slots += int(stack.size)
        unique_slots += int(np.unique(stack).size)
    elapsed = time.perf_counter() - started
    return {
        "seconds": round(elapsed, 3),
        "seconds_per_batch_ms": round(1000.0 * elapsed / batches, 2),
        "slot_reduction": round(1.0 - unique_slots / slots, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--batches", type=int, default=25)
    args = parser.parse_args()
    results = {"dataset": DATASET, "batch_size": BATCH, "sizes": SIZES, "runs": []}
    with VisionDataset(DATASET, history_contract="causal-segment-v1") as dataset:
        manifest = DatasetManifest.load(Path(DATASET) / "manifest.json")
        train, val = manifest.verify(Path(DATASET))
        # One fixed shard-chunked order for both modes: the only difference is the
        # gather implementation.
        train_order = dataset.shard_chunked_order(train, np.random.default_rng(7))
        for depth in (1, 3):
            for mode, gather in (("fast", vision_train._gather), ("naive", naive_gather)):
                for name, order, batches in (
                    ("train-shard-chunked", train_order, TRAIN_BATCHES),
                    ("val-order", val, VAL_BATCHES),
                ):
                    cold = measure(dataset, order, depth, gather, args.batches)
                    warm = measure(dataset, order, depth, gather, args.batches)
                    entry = {
                        "depth": depth,
                        "mode": mode,
                        "order": name,
                        "measured_batches": args.batches,
                        "cold": cold,
                        "warm": warm,
                        "epoch_seconds_from_warm": round(
                            warm["seconds_per_batch_ms"] * batches / 1000.0, 1
                        ),
                    }
                    results["runs"].append(entry)
                    print(json.dumps(entry), flush=True)
        coverage = dataset.cache_stats()
    results["cache_stats"] = coverage
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
