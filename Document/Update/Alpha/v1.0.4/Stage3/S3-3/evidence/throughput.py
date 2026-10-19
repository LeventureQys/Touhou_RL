"""Steady-state epoch throughput: de-duplicating gather vs the naive per-slot gather.

The parent task reported depth-1 = 39.0 s and depth-3 = 69.5-70.2 s per epoch
(7241-frame validation included) before the de-duplication, over the 60 s/epoch
hard gate.  This harness measures both implementations on the *same* code base:
the only difference is ``vision_train._gather``, which is monkeypatched with the
v1.0.3 implementation for the ``naive`` runs.

    python Document/Update/Alpha/v1.0.4/Stage3/S3-3/evidence/throughput.py \
        --out Document/Update/Alpha/v1.0.4/Stage3/S3-3/evidence/throughput.json

Runs sequentially (one GPU, no concurrency): d1-fast, d3-fast, d1-naive,
d3-naive, four epochs each with full validation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from touhou_rl import vision, vision_train

DATASET = "third_party/th08/build/rl/datasets/pilot-ud0059-v2"
RUNS = Path("third_party/th08/build/rl/runs")


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


def equality_spot_check() -> dict:
    """Same real batch through both gathers: value/dtype/shape must be identical."""
    from touhou_rl.vision_data import VisionDataset

    with VisionDataset(DATASET, history_contract="causal-segment-v1") as dataset:
        indices = list(range(70000, 70096))
        sizes = {"playfield": (224, 192), "hud_score": (34, 216), "hud_status": (126, 216)}
        fast, fast_labels = vision_train._gather(dataset, indices, 3, sizes)
        slow, slow_labels = naive_gather(dataset, indices, 3, sizes)
        stack_ids = dataset.stack_indices(indices, 3).reshape(-1)
        slots = int(stack_ids.size)
        unique = int(np.unique(stack_ids).size)
    identical = all(
        bool(torch.equal(fast[name], slow[name])) and fast[name].dtype == slow[name].dtype for name in sizes
    )
    labels_identical = all(bool(np.array_equal(fast_labels[key], slow_labels[key])) for key in fast_labels)
    return {
        "indices": [indices[0], indices[-1]],
        "depth": 3,
        "slots": slots,
        "unique_frames": unique,
        "duplicate_slots": slots - unique,
        "tensors_identical": bool(identical),
        "labels_identical": bool(labels_identical),
    }


def run_one(depth: int, mode: str, epochs: int, tag: str = "") -> dict:
    suffix = f"-{tag}" if tag else ""
    out_dir = RUNS / f"v104-s33-throughput-{mode}-d{depth}{suffix}-s7"
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; refusing to overwrite a throughput run")
    cfg = vision_train.VisionTrainConfig(
        dataset=Path(DATASET),
        out_dir=out_dir,
        arch="v2",
        stage="state",
        epochs_state=epochs,
        epochs_policy=0,
        batch_size=96,
        depth=depth,
        device="cuda",
        selection_contract="v104",
        history_contract="causal-segment-v1",
        max_train_batches=0,
        max_val_batches=0,
    )
    real_gather = vision_train._gather
    if mode == "naive":
        vision_train._gather = naive_gather
    started = time.perf_counter()
    try:
        summary = vision_train.train(cfg)
    finally:
        vision_train._gather = real_gather
    elapsed = time.perf_counter() - started
    history = summary["stages"]["state"]["history"]
    rows = [
        {
            "epoch": row["epoch"],
            "seconds": row["seconds"],
            "train_loss": row["train_loss"],
            "val_player_pos_px": row.get("val_player_pos_px"),
            "val_frames": row["val_frames"],
        }
        for row in history
    ]
    steady = [row["seconds"] for row in rows[2:]]  # epochs 3 and 4
    coverage = summary["coverage"]["val"]
    return {
        "depth": depth,
        "mode": mode,
        "epochs": rows,
        "epoch3_seconds": rows[2]["seconds"] if len(rows) > 2 else None,
        "epoch4_seconds": rows[3]["seconds"] if len(rows) > 3 else None,
        "steady_mean_seconds": round(float(np.mean(steady)), 2) if steady else None,
        "steady_min_seconds": min(steady) if steady else None,
        "max_seconds": max(row["seconds"] for row in rows),
        "elapsed_seconds": round(elapsed, 1),
        "val_frames": rows[-1]["val_frames"],
        "coverage_val_actual_frames": coverage["actual_frames"],
        "coverage_val_complete": coverage["complete"],
        "coverage_val_sha256": coverage["visited_index_sha256"],
        "summary": str(out_dir / "vision-summary.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--skip-naive", action="store_true")
    parser.add_argument("--only", default="", help="comma list like 'd1-fast,d3-fast'")
    parser.add_argument("--tag", default="", help="run-directory suffix, e.g. 'recheck'")
    args = parser.parse_args()
    wanted = {item.strip() for item in args.only.split(",") if item.strip()}
    results = {"spot_check": equality_spot_check(), "runs": []}
    print(json.dumps({"spot_check": results["spot_check"]}, indent=2), flush=True)
    for depth in (1, 3):
        for mode in ("fast", "naive"):
            if mode == "naive" and args.skip_naive:
                continue
            if wanted and f"d{depth}-{mode}" not in wanted:
                continue
            print(f"[throughput] depth {depth} mode {mode} ...", flush=True)
            result = run_one(depth, mode, args.epochs, args.tag)
            results["runs"].append(result)
            print(json.dumps(result, indent=2), flush=True)
    by_key = {(row["depth"], row["mode"]): row for row in results["runs"]}
    for depth in (1, 3):
        fast = by_key.get((depth, "fast"))
        slow = by_key.get((depth, "naive"))
        if fast and slow:
            results[f"depth{depth}_speedup"] = round(
                slow["steady_mean_seconds"] / fast["steady_mean_seconds"], 2
            )
            results[f"depth{depth}_under_60s"] = bool(fast["max_seconds"] <= 60.0)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in results.items() if key != "runs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
