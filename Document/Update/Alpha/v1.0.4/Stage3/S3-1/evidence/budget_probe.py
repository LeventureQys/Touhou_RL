#!/usr/bin/env python
"""Measure the D10 residency peak over a simulated epoch (independent process).

``cache_stats()`` is sampled after every load, so ``peak_resident_bytes`` is what
the contract actually promises: the resident byte count reported by the loader
never exceeds ``cache_budget_bytes``.  (RSS is *not* promised - mmap pages, the OS
page cache, temporary batches and the thread pool are explicitly outside the
budget.)

Two modes:

``cached``
    The realistic training path: decode cache on, batches of ``--batch`` over every
    ``--stride``-th sample, reading the three ROIs and the centre labels.  Only the
    label records are resident, because pixel frames are mmapped.

``npz``
    The worst case: decode cache off and one sample per shard in order, which loads
    every full shard record (blobs included) and forces LRU eviction against the
    512 MiB budget.

    python Document/Update/Alpha/v1.0.4/Stage3/S3-1/evidence/budget_probe.py \
        --dataset third_party/th08/build/rl/datasets/pilot-ud0063 --mode npz \
        --out Document/Update/Alpha/v1.0.4/Stage3/S3-1/evidence/budget-B.json

Exit 0 when the peak stayed inside the budget, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "touhou_rl" / "vision_data.py").exists():
            return candidate
    raise SystemExit("cannot locate the repository root")


REPO_ROOT = _repo_root(Path(__file__).resolve().parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from touhou_rl.vision_data import ROI_SIZES, VisionDataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("cached", "npz"), default="cached")
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--budget", type=int, default=None)
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset)
    kwargs = {"use_decode_cache": args.mode == "cached"}
    if args.budget is not None:
        kwargs["cache_budget_bytes"] = int(args.budget)
    dataset = VisionDataset(dataset_dir, **kwargs)
    peak = 0
    over_budget_samples = 0
    loads = 0
    started = time.time()
    try:
        budget = dataset.cache_stats()["budget_bytes"]
        if args.mode == "cached":
            step = max(1, int(args.stride))
            samples = list(range(0, len(dataset), step))
            for offset in range(0, len(samples), max(1, int(args.batch))):
                batch = samples[offset : offset + max(1, int(args.batch))]
                dataset.labels(batch)
                for region, (height, width) in ROI_SIZES.items():
                    dataset.frames(batch, region, height, width)
                stats = dataset.cache_stats()
                loads += 1
                peak = max(peak, stats["resident_bytes"])
                over_budget_samples += int(stats["resident_bytes"] > stats["budget_bytes"])
        else:
            for _path, start, _size in dataset.offsets:
                dataset.frames([start], "playfield", *ROI_SIZES["playfield"])
                stats = dataset.cache_stats()
                loads += 1
                peak = max(peak, stats["resident_bytes"])
                over_budget_samples += int(stats["resident_bytes"] > stats["budget_bytes"])
        final = dataset.cache_stats()
        decode = dataset.decode_cache_stats()
    finally:
        dataset.close()
    payload = {
        "contract_version": "v104-budget-probe/1",
        "dataset": str(dataset_dir),
        "mode": args.mode,
        "budget_bytes": budget,
        "loads_sampled": loads,
        "peak_resident_bytes": int(peak),
        "over_budget_samples": int(over_budget_samples),
        "within_budget": bool(over_budget_samples == 0),
        "final_cache_stats": final,
        "decode_cache_stats": decode,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "dataset",
                    "mode",
                    "budget_bytes",
                    "loads_sampled",
                    "peak_resident_bytes",
                    "over_budget_samples",
                    "within_budget",
                    "elapsed_seconds",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if payload["within_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
