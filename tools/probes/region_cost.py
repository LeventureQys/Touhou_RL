"""Measure per-region PNG cost on a real screenshot (the number the plan rests on).

Run before committing to a long capture: it prints bytes per region and the total
corpus size for a given sample count, so the disk budget in the plan is measured
rather than guessed.

Usage::

    python tools/probes/region_cost.py --shot .../eval/vision-v1/episode-01.png --samples 1527044
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from touhou_rl import vision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure per-region PNG storage cost")
    parser.add_argument("--shot", action="append", required=True)
    parser.add_argument("--samples", type=int, default=1_527_044)
    args = parser.parse_args(argv)

    from PIL import Image

    rows = []
    for shot_path in args.shot:
        shot = np.asarray(Image.open(shot_path).convert("RGB"), dtype=np.uint8)
        crops = vision.extract_regions(shot)
        total = 0
        per_region = []
        for name, image in crops.items():
            blob = vision.encode_png(image)
            total += len(blob)
            per_region.append(
                {"region": name, "width": int(image.shape[1]), "height": int(image.shape[0]), "bytes": len(blob)}
            )
        rows.append(
            {
                "shot": Path(shot_path).name,
                "regions": per_region,
                "bytes_per_sample": total,
                "kb_per_sample": round(total / 1024, 1),
                "corpus_gb": round(total * args.samples / 1e9, 1),
            }
        )
    print(json.dumps({"samples": args.samples, "rows": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
