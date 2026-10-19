"""One-off probe: why is the measured data path slower than S3-3's gather probe?

Not part of the deliverable tool.  It answers one question for the report: is a
persistent decoded-frame cache active for the A dataset in this session, and how
long does one depth-3 batch take hot with and without de-duplication?
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "third_party").is_dir())
sys.path.insert(0, str(ROOT))

from touhou_rl.vision_data import VisionDataset  # noqa: E402

DATASET = ROOT / "third_party/th08/build/rl/datasets/pilot-ud0059-v2"
OUT = Path(__file__).resolve().with_name("data-path-probe.json")


def main() -> int:
    import torch

    torch.set_num_threads(1)
    sizes = {"playfield": (224, 192), "hud_score": (34, 216), "hud_status": (126, 216)}
    dataset = VisionDataset(DATASET, history_contract="causal-segment-v1")
    try:
        chunk = list(range(70000, 70096))
        history = dataset.stack_indices(chunk, 3)
        flat = history.reshape(-1)
        unique, inverse = np.unique(flat, return_inverse=True)
        rows = {}
        for mode, indices in (("naive", flat.tolist()), ("dedup", unique.tolist())):
            dataset.region_frames(indices, sizes)  # cold touch
            timings = []
            for _ in range(5):
                start = time.perf_counter()
                dataset.region_frames(indices, sizes)
                timings.append(time.perf_counter() - start)
            rows[mode] = {
                "slot_count": len(indices),
                "hot_seconds_mean": float(np.mean(timings)),
                "hot_seconds_min": float(np.min(timings)),
            }
        rows["unique_slots"] = int(unique.size)
        rows["requested_slots"] = int(flat.size)
        rows["decode_cache"] = dataset.decode_cache_stats()
        rows["cache_stats"] = dataset.cache_stats()
        labels_start = time.perf_counter()
        for _ in range(5):
            dataset.labels(chunk)
        rows["labels_hot_seconds_mean"] = (time.perf_counter() - labels_start) / 5
        rows["dataset"] = str(DATASET.relative_to(ROOT).as_posix())
    finally:
        dataset.close()
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
