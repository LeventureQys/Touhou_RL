#!/usr/bin/env python
"""Independent-process check: mmap decoded cache == PNG decode, value by value.

Run this from a plain shell (not from pytest) so the numbers are produced by a
process that shares nothing with the test run:

    python Document/Update/Alpha/v1.0.4/Stage3/S3-1/evidence/verify_mmap_vs_png.py \
        --dataset third_party/th08/build/rl/datasets/pilot-ud0063 \
        --out Document/Update/Alpha/v1.0.4/Stage3/S3-1/evidence/pixel-consistency-B.json

For every sampled shard it compares, per region:

* the ``.npy`` cache read through ``np.load(..., mmap_mode="r")`` against a fresh
  PNG decode of the same rows (shape, dtype, ``np.array_equal`` per value);
* ``VisionDataset.frames()`` with the cache enabled against a dataset built with
  ``use_decode_cache=False`` for the same global indices.

Exit 0 when every comparison is exact, 1 otherwise.  Nothing is written except
the ``--out`` JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def _repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "touhou_rl" / "vision_data.py").exists():
            return candidate
    raise SystemExit("cannot locate the repository root (no touhou_rl/vision_data.py above this file)")


REPO_ROOT = _repo_root(Path(__file__).resolve().parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from touhou_rl import vision  # noqa: E402
from touhou_rl.vision_data import (  # noqa: E402
    ROI_SIZES,
    VisionDataset,
    decoded_cache_path,
    read_decode_sidecar,
)


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _sample_rows(count: int, wanted: int) -> list[int]:
    if count <= 0:
        return []
    rows = {0, count - 1, count // 2}
    step = max(1, count // max(1, wanted))
    rows.update(range(0, count, step))
    return sorted(row for row in rows if 0 <= row < count)


def check_region(dataset: VisionDataset, shard: Path, start: int, size: int, region: str,
                 height: int, width: int, full: bool) -> dict:
    cache_path = decoded_cache_path(shard, region, height, width)
    record: dict = {
        "shard": shard.name,
        "region": region,
        "cache": cache_path.name,
        "cache_bytes": int(cache_path.stat().st_size),
        "cache_sha256": _digest(cache_path),
        "rows": "all" if full else "sample",
        "shape": None,
        "dtype": None,
        "rows_checked": 0,
        "mismatched_rows": 0,
        "mismatched_values": 0,
        "max_abs_diff": 0,
        "equal": False,
    }
    sidecar = read_decode_sidecar(cache_path)
    record["sidecar_schema"] = (sidecar or {}).get("schema")
    record["sidecar_source_sha256"] = ((sidecar or {}).get("source") or {}).get("sha256")
    with np.load(shard) as data:
        blob = data[f"{region}_blob"]
        offsets = data[f"{region}_offset"]
    cache = np.load(cache_path, mmap_mode="r")  # memmap: closed by dropping the reference
    record["shape"] = [int(value) for value in cache.shape]
    record["dtype"] = str(cache.dtype)
    rows = list(range(size)) if full else _sample_rows(size, 8)
    mismatched_values = 0
    mismatched_rows = 0
    largest = 0
    for row in rows:
        lo, hi = int(offsets[row]), int(offsets[row + 1])
        expected = vision.decode_png(blob[lo:hi].tobytes(), height, width)
        got = np.asarray(cache[row])
        if not np.array_equal(got, expected):
            mismatched_rows += 1
            difference = np.abs(got.astype(np.int16) - expected.astype(np.int16))
            mismatched_values += int((difference > 0).sum())
            largest = max(largest, int(difference.max()))
    del cache
    record["rows_checked"] = len(rows)
    record["mismatched_rows"] = mismatched_rows
    record["mismatched_values"] = mismatched_values
    record["max_abs_diff"] = largest
    record["equal"] = (
        mismatched_rows == 0
        and record["dtype"] == "uint8"
        and record["shape"] == [int(size), int(height), int(width), 3]
    )
    return record


def check_dataset_path(dataset_dir: Path, shards: list[int], wanted: int) -> dict:
    """``frames()`` (cache on) vs ``frames()`` (cache off) for sampled indices."""
    cached = VisionDataset(dataset_dir, history_contract="legacy")
    plain = VisionDataset(dataset_dir, use_decode_cache=False)
    records = []
    try:
        picks: list[int] = []
        for shard_index in shards:
            path, start, size = cached.offsets[shard_index]
            picks.extend(start + row for row in _sample_rows(size, wanted))
        picks = sorted(set(picks))
        for region, (height, width) in ROI_SIZES.items():
            got = cached.frames(picks, region, height, width)
            expected = plain.frames(picks, region, height, width)
            records.append(
                {
                    "region": region,
                    "indices": len(picks),
                    "shape": [int(value) for value in got.shape],
                    "dtype": str(got.dtype),
                    "equal": bool(np.array_equal(got, expected)),
                    "mismatched_values": int((got != expected).sum()),
                }
            )
        stats = cached.decode_cache_stats()
    finally:
        cached.close()
        plain.close()
    return {"frames_checks": records, "decode_cache_stats": stats}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shards", type=int, default=8, help="how many shards to sample")
    parser.add_argument("--frames", type=int, default=8, help="rows sampled per shard/region")
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset)
    dataset = VisionDataset(dataset_dir, use_decode_cache=False)
    try:
        total_shards = len(dataset.offsets)
        if total_shards == 0:
            raise SystemExit("no shards in dataset")
        wanted = max(1, min(int(args.shards), total_shards))
        step = max(1, total_shards // wanted)
        shard_indexes = sorted({min(index, total_shards - 1) for index in range(0, total_shards, step)})
        shard_indexes = shard_indexes[:wanted]
        if total_shards - 1 not in shard_indexes:
            shard_indexes[-1] = total_shards - 1
        region_records = []
        for position, shard_index in enumerate(shard_indexes):
            path, start, size = dataset.offsets[shard_index]
            for region, (height, width) in ROI_SIZES.items():
                if region not in dataset.region_hw:
                    continue
                region_records.append(
                    check_region(
                        dataset,
                        path,
                        start,
                        size,
                        region,
                        height,
                        width,
                        full=(position == 0),  # one whole file, every row and pixel
                    )
                )
    finally:
        dataset.close()

    path_check = check_dataset_path(dataset_dir, shard_indexes, int(args.frames))
    ok = all(record["equal"] for record in region_records) and all(
        record["equal"] for record in path_check["frames_checks"]
    )
    payload = {
        "contract_version": "v104-pixel-consistency/1",
        "dataset": str(dataset_dir),
        "shards_sampled": shard_indexes,
        "shards_total": total_shards,
        "region_files_checked": len(region_records),
        "all_equal": ok,
        "region_checks": region_records,
        **path_check,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "dataset": str(dataset_dir),
        "region_files_checked": len(region_records),
        "all_region_files_equal": all(record["equal"] for record in region_records),
        "frames_path_equal": all(record["equal"] for record in path_check["frames_checks"]),
        "values_compared": sum(record["rows_checked"] for record in region_records),
        "mismatched_values": sum(record["mismatched_values"] for record in region_records),
        "dtypes": sorted({record["dtype"] for record in region_records}),
        "decode_cache_stats": path_check["decode_cache_stats"],
        "out": str(out),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
