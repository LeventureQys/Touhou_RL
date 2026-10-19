#!/usr/bin/env python
"""S3-8 A-06 re-check: D9 (coverage / tail batch), D10 (budget / LRU / values), D11 (B cache).

The task book allows citing the S3-3 / S3-1 evidence for D9/D10/D11 *after*
re-checking its sha256 and the current state.  This script does exactly that and
produces three fresh, machine-readable artifacts whose observations are the
contracted fields:

``d9-recheck.json``
    Re-derives the fixed validation split from A's manifest, reads the coverage
    the formal E5 run recorded, and re-runs the trainer's own ``make_batches`` to
    prove ``ceil(7241/96) = 76`` batches with a **41**-frame tail.

``d10-recheck.json``
    Re-runs the two residency paths on B (cached: label records only; npz: every
    full shard record in order, which forces byte-budget eviction), samples
    ``cache_stats()`` after every load, and adds an explicit value-equality test:
    a frame set is read, evicted by loading other shards, then re-read and
    compared byte for byte.  The default budget constant is read from the module
    (``DEFAULT_CACHE_BUDGET_BYTES``) so a changed constant cannot pass.

``d11-recheck.json``
    Verifies all of B's decoded-cache files against their sidecars *and* their
    source ``.npz`` digests (each shard hashed once, not once per region),
    counts the files and bytes on disk, and proves pixel equality by reading the
    same frames twice - once through the decode cache, once by decoding the source
    ``.npz`` PNGs - and comparing the arrays.

Exit code: 0 when every required observation holds, 1 otherwise (the JSON says
which one failed; nothing is silently repaired).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path


def _repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "touhou_rl" / "vision_data.py").exists():
            return candidate
    raise SystemExit("cannot locate the repository root")


ROOT = _repo_root(Path(__file__).resolve().parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from touhou_rl import vision_data  # noqa: E402
from touhou_rl.pipeline.manifest import (  # noqa: E402
    DatasetManifest,
    file_digest,
    index_digest,
    split_indices,
)
from touhou_rl.vision_data import (  # noqa: E402
    DEFAULT_CACHE_BUDGET_BYTES,
    ROI_SIZES,
    VisionDataset,
    check_decode_cache,
    discover_vision_shards,
    read_decode_sidecar,
)
from touhou_rl.vision_train import make_batches  # noqa: E402

DATASET_A = ROOT / "third_party" / "th08" / "build" / "rl" / "datasets" / "pilot-ud0059-v2"
DATASET_B = ROOT / "third_party" / "th08" / "build" / "rl" / "datasets" / "pilot-ud0063"
RUN_SUMMARY = ROOT / "third_party" / "th08" / "build" / "rl" / "runs" / "v104-v3-E5-s7" / "vision-summary.json"
MANIFEST_B = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "S3-1" / "evidence" / "ud0063-evaluation-manifest.json"
S33_THROUGHPUT = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "S3-3" / "evidence" / "throughput.json"
S31_CACHE_B = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "S3-1" / "evidence" / "cache-B.json"
S31_BUDGET_B_CACHED = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "S3-1" / "evidence" / "budget-B-cached.json"
S31_BUDGET_B_NPZ = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "S3-1" / "evidence" / "budget-B-npz.json"
S31_PIXEL_B = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "S3-1" / "evidence" / "pixel-consistency-B.json"
VAL_FRAMES = 7241
BATCH = 96
BUDGET = 536_870_912


def rel(path) -> str:
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def dig(path: Path) -> str:
    return file_digest(path)


def reference_block(paths: list[Path]) -> dict:
    """sha256 of every earlier-stage artifact this re-check cites."""
    block = {}
    for path in paths:
        entry = {"path": rel(path), "sha256": None, "bytes": None, "exists": path.is_file()}
        if path.is_file():
            entry["sha256"] = dig(path)
            entry["bytes"] = path.stat().st_size
        block[path.name] = entry
    return block


# --------------------------------------------------------------------------- #
# D9
# --------------------------------------------------------------------------- #
def check_d9(out: Path) -> dict:
    started = time.perf_counter()
    summary = json.loads(RUN_SUMMARY.read_text(encoding="utf-8"))
    coverage = ((summary.get("coverage") or {}).get("val")) or {}
    manifest = DatasetManifest.load(DATASET_A / "manifest.json")
    splits = dict(manifest.splits or {})
    train, val = split_indices(int(manifest.total_frames), int(splits["seed"]), float(splits["val_fraction"]))
    digest = index_digest(val)
    batches = make_batches(val, batch_size=BATCH, shuffle=False, drop_last=False)
    sizes = [len(batch) for batch in batches]
    last_batch_size = sizes[-1] if sizes else None
    expected_batches = math.ceil(len(val) / BATCH)
    refs = reference_block([S33_THROUGHPUT])
    throughput = json.loads(S33_THROUGHPUT.read_text(encoding="utf-8")) if S33_THROUGHPUT.is_file() else {}
    cited = []
    for round_name in ("round1", "round2"):
        for run in (throughput.get(round_name) or {}).get("runs", []):
            cited.append(
                {
                    "round": round_name,
                    "depth": run.get("depth"),
                    "mode": run.get("mode"),
                    "coverage_val_actual_frames": run.get("coverage_val_actual_frames"),
                    "coverage_val_complete": run.get("coverage_val_complete"),
                    "coverage_val_sha256": run.get("coverage_val_sha256"),
                    "val_frames": run.get("val_frames"),
                }
            )
    cited_ok = bool(cited) and all(
        row["coverage_val_actual_frames"] == VAL_FRAMES
        and row["coverage_val_complete"] is True
        and row["coverage_val_sha256"] == digest
        for row in cited
    )
    observations = {
        "expected_frames": int(coverage.get("expected_frames")),
        "actual_frames": int(coverage.get("actual_frames")),
        "visited_index_sha256": coverage.get("visited_index_sha256"),
        "last_batch_size": int(last_batch_size),
        "batches": len(sizes),
        "expected_batches": int(expected_batches),
        "duplicated_indices": int(coverage.get("duplicated_indices", -1)),
        "missing_indices": int(coverage.get("missing_indices", -1)),
        "unexpected_indices": int(coverage.get("unexpected_indices", -1)),
        "complete": bool(coverage.get("complete")),
        "manifest_val_index_sha256": digest,
        "cited_s3_3_runs": len(cited),
        "cited_s3_3_coverage_ok": cited_ok,
        "run_summary": rel(RUN_SUMMARY),
        "run_summary_sha256": dig(RUN_SUMMARY),
        "reference_files": refs,
    }
    ok = (
        observations["expected_frames"] == VAL_FRAMES
        and observations["actual_frames"] == VAL_FRAMES
        and observations["visited_index_sha256"] == digest == splits.get("val_index_sha256")
        and observations["last_batch_size"] == VAL_FRAMES - (expected_batches - 1) * BATCH
        and observations["batches"] == expected_batches
        and observations["complete"]
        and observations["duplicated_indices"] == 0
        and observations["missing_indices"] == 0
        and observations["unexpected_indices"] == 0
        and cited_ok
    )
    payload = {
        "schema": "v104-s3-8-a06-d9/1",
        "kind": "d9",
        "created_at": now_iso(),
        "producer": "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/d9d10d11_recheck.py",
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "observations": observations,
        "batch_sizes_head": sizes[:3],
        "batch_sizes_tail": sizes[-3:],
        "cited_runs": cited,
        "status": "pass" if ok else "fail",
    }
    (out / "d9-recheck.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"d9": payload["status"], "observations": observations}, ensure_ascii=False), flush=True)
    return payload


# --------------------------------------------------------------------------- #
# D10
# --------------------------------------------------------------------------- #
def _dataset_b(*, use_decode_cache: bool, budget: int | None = None, verify_source: bool = False) -> VisionDataset:
    kwargs = {"use_decode_cache": use_decode_cache, "verify_source_digest": verify_source}
    if budget is not None:
        kwargs["cache_budget_bytes"] = int(budget)
    return VisionDataset(DATASET_B, **kwargs)


def check_d10(out: Path) -> dict:
    started = time.perf_counter()
    reference = json.loads(S31_BUDGET_B_CACHED.read_text(encoding="utf-8")) if S31_BUDGET_B_CACHED.is_file() else {}
    reference_npz = json.loads(S31_BUDGET_B_NPZ.read_text(encoding="utf-8")) if S31_BUDGET_B_NPZ.is_file() else {}

    # cached path: label records only (real training shape), decode cache on
    cached_peak = 0
    cached_over = 0
    cached_loads = 0
    dataset = _dataset_b(use_decode_cache=True, verify_source=True)
    budget_reported = int(dataset.cache_stats()["budget_bytes"])
    try:
        stride = 8
        samples = list(range(0, len(dataset), stride))
        for offset in range(0, len(samples), BATCH):
            batch = samples[offset : offset + BATCH]
            dataset.labels(batch)
            for region, (height, width) in ROI_SIZES.items():
                dataset.frames(batch, region, height, width)
            stats = dataset.cache_stats()
            cached_loads += 1
            cached_peak = max(cached_peak, int(stats["resident_bytes"]))
            cached_over += int(stats["resident_bytes"] > stats["budget_bytes"])
        cached_final = dataset.cache_stats()
        cached_decode = dataset.decode_cache_stats()
    finally:
        dataset.close()

    # npz path: every full shard record in order, which forces eviction
    npz_peak = 0
    npz_over = 0
    npz_loads = 0
    dataset = _dataset_b(use_decode_cache=False)
    try:
        for _path, start, _size in dataset.offsets:
            dataset.frames([start], "playfield", *ROI_SIZES["playfield"])
            stats = dataset.cache_stats()
            npz_loads += 1
            npz_peak = max(npz_peak, int(stats["resident_bytes"]))
            npz_over += int(stats["resident_bytes"] > stats["budget_bytes"])
        npz_final = dataset.cache_stats()
    finally:
        dataset.close()

    # LRU + value equality: a hit must be recognised (LRU touch), an eviction must
    # really drop the record (the re-read is a miss) and the values must survive.
    small_budget = 96 * 1024 * 1024
    dataset = _dataset_b(use_decode_cache=False, budget=small_budget)
    try:
        path0, start0, size0 = dataset.offsets[0]
        idx0 = [int(start0)]
        before = np.asarray(dataset.frames(idx0, "playfield", *ROI_SIZES["playfield"]))
        stats_first = dataset.cache_stats()
        again = np.asarray(dataset.frames(idx0, "playfield", *ROI_SIZES["playfield"]))
        stats_hit = dataset.cache_stats()
        hit_recognised = bool(stats_hit["hits"] == stats_first["hits"] + 1)
        same_on_hit = bool(np.array_equal(before, again))
        touched = 0
        for path, start, size in dataset.offsets[1:]:
            dataset.frames([int(start)], "playfield", *ROI_SIZES["playfield"])
            touched += 1
            if dataset.cache_stats()["evictions"] > 0:
                break
        stats_after = dataset.cache_stats()
        misses_before_reload = int(stats_after["misses"])
        after = np.asarray(dataset.frames(idx0, "playfield", *ROI_SIZES["playfield"]))
        stats_reload = dataset.cache_stats()
        reload_was_miss = bool(stats_reload["misses"] == misses_before_reload + 1)
        value_equal = bool(before.shape == after.shape and np.array_equal(before, after))
    finally:
        dataset.close()
    lru_verified = bool(
        hit_recognised and stats_after["evictions"] > 0 and reload_was_miss and same_on_hit and value_equal
    )

    observations = {
        "budget_bytes": int(budget_reported),
        "max_resident_bytes": int(max(cached_peak, npz_peak)),
        "lru_verified": lru_verified,
        "value_equal": value_equal,
        "cached_peak_resident_bytes": int(cached_peak),
        "npz_peak_resident_bytes": int(npz_peak),
        "over_budget_samples": int(cached_over + npz_over),
        "cached_loads": int(cached_loads),
        "npz_loads": int(npz_loads),
        "cached_evictions": int(cached_final["evictions"]),
        "npz_evictions": int(npz_final["evictions"]),
        "lru_small_budget_bytes": int(small_budget),
        "lru_evictions": int(stats_after["evictions"]),
        "lru_shards_touched": int(touched),
        "lru_hit_recognised": hit_recognised,
        "lru_hits_after_reload": int(stats_reload["hits"]),
        "lru_reload_was_miss": reload_was_miss,
        "lru_values_equal_on_hit": same_on_hit,
        "hits_before_eviction": int(stats_hit["hits"]),
        "misses_before_reload": misses_before_reload,
        "default_cache_budget_constant": int(DEFAULT_CACHE_BUDGET_BYTES),
        "dataset": rel(DATASET_B),
        "reference_files": reference_block(
            [S31_BUDGET_B_CACHED, S31_BUDGET_B_NPZ, S31_CACHE_B, S31_PIXEL_B]
        ),
    }
    ok = (
        observations["budget_bytes"] == BUDGET
        and DEFAULT_CACHE_BUDGET_BYTES == BUDGET
        and observations["max_resident_bytes"] <= BUDGET
        and observations["over_budget_samples"] == 0
        and lru_verified
        and value_equal
        and int(reference.get("budget_bytes", -1)) == BUDGET
        and int(reference_npz.get("peak_resident_bytes", -1)) <= BUDGET
    )
    payload = {
        "schema": "v104-s3-8-a06-d10/1",
        "kind": "d10",
        "created_at": now_iso(),
        "producer": "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/d9d10d11_recheck.py",
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "observations": observations,
        "cached_final_cache_stats": cached_final,
        "npz_final_cache_stats": npz_final,
        "cached_decode_cache_stats": cached_decode,
        "s3_1_reference": {
            "cached_peak_resident_bytes": reference.get("peak_resident_bytes"),
            "npz_peak_resident_bytes": reference_npz.get("peak_resident_bytes"),
            "cached_within_budget": reference.get("within_budget"),
            "npz_within_budget": reference_npz.get("within_budget"),
        },
        "status": "pass" if ok else "fail",
    }
    (out / "d10-recheck.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "d10": payload["status"],
                "budget_bytes": observations["budget_bytes"],
                "max_resident_bytes": observations["max_resident_bytes"],
                "lru_verified": lru_verified,
                "value_equal": value_equal,
                "evictions": observations["lru_evictions"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return payload


# --------------------------------------------------------------------------- #
# D11
# --------------------------------------------------------------------------- #
def check_d11(out: Path, sample_frames: int = 6) -> dict:
    started = time.perf_counter()
    shards = discover_vision_shards(DATASET_B)
    expected_files = len(shards) * len(ROI_SIZES)
    validated = 0
    invalid: list[dict] = []
    reasons: dict[str, int] = {}
    bytes_total = 0
    digest_cache: dict[Path, str] = {}
    for shard in shards:
        for region, (height, width) in ROI_SIZES.items():
            source_digest = digest_cache.get(shard)
            if source_digest is None:
                source_digest = dig(shard)
                digest_cache[shard] = source_digest
            check = check_decode_cache(
                shard,
                region,
                height,
                width,
                frames=None,
                verify_source=True,
                source_digest=source_digest,
                verify_payload=True,
            )
            if check.ok:
                validated += 1
                bytes_total += int(check.bytes)
            else:
                reasons[check.reason] = reasons.get(check.reason, 0) + 1
                if len(invalid) < 50:
                    invalid.append({"shard": shard.name, "region": region, "reason": check.reason})

    # pixel equality through the public API: cached read vs source decode
    total_frames = int(DatasetManifest.load(MANIFEST_B).total_frames)
    indices = [int(index) for index in np.linspace(0, total_frames - 2, sample_frames, dtype=np.int64)]
    pixel_rows = []
    with VisionDataset(DATASET_B) as cached, VisionDataset(DATASET_B, use_decode_cache=False) as source:
        for region, (height, width) in ROI_SIZES.items():
            a = np.asarray(cached.frames(indices, region, height, width))
            b = np.asarray(source.frames(indices, region, height, width))
            pixel_rows.append(
                {
                    "region": region,
                    "frames": len(indices),
                    "shape": list(a.shape),
                    "dtype": str(a.dtype),
                    "equal": bool(a.shape == b.shape and np.array_equal(a, b)),
                }
            )
        cache_stats = cached.cache_stats()
        decode_stats = cached.decode_cache_stats()
    value_equal = all(row["equal"] for row in pixel_rows)
    manifest_sha = dig(MANIFEST_B)
    manifest = DatasetManifest.load(MANIFEST_B)
    observations = {
        "expected_files": int(expected_files),
        "validated_files": int(validated),
        "bytes_total": int(bytes_total),
        "value_equal": bool(value_equal),
        "manifest_sha": manifest_sha,
        "shards": len(shards),
        "regions": len(ROI_SIZES),
        "invalid_files": int(expected_files - validated),
        "invalid_reasons": reasons,
        "manifest_total_frames": int(manifest.total_frames),
        "manifest_val_index_sha256": (manifest.splits or {}).get("val_index_sha256"),
        "decode_cache_hits": int(decode_stats.get("hits", 0)),
        "decode_cache_missing": int(decode_stats.get("missing", 0)),
        "cache_stats": cache_stats,
        "pixel_rows": pixel_rows,
        "dataset": rel(DATASET_B),
        "reference_files": reference_block([S31_CACHE_B, S31_PIXEL_B, MANIFEST_B]),
    }
    s31 = json.loads(S31_CACHE_B.read_text(encoding="utf-8")) if S31_CACHE_B.is_file() else {}
    expected_from_s31 = int(((s31.get("verification") or {}).get("files")) or -1)
    bytes_from_s31 = int(((s31.get("verification") or {}).get("bytes_total")) or -1)
    ok = (
        validated == expected_files
        and expected_files == 1626
        and expected_from_s31 == expected_files
        and bytes_from_s31 == bytes_total
        and value_equal
    )
    payload = {
        "schema": "v104-s3-8-a06-d11/1",
        "kind": "d11",
        "created_at": now_iso(),
        "producer": "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/d9d10d11_recheck.py",
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "observations": observations,
        "invalid": invalid,
        "s3_1_reference": {
            "files": expected_from_s31,
            "bytes_total": bytes_from_s31,
            "payload_verified": (s31.get("verification") or {}).get("payload_verified"),
        },
        "status": "pass" if ok else "fail",
    }
    (out / "d11-recheck.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "d11": payload["status"],
                "expected_files": observations["expected_files"],
                "validated_files": observations["validated_files"],
                "bytes_total": observations["bytes_total"],
                "value_equal": value_equal,
                "manifest_sha": manifest_sha[:16],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="evidence directory for the three JSONs")
    parser.add_argument("--only", default="", help="d9|d10|d11 (default: all)")
    parser.add_argument("--sample-frames", type=int, default=6)
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.only in ("", "d9"):
        results["d9"] = check_d9(out)["status"]
    if args.only in ("", "d10"):
        results["d10"] = check_d10(out)["status"]
    if args.only in ("", "d11"):
        results["d11"] = check_d11(out, sample_frames=args.sample_frames)["status"]
    print(json.dumps({"results": results}, ensure_ascii=False), flush=True)
    return 0 if all(status == "pass" for status in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
