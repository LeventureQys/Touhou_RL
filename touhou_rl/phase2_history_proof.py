"""Independent Phase-2 history isolation proof.

This verifier intentionally does not import trainer proof code.  It loads the source
vision shards, cache episode labels, and preregistered protocol, then asks the real
``VisionDataset.stack_indices`` implementation for every student center frame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .vision_data import VisionDataset

SCHEMA = "v200-phase2-history-proof/1"
DEPTH = 3


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _load_protocol(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("protocol must be an object")
    episodes = obj.get("episodes")
    if not isinstance(episodes, dict):
        exp = obj.get("experiment")
        splits = exp.get("splits") if isinstance(exp, dict) else None
        if isinstance(splits, dict):
            episodes = {k.removeprefix("student_"): v for k, v in splits.items()
                        if k in ("student_train", "student_dev", "student_test")}
    if not isinstance(episodes, dict) or set(episodes) != {"train", "dev", "test"}:
        raise ValueError("protocol must declare episodes train/dev/test")
    out = {}
    seen: dict[int, str] = {}
    for split in ("train", "dev", "test"):
        values = episodes[split]
        if not isinstance(values, list) or not values:
            raise ValueError(f"protocol episode split {split} is empty or invalid")
        out[split] = []
        for ep in values:
            if isinstance(ep, bool) or not isinstance(ep, (int, str)):
                raise ValueError("episode ids must be integers or strings")
            key = str(ep)
            if key in seen:
                raise ValueError(f"episode appears in multiple protocol splits: {ep}")
            seen[key] = split
            out[split].append(ep)
    return out


def _source_report(source: Path) -> Path:
    if source.is_file():
        return source
    for name in ("student-rollout-report.json", "source-report.json", "report.json"):
        candidate = source / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("source report JSON not found")


def _cache_arrays(cache: Path) -> tuple[dict[str, np.ndarray], Path]:
    identity = cache / "chunk-identity.json"
    arrays_path = cache / "chunk-arrays.npz"
    if not identity.exists() or not arrays_path.exists():
        raise FileNotFoundError("cache requires chunk-identity.json and chunk-arrays.npz")
    with np.load(arrays_path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    if "episode" not in arrays:
        raise ValueError("cache missing episode mapping")
    if "sample_index" in arrays and not np.array_equal(arrays["sample_index"], np.arange(len(arrays["episode"]))):
        raise ValueError("cache sample_index is not contiguous")
    return arrays, identity


def _source_segments(dataset: VisionDataset) -> list[dict]:
    segments = []
    for path, start, size in dataset.offsets:
        meta = dataset.meta_of_shard(path)
        if len(meta) != size:
            raise ValueError(f"source shard meta length mismatch: {path.name}")
        breaks = []
        for i in range(1, len(meta)):
            reasons = []
            if int(meta[i, 0]) - int(meta[i - 1, 0]) != 2:
                reasons.append("frame_gap")
            if int(meta[i, 1]) != int(meta[i - 1, 1]):
                reasons.append("stage")
            if int(meta[i, 7]) != int(meta[i - 1, 7]):
                reasons.append("death")
            if reasons:
                breaks.append({"local_index": i, "global_index": int(start + i), "reasons": reasons,
                               "before": meta[i - 1].tolist(), "after": meta[i].tolist()})
        segments.append({"shard": path.name, "start": int(start), "rows": int(size), "breaks": breaks})
    return segments


def verify(source: str | Path, cache: str | Path, protocol: str | Path, out: str | Path) -> dict:
    source, cache, protocol, out = map(Path, (source, cache, protocol, out))
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    report_path = _source_report(source)
    protocol_obj = _load_protocol(protocol)
    arrays, identity_path = _cache_arrays(cache)
    episodes = np.asarray(arrays["episode"]).reshape(-1)
    source_dir = source if source.is_dir() else source.parent
    with VisionDataset(source_dir, history_contract="causal-segment-v1", decode_threads=1,
                       use_decode_cache=False) as dataset:
        if len(dataset) != len(episodes):
            raise ValueError(f"cache/source row mismatch: {len(episodes)} != {len(dataset)}")
        split_by_ep = {str(ep): split for split, values in protocol_obj.items() for ep in values}
        unknown = sorted({str(ep) for ep in episodes.tolist()} - set(split_by_ep))
        if unknown:
            raise ValueError(f"unknown episode ids in cache: {unknown}")
        centers = list(range(len(dataset)))
        history = np.asarray(dataset.stack_indices(centers, DEPTH), dtype=np.int64)
        if history.shape != (len(centers), DEPTH):
            raise ValueError("stack_indices returned unexpected shape")
        center_split = [split_by_ep[str(ep)] for ep in episodes.tolist()]
        crossings = []
        for center, row in enumerate(history.tolist()):
            for slot, index in enumerate(row):
                if index < 0 or index >= len(episodes):
                    crossings.append({"center": center, "slot": slot, "index": int(index), "reason": "out_of_range"})
                    continue
                reasons = []
                if str(episodes[index]) != str(episodes[center]): reasons.append("episode")
                if split_by_ep[str(episodes[index])] != center_split[center]: reasons.append("split")
                if reasons:
                    crossings.append({"center": center, "slot": slot, "index": int(index), "center_episode": episodes[center].item(),
                                      "history_episode": episodes[index].item(), "center_split": center_split[center],
                                      "history_split": split_by_ep[str(episodes[index])], "reasons": reasons})
        history_digest = hashlib.sha256(np.ascontiguousarray(history).tobytes()).hexdigest()
        report = {"schema": SCHEMA, "source_report": str(report_path), "source_report_sha256": sha(report_path),
                  "cache": str(cache.resolve()), "cache_sha256": sha(identity_path), "protocol": str(protocol.resolve()),
                  "protocol_sha256": sha(protocol), "history_index_sha256": history_digest, "depth": DEPTH,
                  "total_rows": len(centers), "crossing_count": len(crossings), "crossing_samples": crossings[:100],
                  "source_shard_meta_death_gap_segments": _source_segments(dataset), "passed": not crossings}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.source, args.cache, args.protocol, args.out), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
