"""Build/reuse the fingerprinted v1.0.4 teacher cache for dataset A and audit it.

Run (workspace root, PYTHONPATH set):

    python Document/Update/Alpha/v1.0.4/Stage3/S3-3/evidence/teacher_cache_probe.py \
        --dataset third_party/th08/build/rl/datasets/pilot-ud0059-v2 \
        --out Document/Update/Alpha/v1.0.4/Stage3/S3-3/evidence/teacher-cache.json

Exit 0 on success; the JSON records the cache path, fingerprint, sidecar digest,
rebuild flag and the float16 quantisation audit.  The label precision is not
changed: the audit only *counts* the ties float16 creates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from touhou_rl.pipeline.manifest import DatasetManifest, file_digest
from touhou_rl.vision_data import VisionDataset
from touhou_rl.vision_train import VisionTrainConfig, load_or_build_teacher_values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--history-contract", default="causal-segment-v1")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset)
    manifest = DatasetManifest.load(dataset_dir / "manifest.json")
    train, val = manifest.verify(dataset_dir)
    val_sha = str(manifest.splits["val_index_sha256"])
    cfg = VisionTrainConfig(dataset=dataset_dir, out_dir=Path(args.out).parent, arch="v2", depth=3)
    started = time.perf_counter()
    with VisionDataset(dataset_dir, history_contract=args.history_contract) as dataset:
        cache = load_or_build_teacher_values(
            cfg, dataset, manifest=manifest, val_index_sha256=val_sha
        )
    payload = {
        "dataset": str(dataset_dir),
        "total_frames": manifest.total_frames,
        "train_frames": len(train),
        "val_frames": len(val),
        "val_index_sha256": val_sha,
        "manifest_sha256": file_digest(dataset_dir / "manifest.json"),
        "cache": cache["path"],
        "cache_absolute": cache["path_absolute"],
        "fingerprint": cache["fingerprint"],
        "cache_sha256": cache["cache_sha256"],
        "rebuilt": cache["rebuilt"],
        "explicit": cache["explicit"],
        "dtype": str(cache["values"].dtype),
        "shape": list(cache["values"].shape),
        "audit": cache["audit"],
        "identity": {
            "schema": cache["identity"]["schema"],
            "candidate_order_version": cache["identity"]["candidate_order_version"],
            "label_version": cache["identity"]["label_version"],
            "manifest_sha256": cache["identity"]["manifest_sha256"],
            "val_index_sha256": cache["identity"]["val_index_sha256"],
            "shard_count": cache["identity"]["dataset"]["shard_count"],
            "shard_digest_source": cache["identity"]["dataset"]["shard_digest_source"],
            "first_shard": cache["identity"]["dataset"]["shards"][0],
            "teacher_config_switch_penalty": cache["identity"]["teacher"]["config"]["switch_penalty"],
            "teacher_source_sha256": cache["identity"]["teacher"]["source_sha256"],
        },
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
