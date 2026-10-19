import json, time
from pathlib import Path
from touhou_rl.pipeline.manifest import DatasetManifest, index_digest
root = Path("third_party/th08/build/rl/datasets/pilot-ud0059-v2")
manifest = DatasetManifest.load(root / "manifest.json")
started = time.time()
train, val = manifest.verify(root, deep=True)
report = {
    "mode": "a-manifest-verify-deep",
    "dataset": str(root),
    "manifest_sha256_note": "shard sizes and SHA-256 re-hashed from disk, split re-derived",
    "total_frames": manifest.total_frames,
    "shards": len(manifest.shards),
    "train_frames": len(train),
    "val_frames": len(val),
    "val_index_sha256": index_digest(val),
    "manifest_val_index_sha256": manifest.splits["val_index_sha256"],
    "matches_manifest": index_digest(val) == manifest.splits["val_index_sha256"],
    "has_teacher_values": manifest.has_teacher_values,
    "elapsed_seconds": round(time.time() - started, 1),
}
out = Path(r"Document/Update/Alpha/v1.0.4/Stage3/S3-1/evidence/a-manifest-verify.json")
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
