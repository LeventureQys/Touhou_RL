#!/usr/bin/env python
"""S3-8 full-validation re-evaluation of the frozen checkpoints (A, 7241 frames).

This is evidence tooling, not business code: it only calls the public APIs of
``touhou_rl`` (``VisionDataset``, ``load_vision_checkpoint``,
``regions_to_batch``, ``vision_train.policy_forward``/``coverage_record``,
``vision_metrics.all_state_metrics``/``policy_metrics``) exactly the way the
trainer's own validation pass does, so the numbers it prints can be compared with
``vision-summary.json`` without a second metric implementation.

Two kinds:

``state_recheck``
    A-01: the frozen ``state-pass.pt`` over every validation index, with the
    position error (``(|dx|*384 + |dy|*448)/2``, native px) and the per-frame
    danger Pearson median/valid frames.

``policy_full_val``
    A-02: the frozen ``policy.pt`` over every validation index, with the
    teacher-agreement metrics (0-15 bucket regret, top3, by_danger) computed on
    the *raw* teacher px bound to the fingerprinted cache.  The position/danger
    numbers of the same checkpoint are reported separately as diagnostics - they
    never replace A-01.

Everything the JSON claims is recomputable from it: the manifest/checkpoint
digests, the visited-index digest, the per-frame arrays' summary statistics and
the exact command line.  A failure to cover all 7241 validation indices is
``v104:COVERAGE`` (exit 2), never a silent partial run.

    python Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/full_val_recheck.py \
        --kind state_recheck \
        --checkpoint third_party/th08/build/rl/runs/v104-v3-E5-s7/state-pass.pt \
        --out Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/state-recheck.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
import torch  # noqa: E402

from touhou_rl import vision_metrics, vision_train  # noqa: E402
from touhou_rl.pipeline.manifest import (  # noqa: E402
    DatasetManifest,
    index_digest,
    split_indices,
)
from touhou_rl.vision import regions_to_batch  # noqa: E402
from touhou_rl.vision_data import VisionDataset  # noqa: E402
from touhou_rl.vision_history import HISTORY_LEGACY, normalize_history_contract, v104_error  # noqa: E402
from touhou_rl.vmodels import load_vision_checkpoint, vision_arch  # noqa: E402

DATASET_A = ROOT / "third_party" / "th08" / "build" / "rl" / "datasets" / "pilot-ud0059-v2"
VAL_FRAMES = 7241
TOTAL_FRAMES = 72413
VAL_INDEX_SHA256 = "e5d4ebbe6d8de05793520a393fc1fdbea2b1e2c4db9ff7c1632908340c76655f"
POS_SCALE_X = 384
POS_SCALE_Y = 448
BATCH = 96


def rel(path) -> str:
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def position_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return (
        np.abs(pred[:, 0] - truth[:, 0]) * POS_SCALE_X
        + np.abs(pred[:, 1] - truth[:, 1]) * POS_SCALE_Y
    ) / 2.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("state_recheck", "policy_full_val"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default=str(DATASET_A))
    parser.add_argument("--manifest", default="")
    parser.add_argument("--teacher-cache", default="")
    parser.add_argument("--teacher-fingerprint", default="")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise v104_error("CHECKPOINT", f"checkpoint not found: {checkpoint}")
    checkpoint_sha = digest(checkpoint)
    dataset_dir = Path(args.dataset).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else dataset_dir / "manifest.json"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, extra = load_vision_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()
    arch = vision_arch(model)
    depth = int(model.config.depth)
    sizes = {name: tuple(int(v) for v in size) for name, size in model.config.region_sizes.items()}
    if isinstance(extra, dict) and extra.get("history_contract") is not None:
        history_contract = normalize_history_contract(extra["history_contract"])
        legacy_history = False
    else:
        history_contract, legacy_history = HISTORY_LEGACY, True

    # --- manifest and split verification (never re-derive silently) ----------
    manifest = DatasetManifest.load(manifest_path)
    manifest.verify(dataset_dir, deep=True)
    splits = dict(manifest.splits or {})
    _train, val = split_indices(int(manifest.total_frames), int(splits["seed"]), float(splits["val_fraction"]))
    val_digest = index_digest(val)
    if val_digest != splits.get("val_index_sha256"):
        raise v104_error("DATA", f"split digest {val_digest} != manifest {splits.get('val_index_sha256')}")
    if val_digest != VAL_INDEX_SHA256:
        raise v104_error("DATA", f"split digest {val_digest} != the frozen A validation digest")
    if len(val) != VAL_FRAMES or int(manifest.total_frames) != TOTAL_FRAMES:
        raise v104_error("DATA", f"val/total frames {len(val)}/{manifest.total_frames} != {VAL_FRAMES}/{TOTAL_FRAMES}")

    teacher_values = None
    teacher_block: dict = {}
    if args.kind == "policy_full_val":
        if not args.teacher_cache:
            raise v104_error("TEACHER", "--kind policy_full_val needs --teacher-cache")
        cache = Path(args.teacher_cache).resolve()
        fingerprint = args.teacher_fingerprint
        if not fingerprint:
            raise v104_error("TEACHER", "--kind policy_full_val needs --teacher-fingerprint")
        values, reason = vision_train._inspect_teacher_cache(cache, fingerprint, TOTAL_FRAMES)
        if reason != "ok":
            raise v104_error("TEACHER", f"teacher cache rejected by the trainer's own check: {reason}")
        teacher_values = values
        teacher_block = {
            "cache": rel(cache),
            "cache_sha256": digest(cache),
            "fingerprint": fingerprint,
            "dtype": str(values.dtype),
            "shape": list(values.shape),
            "verified_by": "vision_train._inspect_teacher_cache (schema/fingerprint/shape/dtype/finite/digest)",
        }

    order = [int(index) for index in val]
    positions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    danger_pred: list[np.ndarray] = []
    danger_true: list[np.ndarray] = []
    heatmaps: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    teachers: list[np.ndarray] = []
    bullets: list[np.ndarray] = []
    stages: list[np.ndarray] = []
    visited: list[int] = []
    per_frame_error: list[np.ndarray] = []
    per_frame_danger_valid: list[np.ndarray] = []

    with VisionDataset(dataset_dir, history_contract=history_contract) as dataset:
        if len(dataset) != TOTAL_FRAMES:
            raise v104_error("DATA", f"dataset has {len(dataset)} frames, manifest says {manifest.total_frames}")
        for start in range(0, len(order), int(args.batch)):
            chunk = order[start : start + int(args.batch)]
            visited.extend(chunk)
            history = dataset.stack_indices(chunk, depth)
            flat = history.reshape(-1).tolist()
            frames = dataset.region_frames(flat, sizes)
            stacked = {
                name: array.reshape(len(chunk), depth, sizes[name][0], sizes[name][1], 3)
                for name, array in frames.items()
            }
            batch = regions_to_batch(stacked, device)
            labels = dataset.labels(chunk)
            truth_self = np.asarray(labels["self_feat"], dtype=np.float64)
            with torch.no_grad():
                state = model.decode_state(batch)
                pred_self = state["self_feat"].detach().cpu().numpy().astype(np.float64)
                pred_position = pred_self[:, :2]
                pred_danger = state["danger_log"].detach().cpu().numpy().astype(np.float64)
                heatmap = None
                if "heatmap_logits" in state:
                    heatmap = state["heatmap_logits"].detach().cpu().numpy().astype(np.float64)
                if args.kind == "policy_full_val":
                    predicted, _aux = vision_train.policy_forward(model, batch)
                    scores.append(predicted.detach().cpu().numpy().astype(np.float64))
            errors = position_error(pred_position, truth_self)
            if not np.isfinite(errors).all() or not np.isfinite(pred_danger).all():
                raise v104_error("DATA", f"non-finite prediction on indices {chunk[:3]}")
            per_frame_error.append(errors)
            # labels["danger"] is the per-cell count in the model's [B, 16, 8]
            # grid; vision_metrics flattens it to [B, 128] (``_as_2d``), so the
            # same reshape is used here for the validity mask.
            danger_cells = np.asarray(labels["danger"], dtype=np.float64).reshape(len(chunk), -1)
            # per-frame danger validity uses exactly the definition of
            # vision_metrics.danger_metrics, so the median can be recomputed.
            truth_log = np.log1p(danger_cells / vision_metrics.DANGER_SCALE)
            centred_pred = pred_danger - pred_danger.mean(axis=1, keepdims=True)
            centred_true = truth_log - truth_log.mean(axis=1, keepdims=True)
            norms = np.sqrt((centred_pred * centred_pred).sum(axis=1)) * np.sqrt(
                (centred_true * centred_true).sum(axis=1)
            )
            per_frame_danger_valid.append(np.isfinite(pred_danger).all(axis=1) & (norms > 0.0))
            positions.append(pred_self)
            truths.append(truth_self)
            danger_pred.append(pred_danger)
            danger_true.append(danger_cells)
            if heatmap is not None:
                heatmaps.append(heatmap)
            stages.append(np.asarray(labels["meta"], dtype=np.float64)[:, 1].astype(np.int64))
            if teacher_values is not None:
                teachers.append(np.asarray(teacher_values[chunk], dtype=np.float32))
                bullets.append(np.asarray(labels["bullets"], dtype=np.float32))

    coverage = vision_train.coverage_record(order, visited)
    if not coverage["complete"] or coverage["actual_frames"] != VAL_FRAMES:
        raise v104_error("COVERAGE", f"validation coverage is incomplete: {coverage}")

    pred_all = np.concatenate(positions)
    truth_all = np.concatenate(truths)
    error_all = np.concatenate(per_frame_error)
    stage_all = np.concatenate(stages)
    metrics = vision_metrics.all_state_metrics(
        pred_all,
        np.concatenate(danger_pred),
        truth_all,
        np.concatenate(danger_true),
        heatmap_logits=np.concatenate(heatmaps) if heatmaps else None,
    )
    distribution = {
        "mean": float(error_all.mean()),
        "median": float(np.median(error_all)),
        "p90": float(np.percentile(error_all, 90)),
        "p95": float(np.percentile(error_all, 95)),
        "p99": float(np.percentile(error_all, 99)),
        "max": float(error_all.max()),
        "under_10px_share": float((error_all <= 10).mean()),
        "tail_gt30_share": float((error_all > 30).mean()),
    }
    by_stage = {}
    for stage in sorted(set(int(value) for value in stage_all.tolist())):
        mask = stage_all == stage
        values = error_all[mask]
        by_stage[str(stage)] = {
            "frames": int(mask.sum()),
            "position_mean_px": float(values.mean()),
            "position_median_px": float(np.median(values)),
            "position_p90_px": float(np.percentile(values, 90)),
        }

    payload = {
        "schema": "v104-s3-8-full-val/1",
        "kind": args.kind,
        "created_at": now_iso(),
        "producer": "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/full_val_recheck.py",
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "checkpoint": rel(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "arch": arch,
        "depth": depth,
        "history_contract": history_contract,
        "legacy_history": bool(legacy_history),
        "parameters": int(model.num_parameters()),
        "dataset": rel(dataset_dir),
        "manifest": rel(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "val_index_sha256": val_digest,
        "frames": int(coverage["actual_frames"]),
        "coverage": coverage,
        "metrics": metrics,
        "position_distribution_px": distribution,
        "by_stage": by_stage,
        "teacher": teacher_block,
        "note": (
            "state_recheck = A-01 object; policy_full_val position/danger are diagnostics of the"
            " final policy checkpoint and never replace A-01"
            if args.kind == "policy_full_val"
            else "A-01 full re-evaluation of the frozen state-pass.pt"
        ),
    }
    observations = {
        "checkpoint_sha256": checkpoint_sha,
        "frames": int(coverage["actual_frames"]),
        "coverage_sha256": coverage["visited_index_sha256"],
        "player_pos_px": float(metrics["player_pos_px"]),
        "danger_corr_perframe_median": float(metrics["danger_corr_perframe_median"]),
        "danger_corr_perframe_valid": int(metrics["danger_corr_perframe_valid"]),
    }
    if args.kind == "policy_full_val":
        policy = vision_metrics.policy_metrics(
            np.concatenate(scores), np.concatenate(teachers), np.concatenate(bullets)
        )
        payload["policy_metrics"] = policy
        payload["policy_checkpoint_stage"] = "policy"
        observations["by_danger"] = policy["by_danger"]
        observations["top3"] = float(policy["top3"])
        observations["regret_px"] = float(policy["regret_px"])
        observations["position_diagnostic"] = {
            "player_pos_px": float(metrics["player_pos_px"]),
            "danger_corr_perframe_median": float(metrics["danger_corr_perframe_median"]),
            "note": "final policy checkpoint perception - diagnostic only, A-01 stays with state-pass.pt",
        }
    payload["observations"] = observations

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "kind": args.kind,
                "checkpoint_sha256": checkpoint_sha[:16],
                "frames": observations["frames"],
                "coverage_sha256": observations["coverage_sha256"][:16],
                "player_pos_px": observations["player_pos_px"],
                "danger_corr_perframe_median": observations["danger_corr_perframe_median"],
                "danger_corr_perframe_valid": observations["danger_corr_perframe_valid"],
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.kind == "policy_full_val":
        bucket = next((row for row in payload["policy_metrics"]["by_danger"] if row["bucket"] == "0-15"), None)
        print(
            json.dumps(
                {
                    "bucket_0_15": bucket,
                    "top3": payload["policy_metrics"]["top3"],
                    "regret_px": payload["policy_metrics"]["regret_px"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    print(f"written: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        message = str(exc)
        if message.startswith("v104:"):
            print(message, file=sys.stderr, flush=True)
            raise SystemExit(2)
        raise
