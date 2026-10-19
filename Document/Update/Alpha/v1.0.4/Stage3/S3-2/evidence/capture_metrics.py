"""Capture the v1.0.3-era vision metric outputs so S3-2 can prove zero drift.

Run before and after the S3-2 edit with the same interpreter:

    $env:PYTHONPATH = "D:\WorkShop\Touhou_RL"
    python capture_metrics.py <out.json>

The payload only uses the keys that existed before S3-2 (``top1``,
``regret_px``, per-bucket ``bucket/frames/top1/regret_px``, the 12
``all_state_metrics`` keys, ``heatmap_hit_rate`` and the standalone metric
functions), so a post-edit run can be compared byte for byte.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from touhou_rl import vision_metrics as vm


def _policy_case() -> dict:
    rng = np.random.default_rng(20261019)
    frames = 40
    pred = rng.normal(0.0, 1.0, size=(frames, 18)).astype(np.float32)
    teacher = rng.uniform(-8.0, 40.0, size=(frames, 18)).astype(np.float32)
    teacher[:, 3] += 5.0  # make the teacher peak non-degenerate
    bullets = np.zeros((frames, 64, 6), dtype=np.float32)
    for index in range(frames):
        # One valid bullet per frame, distances spread across every bucket.
        distance = [5.0, 20.0, 45.0, 90.0, 500.0][index % 5]
        bullets[index, 0, 0] = distance / 192.0
        bullets[index, 0, 1] = 0.0
        bullets[index, 0, 5] = 1.0
    bullets[7, 0, 5] = 0.0  # frame 7 has no valid bullet -> 120+ bucket
    result = vm.policy_metrics(pred, teacher, bullets)
    keep = {
        "top1": result["top1"],
        "regret_px": result["regret_px"],
        "by_danger": [
            {
                "bucket": bucket["bucket"],
                "frames": bucket["frames"],
                "top1": bucket["top1"],
                "regret_px": bucket["regret_px"],
            }
            for bucket in result["by_danger"]
        ],
    }
    return keep


def _state_case() -> dict:
    rng = np.random.default_rng(4242)
    frames = 12
    truth_self = rng.random((frames, 24)).astype(np.float32)
    truth_self[:, 14] = 0.4
    pred_self = truth_self + rng.normal(0.0, 0.01, size=(frames, 24)).astype(np.float32)
    truth_danger = rng.integers(0, 6, size=(frames, 16, 8)).astype(np.uint8)
    truth_danger[2] = 0  # constant frame: dropped from the per-frame correlation
    pred_danger = np.log1p(truth_danger / 8.0) + rng.normal(0.0, 0.01, size=(frames, 16, 8))
    heatmap = rng.random((frames, 28, 24))
    metrics = vm.all_state_metrics(
        pred_self, pred_danger, truth_self, truth_danger, heatmap_logits=heatmap
    )
    return {
        "all_state_metrics": metrics,
        "player_position_mae_px": list(vm.player_position_mae_px(pred_self, truth_self)),
        "self_mae": vm.self_mae(pred_self, truth_self),
        "ray_mae": vm.ray_mae(pred_self, truth_self),
        "bullet_count_mae": vm.bullet_count_mae(pred_self, truth_self),
        "danger_metrics": vm.danger_metrics(pred_danger, truth_danger),
        "heatmap_28x24": vm.heatmap_hit_rate(heatmap, truth_self),
    }


def _edge_case() -> dict:
    empty_self = np.zeros((0, 24), dtype=np.float32)
    result = {
        "empty_position": list(vm.player_position_mae_px(empty_self, empty_self)),
        "empty_self_mae": vm.self_mae(empty_self, empty_self),
        "empty_ray_mae": vm.ray_mae(empty_self, empty_self),
        "empty_bullet_count_mae": vm.bullet_count_mae(empty_self, empty_self),
        "empty_danger": vm.danger_metrics(np.zeros((0, 128)), np.zeros((0, 16, 8), dtype=np.uint8)),
        "empty_policy": _policy_old_keys(
            vm.policy_metrics(
                np.zeros((0, 18)), np.zeros((0, 18)), np.zeros((0, 64, 6), dtype=np.float32)
            )
        ),
        "empty_heatmap": vm.heatmap_hit_rate(np.zeros((0, 28, 24)), empty_self),
        "constant_danger": vm.danger_metrics(
            np.full((5, 128), 0.3), np.zeros((5, 16, 8), dtype=np.uint8)
        ),
    }
    return result


def _policy_old_keys(result: dict) -> dict:
    return {
        "top1": result["top1"],
        "regret_px": result["regret_px"],
        "by_danger": [
            {
                "bucket": bucket["bucket"],
                "frames": bucket["frames"],
                "top1": bucket["top1"],
                "regret_px": bucket["regret_px"],
            }
            for bucket in result["by_danger"]
        ],
    }


def main() -> int:
    out = sys.argv[1]
    payload = {
        "policy": _policy_case(),
        "state": _state_case(),
        "edge": _edge_case(),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
