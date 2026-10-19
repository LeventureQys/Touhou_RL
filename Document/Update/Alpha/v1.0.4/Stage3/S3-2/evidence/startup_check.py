"""S3-2 startup check: run the real functions, not just imports.

Calls ``policy_loss`` for all three objectives on a float16 teacher label batch
and ``policy_metrics``/``heatmap_hit_rate`` on 50 frames, printing every number
the task book asks for: the non-zero loss, the finite stats dictionary, the
gradient norm, the new ranking diagnostics and the hit rate on both the 28x24
and the 56x48 grid.  No dataset, no model, no game window, no training.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import torch

from touhou_rl import vision_losses as vl
from touhou_rl import vision_metrics as vm


def _teacher_batch(frames: int, seed: int = 7) -> torch.Tensor:
    """Raw pilot clearance in px as recorded: float16, one frame fully tied."""
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(frames, vl.N_CANDIDATES, generator=generator) * 200.0
    values[-1] = 7.0  # a frame whose 18 clearances are all identical
    return values.half()


def _bullets(frames: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bullets = np.zeros((frames, 64, 6), dtype=np.float32)
    for index in range(frames):
        if index % 7 == 0:
            continue  # a frame with no valid bullet -> 120+ bucket
        for slot in range(3):
            bullets[index, slot, 0] = rng.normal(0.0, 0.4)
            bullets[index, slot, 1] = rng.normal(0.0, 0.4)
            bullets[index, slot, 5] = 1.0
        bullets[index, 0, 0] = np.abs(bullets[index, 0, 0])  # keep one side positive
    return bullets


def _heatmap_case(frames: int, height: int, width: int, seed: int = 5) -> tuple:
    """Logits that hit the truth cell on even frames and miss it on odd ones."""
    rng = np.random.default_rng(seed)
    truth = rng.random((frames, 24)).astype(np.float32)
    logits = rng.normal(0.0, 0.1, size=(frames, height, width)).astype(np.float32)
    for index in range(frames):
        want_x = int(np.clip(np.rint(truth[index, 0] * width - 0.5), 0, width - 1))
        want_y = int(np.clip(np.rint(truth[index, 1] * height - 0.5), 0, height - 1))
        if index % 2 == 0:
            logits[index, want_y, want_x] = 5.0
        else:
            far_y = 0 if want_y >= height // 2 else height - 1
            far_x = 0 if want_x >= width // 2 else width - 1
            logits[index, far_y, far_x] = 5.0
    return logits, truth


def main() -> int:
    print("interpreter:", sys.executable)
    print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
    print("device used:", "cpu")
    print("argv:", sys.argv)
    print()

    frames = 8
    generator = torch.Generator().manual_seed(3)
    scores = torch.randn(frames, vl.N_CANDIDATES, generator=generator)
    teacher = _teacher_batch(frames)
    print(f"scores: shape={tuple(scores.shape)} dtype={scores.dtype}")
    print(f"teacher: shape={tuple(teacher.shape)} dtype={teacher.dtype} "
          f"min={float(teacher.min()):.3f} max={float(teacher.max()):.3f}")
    print()

    recipes = [
        ("legacy-reg", {"rank_weight": 0.0, "regression_weight": 1.0}),
        ("centered-reg", {"rank_weight": 0.0, "regression_weight": 1.0}),
        ("rank R-a (1, 0)", {"rank_weight": 1.0, "regression_weight": 0.0}),
        ("rank R-b (1, 0.1)", {"rank_weight": 1.0, "regression_weight": 0.1}),
        ("rank R-c (1, 1)", {"rank_weight": 1.0, "regression_weight": 1.0}),
    ]
    for label, kwargs in recipes:
        objective = label.split()[0]
        leaf = scores.clone().requires_grad_(True)
        loss, stats = vl.policy_loss(leaf, teacher, objective, **kwargs)
        if not loss.requires_grad:
            raise AssertionError("loss is not differentiable")
        loss.backward()
        assert leaf.grad is not None
        grad_norm = float(leaf.grad.norm())
        value = float(loss.detach())
        assert math.isfinite(value), "loss is not finite"
        for key in ("rank_loss", "regression_loss", "valid_pairs",
                    "frames_with_pairs", "frames_without_pairs"):
            assert math.isfinite(float(stats[key])), f"{key} is not finite"
        print(f"[{label}] loss={value:.6f} grad_norm={grad_norm:.6f} "
              f"grad_nonzero={grad_norm > 0.0}")
        print("  stats:", json.dumps(stats, ensure_ascii=False, sort_keys=True))
    print()

    n = 50
    rng = np.random.default_rng(7)
    pred = rng.normal(0.0, 1.0, size=(n, vl.N_CANDIDATES)).astype(np.float32)
    teacher_px = rng.uniform(-20.0, 250.0, size=(n, vl.N_CANDIDATES)).astype(np.float32)
    bullets = _bullets(n)
    metrics = vm.policy_metrics(pred, teacher_px, bullets)
    print(f"policy_metrics on {n} frames (teacher px is raw, unclipped):")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print()
    for height, width in ((28, 24), (56, 48)):
        logits, truth = _heatmap_case(n, height, width)
        rate = vm.heatmap_hit_rate(logits, truth)
        print(f"heatmap_hit_rate {height}x{width} on {n} frames: {rate:.4f}")
    print()
    print("startup check: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
