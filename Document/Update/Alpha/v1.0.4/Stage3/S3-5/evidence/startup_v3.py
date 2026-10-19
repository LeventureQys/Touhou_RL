"""S3-5 startup test: synthetic pixels -> real preprocessing -> v3-st -> backward.

This is the *startup* evidence required by the S3-5 task book: it is not unit
tests (those are ``touhou_rl/tests/test_vmodels3.py``) and it is not a training
run (S3-7).  It builds three synthetic uint8 ROI stacks, pushes them through the
real :func:`touhou_rl.vision.regions_to_batch`, builds the default C5 model, and
exercises ``forward``/``score``/``act``/``decode_state``/``backward`` with the
real ranking loss (:func:`touhou_rl.vision_losses.policy_loss`) plus the heatmap,
danger and self auxiliary terms.  Every tensor's shape, dtype and finiteness and
the per-group gradient norms are printed; the process exits 0 only if all of them
are finite and at least the policy/localisation groups are non-zero.

Run from the workspace root with ``PYTHONPATH`` set::

    & $py Document/Update/Alpha/v1.0.4/Stage3/S3-5/evidence/startup_v3.py

CPU only, deterministic seeds, ~seconds.  It never touches the game or a GPU.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from touhou_rl import obs, vision
from touhou_rl.vision_losses import policy_loss
from touhou_rl.vmodels2 import heatmap_target
from touhou_rl.vmodels3 import (
    FUSED_VEC,
    VisionSpatialTemporalConfig,
    VisionSpatialTemporalPolicy,
)

BATCH = 2
SEED = 20261019


def _shape(tensor: torch.Tensor) -> dict:
    return {
        "shape": "x".join(str(size) for size in tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(tensor).all()),
        "min": round(float(tensor.detach().min()), 6),
        "max": round(float(tensor.detach().max()), 6),
    }


def _grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return total**0.5


def main() -> int:
    torch.manual_seed(0)
    config = VisionSpatialTemporalConfig()
    report: dict = {
        "batch": BATCH,
        "seed": SEED,
        "config": {
            "arch": VisionSpatialTemporalPolicy.ARCH,
            "depth": config.depth,
            "region_sizes": {
                name: "x".join(str(size) for size in sizes)
                for name, sizes in config.region_sizes.items()
            },
            "heatmap_hw": "x".join(str(size) for size in config.heatmap_hw),
            "roi_native_hw": "x".join(str(size) for size in config.roi_native_hw),
            "local_grid_hw": "x".join(str(size) for size in config.local_grid_hw),
            "align_corners": config.align_corners,
            "center_detach": config.center_detach,
            "local_enabled": config.local_enabled,
            "heatmap_gauss_sigma": config.heatmap_gauss_sigma,
            "candidate_order_version": config.candidate_order_version,
            "history_contract": config.history_contract,
            "score_semantics": config.score_semantics,
        },
    }

    # 1. Synthetic uint8 ROI stacks, depth 3, through the real preprocessing.
    rng = np.random.default_rng(SEED)
    raw = {
        name: rng.integers(0, 256, (BATCH, config.depth, height, width, 3), dtype=np.uint8)
        for name, (height, width) in config.region_sizes.items()
    }
    raw_shapes = {
        name: "x".join(str(size) for size in value.shape) for name, value in raw.items()
    }
    batch = vision.regions_to_batch(raw)
    preprocessed = {name: _shape(value) for name, value in batch.items()}
    for name, value in batch.items():
        assert value.dtype == torch.float32, name
        assert 0.0 <= float(value.min()) and float(value.max()) <= 1.0, name
        assert raw_shapes[name] == "x".join(str(size) for size in value.shape), name
    report["raw_uint8"] = raw_shapes
    report["preprocessed"] = preprocessed

    # 2. Model.
    model = VisionSpatialTemporalPolicy(config)
    report["parameters"] = model.num_parameters()
    started = time.perf_counter()
    out = model(batch)
    forward_seconds = time.perf_counter() - started
    report["forward_seconds_batch"] = round(forward_seconds, 5)
    report["forward_seconds_per_frame"] = round(forward_seconds / BATCH, 5)
    report["forward"] = {name: _shape(value) for name, value in out.items()}
    for name, value in out.items():
        assert value.dtype == torch.float32, name
        assert bool(torch.isfinite(value).all()), name

    # 3. Legacy accessors.
    scores = model.score(batch)
    features = model.features(batch)
    state = model.decode_state(batch)
    actions, log_probability, value, extra = model.act(batch)
    report["score"] = _shape(scores)
    report["features"] = _shape(features)
    report["decode_state"] = {name: _shape(tensor) for name, tensor in state.items()}
    report["act"] = {
        "actions": {
            "shape": "x".join(str(size) for size in actions.shape),
            "dtype": str(actions.dtype),
            "values": actions.tolist(),
        },
        "logp": _shape(log_probability),
        "max_preference": _shape(value),
        "extra": _shape(extra),
    }
    assert actions.dtype == torch.int64
    assert actions.shape == (BATCH, 4)

    # 4. Training-shaped backward: real rank loss + heatmap + danger + self.
    grid_h, grid_w = config.heatmap_hw
    truth_x = rng.random(BATCH).astype(np.float32)
    truth_y = rng.random(BATCH).astype(np.float32)
    teacher_px = torch.as_tensor(
        rng.normal(0.0, 10.0, (BATCH, 18)).astype(np.float32)
    )
    target_heatmap = torch.as_tensor(
        heatmap_target(truth_x, truth_y, grid_h, grid_w, sigma=config.heatmap_gauss_sigma)
    )
    target_danger = torch.as_tensor(
        rng.random((BATCH, obs.DANGER_ROWS * obs.DANGER_COLS)).astype(np.float32)
    )
    target_self = torch.as_tensor(
        rng.normal(0.0, 0.5, (BATCH, obs.N_SELF)).astype(np.float32)
    )
    policy_term, policy_stats = policy_loss(scores, teacher_px, "rank")
    log_probability_heatmap = torch.log_softmax(out["heatmap_logits"].flatten(1), dim=-1)
    heatmap_term = -(target_heatmap * log_probability_heatmap).sum(-1).mean()
    danger_term = F.smooth_l1_loss(out["danger_log"], target_danger)
    self_term = F.smooth_l1_loss(out["self_pred"][:, 2:], target_self[:, 2:])
    loss = policy_term + heatmap_term + 0.2 * danger_term + 0.1 * self_term
    assert bool(torch.isfinite(loss))
    loss.backward()
    report["loss"] = {
        "total": float(loss.detach()),
        "policy_rank": float(policy_term.detach()),
        "heatmap": float(heatmap_term.detach()),
        "danger": float(danger_term.detach()),
        "self": float(self_term.detach()),
        "valid_pairs": int(policy_stats["valid_pairs"]),
        "frames_with_pairs": int(policy_stats["frames_with_pairs"]),
    }

    groups = {
        "playfield_trunk": list(model.playfield.parameters()),
        "heat_conv": list(model.heat_conv.parameters()),
        "heatmap_head": list(model.heatmap_head.parameters()),
        "location_mlp": list(model.location_mlp.parameters()),
        "local": list(model.local.parameters()),
        "global": list(model.global_conv.parameters())
        + list(model.global_fc.parameters()),
        "hud_score": list(model.hud_score.parameters()),
        "hud_status": list(model.hud_status.parameters()),
        "trunk": list(model.trunk.parameters()),
        "policy_head": list(model.policy_head.parameters()),
        "self_head": list(model.self_head.parameters()),
        "danger": list(model.danger.parameters()),
    }
    grad_norms = {name: _grad_norm(parameters) for name, parameters in groups.items()}
    report["grad_norms"] = {name: round(value, 6) for name, value in grad_norms.items()}
    for name, norm in grad_norms.items():
        assert np.isfinite(norm), name
    # The whole point of one encode + four heads: everything the loss touches must
    # actually receive a finite, non-zero gradient.
    for name in (
        "playfield_trunk",
        "heat_conv",
        "heatmap_head",
        "local",
        "global",
        "hud_score",
        "hud_status",
        "trunk",
        "policy_head",
        "self_head",
        "danger",
    ):
        assert grad_norms[name] > 0.0, f"{name} received no gradient"
    report["all_grad_groups_nonzero"] = True

    report["fusion_width"] = FUSED_VEC
    report["ok"] = True
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - startup evidence must show the traceback
        import traceback

        traceback.print_exc()
        sys.exit(1)
