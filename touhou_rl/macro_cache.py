"""Build positive route-witness caches from paired pixels and full snapshots.

Run ``python -m touhou_rl.macro_cache --paired snapshots.npz --baseline vision.pt
--out witnesses.npz``. No game access is performed. Required NPZ fields:
playfield/hud_score/hud_status float32 [N,depth,H,W,3], already normalised to
[0,1]; player_xy floating [N,2]; previous_candidate int64 [N] (focus*9+move);
positions/velocities/halves floating [M,2]; offsets int64 [N+1];
coverage_complete/model_known bool [N]; groups nonempty unicode [N]; scalar
unicode JSON metadata declaring input_kind real_pixels or synthetic_pixels.
Optional target_xy floating [N,2] and target_deadline_frames int64 [N] declare
a goal and per-row arrival deadline. Geometry
accepts float32 or float64. Offsets delimit ALL current hazards, not a top-64
observation. The caller must explicitly attest model_known, including whether
unobserved future emissions invalidate this snapshot's linear model.

Witnesses are conditional on that snapshot linear model, NOT full-future ground
truth or negative safety labels. Beam failure and incomplete/unknown rows remain
unsupervised. Existing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from .escape.forecast import DangerField, ForecastConfig
from .escape.geometry import MoveModel
from .escape.viability import ACTIONS, ViabilityConfig, search_viability
from .macro_train import SCHEMA, file_sha
from .vmodels import load_vision_checkpoint

REGIONS = ("playfield", "hud_score", "hud_status")


def _array(data, name, shape, dtype):
    value = data[name]
    if value.shape != shape or value.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must be {np.dtype(dtype)} {shape}")
    return value


def _geometry(data, name, shape):
    value = data[name]
    if value.shape != shape or value.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(f"{name} must be float32/float64 {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def load_paired(path):
    """Read and validate the complete snapshot contract without pickle."""
    required = set(REGIONS) | {"player_xy", "previous_candidate", "positions", "velocities",
                              "halves", "offsets", "coverage_complete", "model_known", "groups", "metadata"}
    with np.load(path, allow_pickle=False) as archive:
        if not required.issubset(archive.files):
            raise ValueError(f"missing paired fields: {sorted(required - set(archive.files))}")
        names = required | ({"target_xy", "target_deadline_frames"} & set(archive.files))
        data = {name: archive[name].copy() for name in names}
    metadata = data["metadata"]
    if metadata.shape != () or metadata.dtype.kind != "U":
        raise ValueError("metadata must be scalar unicode JSON")
    try:
        meta = json.loads(str(metadata.item()))
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid paired metadata JSON") from exc
    if not isinstance(meta, dict) or meta.get("input_kind") not in ("real_pixels", "synthetic_pixels"):
        raise ValueError("paired metadata requires explicit input_kind real_pixels/synthetic_pixels")
    player = data["player_xy"]
    if player.ndim != 2 or player.shape[1:] != (2,) or not len(player):
        raise ValueError("player_xy must be nonempty [N,2]")
    n = len(player)
    _geometry(data, "player_xy", (n, 2))
    model = MoveModel()
    if np.any(player[:, 0] < model.x_min) or np.any(player[:, 0] > model.x_max) or np.any(player[:, 1] < model.y_min) or np.any(player[:, 1] > model.y_max):
        raise ValueError("player_xy outside movement bounds")
    depth = None
    for name in REGIONS:
        value = data[name]
        if value.dtype != np.float32 or value.ndim != 5 or value.shape[0] != n or value.shape[-1] != 3 or any(d < 1 for d in value.shape[1:]):
            raise ValueError(f"{name} must be float32 [N,depth,H,W,3]")
        if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
            raise ValueError(f"{name} must be finite and already normalised to [0,1]")
        if depth is not None and depth != value.shape[1]:
            raise ValueError("pixel regions must share depth")
        depth = value.shape[1]
    previous = _array(data, "previous_candidate", (n,), np.int64)
    if np.any((previous < 0) | (previous >= 18)):
        raise ValueError("previous_candidate must be in [0,17]")
    positions = data["positions"]
    if positions.ndim != 2 or positions.shape[1:] != (2,):
        raise ValueError("positions must be [M,2]")
    m = len(positions)
    for name in ("positions", "velocities", "halves"):
        _geometry(data, name, (m, 2))
    if np.any(data["halves"] < 0):
        raise ValueError("halves must be nonnegative")
    offsets = _array(data, "offsets", (n + 1,), np.int64)
    if offsets[0] != 0 or offsets[-1] != m or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("offsets must be monotonic from 0 to M")
    for name in ("coverage_complete", "model_known"):
        _array(data, name, (n,), np.bool_)
    groups = data["groups"]
    if groups.shape != (n,) or groups.dtype.kind != "U" or any(not str(v).strip() for v in groups):
        raise ValueError("groups must be nonempty unicode [N]")
    if "target_deadline_frames" in data and "target_xy" not in data:
        raise ValueError("target_deadline_frames requires target_xy")
    if "target_xy" in data:
        target = _geometry(data, "target_xy", (n, 2))
        if any(model.clamp(*xy) != tuple(xy) for xy in target):
            raise ValueError("target_xy outside movement bounds")
    if "target_deadline_frames" in data:
        deadline = _array(data, "target_deadline_frames", (n,), np.int64)
        if np.any(deadline < 1):
            raise ValueError("target_deadline_frames must be positive")
    return data


def build_cache(paired, baseline, out, *, config=None, batch_size=32, device="cpu"):
    """Extract baseline pixel features and search conditional positive witnesses."""
    out = Path(out)
    if out.exists():
        raise FileExistsError("use a new output file; never overwrite a cache")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    config = config or ViabilityConfig()
    data = load_paired(paired)
    snapshot_sha = file_sha(paired)
    baseline_sha = file_sha(baseline)
    backbone, _ = load_vision_checkpoint(baseline, map_location=device)
    backbone = backbone.to(device).eval()
    n = len(data["player_xy"])
    paired_meta = json.loads(str(data["metadata"].item()))
    if paired_meta.get("normalization", "rgb255-v1") != "rgb255-v1":
        raise ValueError("paired normalization must be rgb255-v1")
    for name in REGIONS:
        expected = (backbone.config.depth, *backbone.config.region_sizes[name], 3)
        if "depth" in paired_meta and paired_meta["depth"] != backbone.config.depth:
            raise ValueError("paired metadata depth differs from baseline")
        if "region_sizes" in paired_meta and paired_meta["region_sizes"].get(name) != list(backbone.config.region_sizes[name]):
            raise ValueError("paired metadata region_sizes differs from baseline")
        if data[name].shape[1:] != expected:
            raise ValueError(f"{name} shape differs from baseline contract {expected}")
    features, logits = [], []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            regions = {name: torch.from_numpy(data[name][start:stop]).to(device) for name in REGIONS}
            core = backbone.forward_core(regions)
            for key, width, target in (("features", backbone.config.trunk_hidden, features),
                                       ("policy_logits", 18, logits)):
                value = core[key]
                if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or tuple(value.shape) != (stop - start, width) or not bool(torch.isfinite(value).all()):
                    raise ValueError(f"invalid baseline {key}: expected finite float32 [B,{width}]")
                target.append(value.detach().cpu().numpy().copy())
    features, logits = np.concatenate(features), np.concatenate(logits)
    witness = np.zeros((n, 18), dtype=np.bool_)
    valid = np.zeros(n, dtype=np.bool_)
    route_target = np.zeros((n, 2), dtype=np.float32)
    model = MoveModel()
    scale = np.array([model.x_max - model.x_min, model.y_max - model.y_min])
    statuses = []
    reason_counts = {}
    for i in range(n):
        lo, hi = data["offsets"][i:i+2]
        count = int(hi - lo)
        # Direct construction retains every hazard, with no compressed/top-k view.
        field = DangerField(frame_base=0, positions=data["positions"][lo:hi],
                            velocities=data["velocities"][lo:hi], halves=data["halves"][lo:hi],
                            spawn=np.zeros(count, dtype=np.int64),
                            despawn=np.full(count, config.horizon_frames + 1, dtype=np.int64),
                            config=ForecastConfig(horizon_frames=config.horizon_frames,
                                                  calibrated_frames=90, max_entities=max(1, count)))
        complete, known = bool(data["coverage_complete"][i]), bool(data["model_known"][i])
        row_config = config
        if "target_deadline_frames" in data:
            deadline = int(data["target_deadline_frames"][i])
            if deadline > config.horizon_frames:
                raise ValueError("target deadline exceeds horizon")
            row_config = replace(config, target_deadline_frames=deadline)
        target = tuple(map(float, data["target_xy"][i])) if "target_xy" in data else None
        result = search_viability(field, tuple(map(float, data["player_xy"][i])),
                                  ACTIONS[int(data["previous_candidate"][i])], logits[i],
                                  config=row_config, model=model, coverage_complete=complete, model_known=known,
                                  target=target)
        witness[i] = result.witness_mask
        statuses.append(result.status)
        if result.status != "feasible":
            for reason in result.reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        valid[i] = result.status == "feasible" and complete and known and witness[i].any() and result.route is not None
        if valid[i]:
            endpoint = target if target is not None else result.route.positions[-1]
            route_target[i] = (np.asarray(endpoint) - data["player_xy"][i]) / scale
    paired_meta = json.loads(str(data["metadata"].item()))
    metadata = {"schema": SCHEMA, "source_sha256": baseline_sha,
                "input_kind": paired_meta["input_kind"], "label_semantics": "positive_verified_route_witnesses",
                "route_conditioning": "goal-conditioned" if "target_xy" in data and ("target_deadline_frames" in data or config.target_deadline_frames is not None) else "survival-only",
                "early_escape_labels": "goal deadline conditioned" if "target_deadline_frames" in data or config.target_deadline_frames is not None else "not asserted; survival witnesses are not early escape labels",
                "snapshot_sha256": snapshot_sha, "paired_sha256": snapshot_sha,
                "forecast_semantics": "snapshot linear model conditional; not full-future ground truth",
                "model_known_semantics": "explicit caller attestation, including unknown future emissions",
                "hazard_coverage": "all ragged current hazards; no top-k truncation",
                "calibrated_frames": 90, "viability_config": asdict(config),
                "teacher_config": asdict(config),
                "pixel_contract": {"depth": backbone.config.depth,
                                   "region_sizes": {k: list(backbone.config.region_sizes[k]) for k in REGIONS},
                                   "normalization": "rgb255-v1", "range": [0, 1]},
                "target_deadline_frames": data["target_deadline_frames"].tolist() if "target_deadline_frames" in data else None,
                "target_count": n if "target_xy" in data else 0,
                "coverage_complete_count": int(data["coverage_complete"].sum()),
                "model_known_count": int(data["model_known"].sum()),
                "valid_count": int(valid.sum()), "unknown_reason_counts": reason_counts,
                "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))}}
    if not np.isfinite(route_target).all():
        raise ValueError("nonfinite route_target")
    # Exclusive creation also handles a concurrent creator without overwriting it.
    with out.open("xb") as stream:
        np.savez_compressed(stream, features=features, base_logits=logits, witness_mask=witness,
                            route_target=route_target, valid=valid, route_valid=valid.copy(),
                            groups=data["groups"], metadata=np.array(json.dumps(metadata)))
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--frameskip", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--max-frame-checks", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    config = ViabilityConfig(horizon_frames=args.horizon, frameskip=args.frameskip,
                             beam_width=args.beam_width, max_frame_checks=args.max_frame_checks)
    meta = build_cache(args.paired, args.baseline, args.out, config=config,
                       batch_size=args.batch_size, device=args.device)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
