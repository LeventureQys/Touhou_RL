"""v3 checkpoint round-trip probe (G-v3 approved, Q6) - evidence for S3-6.

Saves a real ``VisionSpatialTemporalPolicy`` through ``vmodels.save_vision_checkpoint``,
reloads it with the strict loader and prints the facts a reviewer needs:

* the explicit ``vision-checkpoint/3`` payload and the five C6 identity keys;
* ``vision_arch()`` returning ``v3-st`` from the loaded type;
* strict ``state_dict`` equality (no ``strict=False`` anywhere);
* ``score`` bit-identical before/after, and a working four-tuple ``act``;
* the ``vision_report_node`` contract record.

This is a synthetic CPU probe, not training and not a real-game run.

Usage::

    python v3_roundtrip_probe.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from touhou_rl import vision
from touhou_rl.vmodels import (
    V3_CHECKPOINT_SCHEMA,
    V3_REQUIRED_EXTRA,
    load_vision_checkpoint,
    save_vision_checkpoint,
    validate_v3_payload,
    vision_arch,
    vision_contract_node,
)
from touhou_rl.vmodels3 import VisionSpatialTemporalConfig, VisionSpatialTemporalPolicy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v3 checkpoint round-trip probe (S3-6)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = VisionSpatialTemporalConfig()
    model = VisionSpatialTemporalPolicy(config).eval()
    print("config fields:", sorted(config.__dict__))
    print("params:", model.num_parameters())
    print("region_sizes:", config.region_sizes, "heatmap_hw:", config.heatmap_hw)

    batch = vision.regions_to_batch(
        {
            name: np.zeros((1, config.depth, height, width, 3), dtype=np.uint8)
            for name, (height, width) in config.region_sizes.items()
        }
    )
    with torch.no_grad():
        forward = model.forward(batch)
    print("forward keys:", sorted(forward))
    print("shapes:", {key: tuple(value.shape) for key, value in forward.items()})

    extra = {
        "source_state_sha": "d" * 64,
        "teacher_fingerprint": "a" * 64,
        "manifest_sha": "b" * 64,
        "val_index_sha256": "c" * 64,
        "loss_config": {"objective": "rank", "rank_delta_px": 0.5},
        "stage": "policy",
        "epoch": 1,
        "depth": 3,
        "history_contract": "causal-segment-v1",
    }
    path = out / "v3-probe.pt"
    save_vision_checkpoint(path, model, extra)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    print("payload keys:", sorted(payload))
    print("schema/kind/arch:", payload["schema"], payload["kind"], payload["arch"])
    print("schema is v3:", payload["schema"] == V3_CHECKPOINT_SCHEMA)
    print("frame:", payload["frame"])
    print("extra has all identity keys:", all(key in payload["extra"] for key in V3_REQUIRED_EXTRA))
    contract = validate_v3_payload(payload)
    print(
        "contract:",
        {key: contract[key] for key in ("schema", "arch", "depth", "history_contract")},
    )

    restored, restored_extra = load_vision_checkpoint(path, map_location="cpu")
    print("loaded type:", type(restored).__name__, "arch:", vision_arch(restored))
    print("extra round-trips:", restored_extra == extra)
    weights_equal = all(
        torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items()
    )
    print("strict weights equal:", weights_equal)
    with torch.no_grad():
        before = model.score(batch)
        after = restored.score(batch)
        actions, logp, value, placeholder = restored.act(batch)
    print("score bit-identical:", bool(torch.equal(before, after)))
    print(
        "act tuple:",
        tuple(actions.shape),
        actions.dtype,
        tuple(logp.shape),
        tuple(value.shape),
        tuple(placeholder.shape),
    )
    print("act action:", actions[0].tolist())
    node = vision_contract_node(restored, restored_extra)
    print("report contract keys:", sorted(node))
    print(
        "report contract:",
        json.dumps(
            {
                "schema": node["schema"],
                "arch": node["arch"],
                "candidate_order_version": node["candidate_order_version"],
                "score_semantics": node["score_semantics"],
                "roi_native_hw": list(node["roi_native_hw"]),
                "local_grid_hw": list(node["local_grid_hw"]),
                "align_corners": node["align_corners"],
            },
            ensure_ascii=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
