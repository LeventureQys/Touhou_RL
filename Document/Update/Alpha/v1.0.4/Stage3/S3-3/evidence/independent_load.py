"""Independent process: read a v104 summary, verify the checkpoint hash, load it
strictly, and run ``score``/``act`` on real pixels + a synthetic batch.

This is the S3-3 startup test required by the task book §5: it is a separate
process from training, it trusts nothing but the summary's recorded hash, and it
prints the shapes that deployment reads (``[B,18]`` float32 finite scores and
``int64[B,4]`` actions).

    python Document/Update/Alpha/v1.0.4/Stage3/S3-3/evidence/independent_load.py \
        --run third_party/th08/build/rl/runs/v104-s33-smoke-s7 \
        --out Document/Update/Alpha/v1.0.4/Stage3/S3-3/evidence/independent-load.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from touhou_rl.vision import regions_to_batch
from touhou_rl.vision_data import VisionDataset
from touhou_rl.vmodels import load_vision_checkpoint


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", default="state", choices=["state", "policy"])
    parser.add_argument("--frames", type=int, default=4)
    args = parser.parse_args()
    run = Path(args.run)
    summary = json.loads((run / "vision-summary.json").read_text(encoding="utf-8"))
    stage_block = summary["stages"][args.stage]
    best = stage_block["best"]
    if not best:
        print(f"no best row for stage {args.stage}", file=sys.stderr)
        return 1
    checkpoint = run / best["checkpoint"]
    recorded = best["checkpoint_sha256"]
    actual = sha256_file(checkpoint)
    print(f"summary schema      : {summary['schema']}")
    print(f"contract_version    : {summary.get('contract_version')}")
    print(f"selection_rule      : {best['selection_rule']}")
    print(f"checkpoint          : {checkpoint}")
    print(f"recorded sha256     : {recorded}")
    print(f"actual   sha256     : {actual}")
    print(f"sha256 match        : {recorded == actual}")
    if recorded != actual:
        print("checkpoint digest does not match the summary", file=sys.stderr)
        return 1

    # Strict load: load_vision_checkpoint calls load_state_dict(...) with strict=True.
    model, extra = load_vision_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    depth = int(model.config.depth)
    region_sizes = {name: tuple(size) for name, size in model.config.region_sizes.items()}
    print(f"arch / depth        : {extra.get('arch')} / {depth}")
    print(f"epoch / stage       : {extra.get('epoch')} / {extra.get('stage')}")
    print(f"history_contract    : {extra.get('history_contract')}")
    print(f"region_sizes        : {region_sizes}")

    # Real pixels: the same stack_indices/region_frames/regions_to_batch path the
    # training loop uses, for a handful of validation frames.
    dataset_dir = Path(summary["dataset"])
    with VisionDataset(dataset_dir, history_contract=str(extra.get("history_contract") or "legacy")) as dataset:
        total = len(dataset)
        indices = [int(round(total * 0.95)) + offset for offset in range(args.frames)]
        indices = [min(index, total - 1) for index in indices]
        stack = dataset.stack_indices(indices, depth).reshape(-1).tolist()
        frames = dataset.region_frames(stack, region_sizes)
        real = {
            name: frames[name].reshape(len(indices), depth, *region_sizes[name], 3)
            for name in region_sizes
        }
    batch = {name: torch.as_tensor(value, dtype=torch.uint8) for name, value in real.items()}
    batch = regions_to_batch(batch, "cpu")
    print(f"real pixel batch    : { {name: tuple(value.shape) for name, value in batch.items()} }")

    with torch.no_grad():
        scores = model.score(batch)
        actions, logp, value, extra_zero = model.act(batch)
        state = model.decode_state(batch)
    print(f"scores              : shape={tuple(scores.shape)} dtype={scores.dtype} "
          f"finite={bool(torch.isfinite(scores).all())} min={float(scores.min()):.4f} "
          f"max={float(scores.max()):.4f}")
    print(f"actions             : shape={tuple(actions.shape)} dtype={actions.dtype}")
    print(f"actions rows        : {actions.tolist()}")
    print(f"logp/value/extra    : {tuple(logp.shape)} {tuple(value.shape)} {tuple(extra_zero.shape)}")
    print(f"decode_state        : self_feat={tuple(state['self_feat'].shape)} "
          f"danger_log={tuple(state['danger_log'].shape)} heatmap={tuple(state['heatmap_logits'].shape)}")
    assert tuple(scores.shape) == (len(indices), 18)
    assert scores.dtype == torch.float32 and bool(torch.isfinite(scores).all())
    assert tuple(actions.shape) == (len(indices), 4) and actions.dtype == torch.int64
    assert bool(((actions[:, 0] >= 0) & (actions[:, 0] <= 8)).all())
    assert bool((actions[:, 1] == 1).all()) and bool((actions[:, 3] == 0).all())
    assert bool(((actions[:, 2] == 0) | (actions[:, 2] == 1)).all())
    assert tuple(state["self_feat"].shape) == (len(indices), 24)
    assert tuple(state["danger_log"].shape) == (len(indices), 128)
    assert bool(torch.isfinite(state["self_feat"]).all())
    assert bool(torch.isfinite(state["danger_log"]).all())
    assert bool(torch.isfinite(state["heatmap_logits"]).all())

    # Synthetic pixels: a batch that never touched the archive, to prove the
    # checkpoint is not relying on the training loader's state.
    synthetic = {
        name: torch.zeros(2, depth, height, width, 3, dtype=torch.uint8)
        for name, (height, width) in region_sizes.items()
    }
    synthetic = regions_to_batch({name: value for name, value in synthetic.items()}, "cpu")
    with torch.no_grad():
        synthetic_scores = model.score(synthetic)
    print(f"synthetic scores    : shape={tuple(synthetic_scores.shape)} "
          f"finite={bool(torch.isfinite(synthetic_scores).all())}")

    payload = {
        "run": str(run),
        "summary_schema": summary["schema"],
        "contract_version": summary.get("contract_version"),
        "checkpoint": str(checkpoint),
        "recorded_sha256": recorded,
        "actual_sha256": actual,
        "sha256_match": recorded == actual,
        "arch": extra.get("arch"),
        "depth": depth,
        "epoch": extra.get("epoch"),
        "selection_rule": best["selection_rule"],
        "region_sizes": {name: list(size) for name, size in region_sizes.items()},
        "scores_shape": list(scores.shape),
        "scores_dtype": str(scores.dtype),
        "scores_finite": bool(torch.isfinite(scores).all()),
        "actions_shape": list(actions.shape),
        "actions_dtype": str(actions.dtype),
        "actions": actions.tolist(),
        "decode_state_shapes": {
            "self_feat": list(state["self_feat"].shape),
            "danger_log": list(state["danger_log"].shape),
            "heatmap_logits": list(state["heatmap_logits"].shape),
        },
        "synthetic_finite": bool(torch.isfinite(synthetic_scores).all()),
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("INDEPENDENT LOAD OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
