"""Compare distillation students against the teacher's own value, by danger level.

Why this exists
---------------
"Validation accuracy" is not a comparable number between an action-cloning
network and a value-regressing network: the cloner is scored on matching an
arbitrary tie-break, the value model on giving up as little safety as possible.

This tool scores **both** on the same objective - the teacher's value
(`score-labels.npz`, produced by `python -m touhou_rl.distill`) - and reports two
metrics per danger bucket:

* ``top1``   - share of frames where the student picks the teacher's argmax;
* ``regret`` - teacher value of the teacher's best action minus the teacher value
  of the action the student picked, in pixels of clearance.

Regret is the metric that matters: 0.2 px is a tie, 2 px in a dense pattern is
the difference between dodging and dying.  Bucketing by nearest-bullet distance
is deliberate - that is where the first distillation attempt collapsed
(0.21 direction accuracy inside 15 px).

Usage::

    python tools/rl/compare_students.py \
        --dataset third_party/th08/build/rl/datasets/dagger-mix \
        --student clone=third_party/th08/build/rl/runs/bc-dagger/best.pt \
        --student value=third_party/th08/build/rl/runs/scorer-v1/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from touhou_rl.dataset import ReplayDataset, stacked_batch
from touhou_rl.distill import default_cache_path
from touhou_rl.models import load_checkpoint

BUCKETS = ((0.0, 15.0), (15.0, 30.0), (30.0, 60.0), (60.0, 120.0), (120.0, np.inf))


def _flat_action(actions: np.ndarray) -> np.ndarray:
    """(move, shoot, focus, bomb) -> candidate index ``focus*9 + move``."""
    return actions[:, 2].astype(np.int64) * 9 + actions[:, 0].astype(np.int64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare distillation students by regret")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--labels", default="", help="score-labels.npz (default: <dataset>/score-labels.npz)")
    parser.add_argument("--student", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    if not args.student:
        parser.error("at least one --student LABEL=PATH is required")

    dataset = ReplayDataset(Path(args.dataset))
    labels_path = Path(args.labels) if args.labels else default_cache_path(Path(args.dataset))
    teacher = np.load(labels_path)["values"].astype(np.float32)
    if teacher.shape[0] != len(dataset):
        raise SystemExit(f"labels have {teacher.shape[0]} rows, dataset has {len(dataset)} frames")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(dataset))
    val = perm[: max(1, int(len(dataset) * args.val_fraction))]
    batch = stacked_batch(dataset, val.tolist())
    bullets = batch["bullets"].numpy()
    dx = bullets[:, :, 0] * 192.0
    dy = bullets[:, :, 1] * 224.0
    valid = bullets[:, :, 5] > 0.5
    danger = np.where(valid, np.sqrt(dx * dx + dy * dy), np.inf).min(axis=1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    reference = teacher[val]
    best_value = reference.max(axis=1)
    rows: list[dict] = []
    for spec in args.student:
        if "=" not in spec:
            parser.error(f"--student expects LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        model, _extra = load_checkpoint(Path(path), map_location=device)
        model.to(device).eval()
        observation = {key: batch[key].to(device) for key in ("self_feat", "bullets", "danger")}
        with torch.no_grad():
            actions, *_ = model.act(observation, deterministic=True)
        chosen = _flat_action(actions.cpu().numpy())
        chosen = np.clip(chosen, 0, reference.shape[1] - 1)
        regret = best_value - reference[np.arange(len(val)), chosen]
        top1 = (chosen == reference.argmax(axis=1)).astype(np.float64)
        row = {
            "label": label,
            "path": str(path),
            "frames": int(len(val)),
            "top1": round(float(top1.mean()), 4),
            "regret_px": round(float(regret.mean()), 3),
            "by_danger": [
                {
                    "bucket": f"{int(low)}-{int(high)}" if np.isfinite(high) else f"{int(low)}+",
                    "frames": int(mask.sum()),
                    "top1": round(float(top1[mask].mean()), 4),
                    "regret_px": round(float(regret[mask].mean()), 3),
                }
                for low, high in BUCKETS
                for mask in [(danger >= low) & (danger < high)]
                if mask.any()
            ],
        }
        rows.append(row)

    header = f"| danger (px) | frames | " + " | ".join(
        f"{row['label']} top1 / regret" for row in rows
    ) + " |"
    print(header)
    print("| --- | --- | " + " | ".join("--- |" * 1 for _ in rows))
    buckets = [entry["bucket"] for entry in rows[0]["by_danger"]]
    for index, bucket in enumerate(buckets):
        frames = rows[0]["by_danger"][index]["frames"]
        cells = [
            f"{row['by_danger'][index]['top1']:.3f} / {row['by_danger'][index]['regret_px']:.2f}"
            for row in rows
        ]
        print(f"| {bucket} | {frames} | " + " | ".join(cells) + " |")
    cells = [f"{row['top1']:.3f} / {row['regret_px']:.2f}" for row in rows]
    print(f"| all | {len(val)} | " + " | ".join(cells) + " |")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
