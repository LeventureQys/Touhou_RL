"""Distil the avoidance teacher's *value* into a network (score regression).

Why value instead of actions
----------------------------
The first distillation attempt cloned the teacher's chosen action and failed:
1,336 frames of survival in the live game, no better than the v1.0.2 baseline,
even though the teacher itself survives 6,502.  Breaking the validation set down
by nearest-bullet distance showed where it broke - 0.21 direction accuracy when a
bullet was within 15 px - and why: with several equally safe moves the teacher's
tie-break is arbitrary, so the same situation carried contradictory labels
(``left``, ``right``, ``idle``) and the classifier learned their average, which
is ``idle``: the one action that is certainly wrong.

This module labels every frame with the teacher's value for each of the 18
(focus, direction) candidates and regresses it.  Ties stay ties, and each frame
provides 18 targets instead of one.

Pipeline
--------
1. Load every shard of a dataset (same format as replay capture);
2. Label it: for each frame, run :meth:`touhou_rl.pilot.AvoidPilot._rollout_scores`
   on the *stored observation* (so the target only uses what the network can
   see - 64 nearest bullets).  Cached next to the dataset as ``score-labels.npz``;
3. Train :class:`touhou_rl.models.ScorePolicy` to regress the value;
4. Report the numbers that matter for behaving well, not just loss:
   * ``top1`` - how often the network's argmax equals the teacher's argmax;
   * ``regret`` - how much teacher value the network's choice gives up (px of
     clearance);
   * both broken down by nearest-bullet distance, because that is exactly where
     the action-cloning baseline collapsed.

Cost: labelling is the expensive part (one beam search per frame, measured
~3-14 ms/frame), training on 46k frames is a GPU minute.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .dataset import ReplayDataset, stacked_batch
from .models import N_CANDIDATES, NetConfig, ScorePolicy, save_checkpoint
from .pilot import AvoidPilot, PilotConfig

ROOT = Path(__file__).resolve().parents[1]
# The teacher's value is a clearance in pixels, clipped to a sane range and
# scaled so the regression target is O(1).
VALUE_CLIP = (-8.0, 40.0)
VALUE_SCALE = 10.0


def _to_target(values: np.ndarray) -> np.ndarray:
    return np.clip(values, *VALUE_CLIP) / VALUE_SCALE


def default_cache_path(dataset: Path) -> Path:
    """Labels live *next to* the dataset, never inside it.

    Inside the directory they would be discovered as a shard by
    :func:`touhou_rl.dataset.discover_shards` (which now also filters by the
    ``action`` key, but keeping the directory pure is the cheaper invariant).
    """
    dataset = Path(dataset)
    return dataset.parent / f"{dataset.name}-score-labels.npz"


def label_dataset(
    dataset: ReplayDataset,
    cache: Path,
    pilot: AvoidPilot,
    *,
    chunk: int = 2048,
    verbose: bool = True,
) -> np.ndarray:
    """Compute the teacher value for every frame of ``dataset``; cache to ``cache``."""
    if cache.exists():
        with np.load(cache) as data:
            values = data["values"]
        if values.shape[0] == len(dataset):
            return values
        if verbose:
            print(f"[distill] cached labels have {values.shape[0]} rows, relabelling", flush=True)
    started = time.perf_counter()
    out = np.zeros((len(dataset), N_CANDIDATES), dtype=np.float16)
    total = len(dataset)
    for start in range(0, total, chunk):
        stop = min(total, start + chunk)
        batch = stacked_batch(dataset, list(range(start, stop)))
        self_feat = batch["self_feat"].numpy()
        bullets = batch["bullets"].numpy()
        danger = batch["danger"].numpy()
        for offset in range(stop - start):
            observation = {
                "self_feat": self_feat[offset],
                "bullets": bullets[offset],
                "danger": danger[offset],
            }
            out[start + offset] = pilot.candidate_scores(observation).astype(np.float16)
        if verbose:
            done = stop
            rate = done / max(1e-6, time.perf_counter() - started)
            print(
                f"[distill] labelled {done}/{total} frames ({rate:.0f} frames/s)",
                flush=True,
            )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, values=out)
    return out


@dataclass
class DistillConfig:
    dataset: Path
    out_dir: Path
    cache: Path | None = None
    epochs: int = 24
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.1
    seed: int = 7
    device: str = "cuda"
    net: NetConfig = field(default_factory=NetConfig)
    relabel: bool = False


def _danger(bullets: np.ndarray) -> np.ndarray:
    """Nearest-bullet distance per frame (px), INF when nothing is close."""
    dx = bullets[:, :, 0] * 192.0
    dy = bullets[:, :, 1] * 224.0
    valid = bullets[:, :, 5] > 0.5
    distance = np.sqrt(dx * dx + dy * dy)
    return np.where(valid, distance, np.inf).min(axis=1)


def _bucketed_report(name: str, values: np.ndarray, top1: np.ndarray, regret: np.ndarray) -> list[dict]:
    rows = []
    for low, high in ((0.0, 15.0), (15.0, 30.0), (30.0, 60.0), (60.0, 120.0), (120.0, np.inf)):
        mask = (values >= low) & (values < high)
        if not mask.any():
            continue
        rows.append(
            {
                "bucket": f"{int(low)}-{int(high)}" if np.isfinite(high) else f"{int(low)}+",
                "frames": int(mask.sum()),
                "top1": round(float(top1[mask].mean()), 4),
                "regret_px": round(float(regret[mask].mean()), 3),
            }
        )
    return rows


def train_distill(cfg: DistillConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dataset = ReplayDataset(cfg.dataset)
    total = len(dataset)
    if total < cfg.batch_size * 2:
        raise RuntimeError(f"dataset too small for batch {cfg.batch_size}: {total} frames")

    # switch_penalty=0 on purpose: the switching penalty is a function of the
    # teacher's *previous* action, which the student cannot observe.  Leaving it
    # on would put teacher history into the regression target.
    pilot = AvoidPilot(PilotConfig(switch_penalty=0.0))
    cache = cfg.cache or default_cache_path(Path(cfg.dataset))
    if cfg.relabel and cache.exists():
        cache.unlink()
    raw_values = label_dataset(dataset, cache, pilot)
    targets = _to_target(raw_values.astype(np.float32))

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(total)
    n_val = max(1, int(total * cfg.val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = ScorePolicy(cfg.net).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_idx) / cfg.batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, cfg.epochs * steps_per_epoch))

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": str(cfg.dataset),
        "frames": total,
        "train_frames": int(len(train_idx)),
        "val_frames": int(len(val_idx)),
        "device": str(device),
        "parameters": model.num_parameters(),
        "value_clip": list(VALUE_CLIP),
        "value_scale": VALUE_SCALE,
        "epochs": [],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    def run_epoch(indices: np.ndarray, training: bool) -> tuple[float, float, float]:
        order = rng.permutation(indices)
        batch_size = max(1, min(cfg.batch_size, len(order)))
        total_loss = 0.0
        hits = 0
        regret = 0.0
        seen = 0
        for start in range(0, len(order) - batch_size + 1, batch_size):
            index = order[start : start + batch_size]
            batch = stacked_batch(dataset, index.tolist())
            observation = {
                "self_feat": batch["self_feat"].to(device),
                "bullets": batch["bullets"].to(device),
                "danger": batch["danger"].to(device),
            }
            target = torch.as_tensor(targets[index], dtype=torch.float32, device=device)
            with torch.set_grad_enabled(training):
                predicted = model.score(observation)
                loss = torch.nn.functional.smooth_l1_loss(predicted, target)
                if training:
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
            total_loss += float(loss.detach()) * len(index)
            # Behaviour metrics are computed on the *real* value scale.
            chosen = predicted.argmax(dim=-1).detach().cpu().numpy()
            raw = raw_values[index].astype(np.float32)
            hits += int((chosen == raw.argmax(axis=1)).sum())
            regret += float((raw.max(axis=1) - raw[np.arange(len(index)), chosen]).sum())
            seen += len(index)
        return total_loss / max(1, seen), hits / max(1, seen), regret / max(1, seen)

    best = float("inf")
    try:
        for epoch in range(1, cfg.epochs + 1):
            started = time.perf_counter()
            train_loss, train_top1, train_regret = run_epoch(train_idx, True)
            with torch.no_grad():
                val_loss, val_top1, val_regret = run_epoch(val_idx, False)
            for _ in range(steps_per_epoch):
                scheduler.step()
            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "train_top1": round(train_top1, 4),
                "val_top1": round(val_top1, 4),
                "train_regret_px": round(train_regret, 3),
                "val_regret_px": round(val_regret, 3),
                "seconds": round(time.perf_counter() - started, 1),
            }
            summary["epochs"].append(row)
            print(json.dumps(row), flush=True)
            save_checkpoint(cfg.out_dir / "last.pt", model, {"epoch": epoch, "val_loss": val_loss})
            if val_loss < best:
                best = val_loss
                save_checkpoint(
                    cfg.out_dir / "best.pt",
                    model,
                    {"epoch": epoch, "val_loss": val_loss, "val_top1": val_top1,
                     "val_regret_px": val_regret},
                )

        # Final behaviour report on the validation split, bucketed by danger.
        model.eval()
        order = np.arange(len(val_idx))
        batch = stacked_batch(dataset, val_idx.tolist())
        with torch.no_grad():
            observation = {
                "self_feat": batch["self_feat"].to(device),
                "bullets": batch["bullets"].to(device),
                "danger": batch["danger"].to(device),
            }
            predicted = model.score(observation).argmax(dim=-1).cpu().numpy()
        raw = raw_values[val_idx].astype(np.float32)
        top1 = (predicted == raw.argmax(axis=1)).astype(np.float32)
        regret = raw.max(axis=1) - raw[np.arange(len(val_idx)), predicted]
        danger = _danger(batch["bullets"].numpy())
        summary["validation_by_danger"] = _bucketed_report("val", danger, top1, regret)
        summary["val_top1"] = round(float(top1.mean()), 4)
        summary["val_regret_px"] = round(float(regret.mean()), 3)
    finally:
        summary["best_val_loss"] = best
        summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (cfg.out_dir / "distill-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Distil the avoidance teacher's value into a network")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--cache", default="")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--relabel", action="store_true")
    parser.add_argument("--self-hidden", type=int, default=192)
    parser.add_argument("--bullet-hidden", type=int, default=96)
    parser.add_argument("--trunk-hidden", type=int, default=384)
    args = parser.parse_args(argv)
    cfg = DistillConfig(
        dataset=Path(args.dataset),
        out_dir=Path(args.out) if args.out else Path(args.dataset) / "distill-run",
        cache=Path(args.cache) if args.cache else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        val_fraction=args.val_fraction,
        relabel=args.relabel,
        net=NetConfig(
            self_hidden=args.self_hidden,
            bullet_hidden=args.bullet_hidden,
            trunk_hidden=args.trunk_hidden,
        ),
    )
    summary = train_distill(cfg)
    print(json.dumps({k: v for k, v in summary.items() if k != "epochs"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
