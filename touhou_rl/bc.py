"""Behaviour cloning on replay captures.

The expert labels are the four game inputs (direction, shoot, slow/focus, bomb)
that the replay fed to ``g_GuiMessageInputCurrent``.  Each head is trained with
cross-entropy; the bomb head is extremely imbalanced (a human uses it a handful
of times per run), so it gets class weights derived from the dataset counts.

Runs entirely on the observation shards -- no game process is involved, so this
is the fast part of the pipeline (GPU-bound, thousands of samples/s).
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

from . import spec
from .dataset import ReplayDataset, count_frames, stacked_batch
from .models import DanmakuPolicy, NetConfig, save_checkpoint


@dataclass
class BCConfig:
    dataset: Path
    out_dir: Path
    epochs: int = 8
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.1
    device: str = "cuda"
    seed: int = 7
    max_steps_per_epoch: int | None = None
    log_every: int = 100
    bomb_class_weight: float = 4.0
    net: NetConfig = field(default_factory=NetConfig)


@dataclass
class BCEpoch:
    epoch: int
    train_loss: float
    val_loss: float
    acc: dict[str, float]
    seconds: float
    lr: float


def _class_weights(dataset: ReplayDataset, device) -> list[torch.Tensor | None]:
    counts = [np.zeros(dim, dtype=np.int64) for dim in spec.ACTION_DIMS]
    for path in dataset.paths:
        with np.load(path) as data:
            actions = data["action"]
            for i, dim in enumerate(spec.ACTION_DIMS):
                counts[i] += np.bincount(actions[:, i].astype(np.int64), minlength=dim)
    weights: list[torch.Tensor | None] = []
    for i, dim in enumerate(spec.ACTION_DIMS):
        c = counts[i].astype(np.float64)
        w = np.where(c > 0, c.sum() / (dim * np.maximum(c, 1.0)), 0.0)
        if i == 3:  # bomb: rare-but-important, cap the rebalancing
            w = np.clip(w, 1.0, 8.0)
        else:
            w = np.clip(w, 0.5, 8.0)
        weights.append(torch.tensor(w, dtype=torch.float32, device=device))
    return weights


def split_indices(total: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    n_val = max(1, int(total * val_fraction)) if val_fraction > 0 else 0
    val = perm[:n_val]
    train = perm[n_val:]
    return train, val


def _batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator, max_steps: int | None):
    order = rng.permutation(indices)
    steps = 0
    for start in range(0, len(order) - batch_size + 1, batch_size):
        yield order[start : start + batch_size]
        steps += 1
        if max_steps is not None and steps >= max_steps:
            return


def _run_epoch(
    model: DanmakuPolicy,
    dataset: ReplayDataset,
    indices: np.ndarray,
    cfg: BCConfig,
    optim: torch.optim.Optimizer | None,
    weights: list[torch.Tensor | None],
    rng: np.random.Generator,
    device,
    batch_size: int | None = None,
) -> tuple[float, dict[str, np.ndarray], int]:
    training = optim is not None
    model.train(training)
    losses = []
    correct = np.zeros(4)
    total = 0
    predicted = [np.zeros(dim, dtype=np.int64) for dim in spec.ACTION_DIMS]
    # The validation split can be smaller than the training batch size; shrink
    # the batch instead of silently skipping validation.
    batch_size = max(1, min(batch_size or cfg.batch_size, len(indices)))
    for batch_indices in _batches(indices, batch_size, rng, cfg.max_steps_per_epoch):
        batch = stacked_batch(dataset, batch_indices.tolist())
        observation = {
            "self_feat": batch["self_feat"].to(device),
            "bullets": batch["bullets"].to(device),
            "danger": batch["danger"].to(device),
        }
        targets = batch["action"].to(device)
        with torch.set_grad_enabled(training):
            out = model(observation)
            logits = [out["move_logits"], out["shoot_logits"], out["focus_logits"], out["bomb_logits"]]
            loss = 0.0
            for i, head in enumerate(logits):
                loss = loss + torch.nn.functional.cross_entropy(head, targets[:, i], weight=weights[i])
            if training:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
        losses.append(float(loss.detach()))
        preds = torch.stack([torch.argmax(head, dim=-1) for head in logits], dim=-1)
        correct += (preds == targets).float().sum(dim=0).cpu().numpy()
        for i in range(4):
            predicted[i] += np.bincount(preds[:, i].cpu().numpy(), minlength=spec.ACTION_DIMS[i])
        total += targets.shape[0]
    acc = (correct / max(1, total)).tolist()
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return mean_loss, acc, total, predicted


def train_bc(cfg: BCConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    dataset = ReplayDataset(cfg.dataset)
    total = len(dataset)
    if total < cfg.batch_size * 2:
        raise RuntimeError(f"dataset too small for batch {cfg.batch_size}: {total} frames")
    train_idx, val_idx = split_indices(total, cfg.val_fraction, cfg.seed)
    model = DanmakuPolicy(cfg.net).to(device)
    weights = _class_weights(dataset, device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_idx) / cfg.batch_size)
    if cfg.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, cfg.max_steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, cfg.epochs * steps_per_epoch))
    rng = np.random.default_rng(cfg.seed)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": str(cfg.dataset),
        "frames": total,
        "train_frames": int(len(train_idx)),
        "val_frames": int(len(val_idx)),
        "device": str(device),
        "parameters": model.num_parameters(),
        "class_weights": [w.tolist() if w is not None else None for w in weights],
        "epochs": [],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    best_val = float("inf")
    try:
        for epoch in range(1, cfg.epochs + 1):
            started = time.perf_counter()
            train_loss, train_acc, _, _ = _run_epoch(
                model, dataset, train_idx, cfg, optim, weights, rng, device
            )
            with torch.no_grad():
                val_loss, val_acc, _, val_pred = _run_epoch(
                    model, dataset, val_idx, cfg, None, weights, rng, device
                )
            for _ in range(steps_per_epoch):
                scheduler.step()
            elapsed = time.perf_counter() - started
            row = BCEpoch(
                epoch=epoch,
                train_loss=round(train_loss, 5),
                val_loss=round(val_loss, 5),
                acc={
                    "move": round(train_acc[0], 4),
                    "shoot": round(train_acc[1], 4),
                    "focus": round(train_acc[2], 4),
                    "bomb": round(train_acc[3], 4),
                    "val_move": round(val_acc[0], 4),
                    "val_shoot": round(val_acc[1], 4),
                    "val_focus": round(val_acc[2], 4),
                    "val_bomb": round(val_acc[3], 4),
                },
                seconds=round(elapsed, 1),
                lr=round(optim.param_groups[0]["lr"], 7),
            )
            summary["epochs"].append(row.__dict__)
            summary["val_predicted_distribution"] = [p.tolist() for p in val_pred]
            print(json.dumps(row.__dict__, ensure_ascii=False), flush=True)
            save_checkpoint(cfg.out_dir / "last.pt", model, {"epoch": epoch, "val_loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    cfg.out_dir / "best.pt",
                    model,
                    {"epoch": epoch, "val_loss": val_loss, "train_acc": row.acc},
                )
    finally:
        summary["best_val_loss"] = best_val
        summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (cfg.out_dir / "bc-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Behaviour cloning on TH08 replay captures")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--self-hidden", type=int, default=128)
    parser.add_argument("--bullet-hidden", type=int, default=64)
    parser.add_argument("--trunk-hidden", type=int, default=256)
    args = parser.parse_args(argv)
    out = Path(args.out) if args.out else Path(args.dataset) / "bc-run"
    cfg = BCConfig(
        dataset=Path(args.dataset),
        out_dir=out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        max_steps_per_epoch=args.max_steps_per_epoch or None,
        val_fraction=args.val_fraction,
        net=NetConfig(
            self_hidden=args.self_hidden,
            bullet_hidden=args.bullet_hidden,
            trunk_hidden=args.trunk_hidden,
        ),
    )
    summary = train_bc(cfg)
    print(json.dumps({k: v for k, v in summary.items() if k != "epochs"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
