"""Train the vision student: pixels -> privileged state, then pixels -> values.

Two stages, in this order on purpose
------------------------------------
``state``
    Frames -> the privileged observation (self position, resources, danger map).
    This is a supervised task with free labels, so it answers "does the vision
    path see the game at all" without touching the expensive action signal.  The
    loss is a plain regression; the *useful* metrics are the ones that map back to
    game units (pixels of player position, bullets per danger cell).

``policy``
    Frames -> the teacher's value for each of the 18 candidate actions.  The
    teacher is :mod:`touhou_rl.pilot`, which operates on the *stored privileged
    observation* - i.e. the vision student is distilled from the memory-reading
    teacher, exactly the intended role of the reverse-engineered data.

Label material
--------------
Teacher values are computed here rather than captured: they are a pure function
of the privileged observation, so recomputing them is free and lets the teacher
change without re-capturing video.

Reported metrics are chosen so a human can judge whether the student is usable,
not just whether loss went down: player-position error in pixels, bullet-count
error, danger-map correlation, and (policy stage) teacher top-1 agreement plus
regret in pixels of clearance, bucketed by how close the nearest bullet is -
the same breakdown that exposed the privileged student's failure.
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

from . import obs, spec, vision
from .pilot import AvoidPilot, PilotConfig
from .vision import frames_to_batch
from .vision_data import VisionDataset
from .vmodels import VisionNetConfig, VisionScorer, load_vision_checkpoint, save_vision_checkpoint

ROOT = Path(__file__).resolve().parents[1]
VALUE_CLIP = (-8.0, 40.0)
VALUE_SCALE = 10.0
DANGER_SCALE = 8.0


@dataclass
class VisionTrainConfig:
    dataset: Path
    out_dir: Path
    stage: str = "both"  # state | policy | both
    epochs_state: int = 12
    epochs_policy: int = 12
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.1
    seed: int = 7
    device: str = "cuda"
    depth: int = 3
    width: int = 32
    trunk_hidden: int = 256
    init_from: Path | None = None
    teacher_values: Path | None = None
    max_batches_per_epoch: int = 0
    num_workers: int = 0


def teacher_values_for(dataset: VisionDataset, cache: Path, verbose: bool = True) -> np.ndarray:
    """Candidate values for every frame, computed from stored privileged labels.

    Identical objective to :mod:`touhou_rl.distill`, but the labels come from the
    vision shard's own privileged arrays, so the vision student is taught exactly
    the same target the privileged student was.
    """
    if cache.exists():
        with np.load(cache) as data:
            values = data["values"]
        if values.shape[0] == len(dataset):
            return values
    pilot = AvoidPilot(PilotConfig(switch_penalty=0.0))
    out = np.zeros((len(dataset), 18), dtype=np.float16)
    started = time.perf_counter()
    chunk = 2048
    for start in range(0, len(dataset), chunk):
        stop = min(len(dataset), start + chunk)
        batch = dataset.batch(list(range(start, stop)))
        self_feat = batch["self_feat"].astype(np.float32)
        bullets = batch["bullets"].astype(np.float32)
        danger = batch["danger"]
        for offset in range(stop - start):
            observation = {
                "self_feat": self_feat[offset],
                "bullets": bullets[offset],
                "danger": danger[offset],
            }
            out[start + offset] = pilot.candidate_scores(observation).astype(np.float16)
        if verbose:
            print(f"[vision] teacher-labelled {stop}/{len(dataset)} frames", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, values=out)
    if verbose:
        print(f"[vision] labelling took {time.perf_counter() - started:.1f}s", flush=True)
    return out


def _gather(dataset: VisionDataset, indices: list[int], depth: int) -> tuple[torch.Tensor, dict]:
    stack_ids = dataset.stack_indices(indices, depth).reshape(-1)
    batch = dataset.batch(stack_ids.tolist())
    frames = batch["frame"].reshape(len(indices), depth, vision.VISION_H, vision.VISION_W, vision.VISION_CHANNELS)
    center = dataset.batch(indices)
    labels = {
        "self_feat": center["self_feat"].astype(np.float32),
        "danger": center["danger"].astype(np.float32),
        "action": center["action"].astype(np.int64),
        "bullets": center["bullets"].astype(np.float32),
    }
    return torch.as_tensor(frames, dtype=torch.uint8), labels


def _danger_target(danger: np.ndarray) -> np.ndarray:
    return np.log1p(danger / DANGER_SCALE)


def _state_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict:
    """Turn the regression output back into game units a human can judge."""
    position_px = float(
        np.mean(
            np.abs(predicted[:, :2] - truth[:, :2])
            * np.array([spec.PLAYFIELD_W, spec.PLAYFIELD_H], dtype=np.float32)
        )
    )
    rays = float(np.mean(np.abs(predicted[:, 16:24] - truth[:, 16:24])))
    bullet_count_mae = float(np.mean(np.abs(predicted[:, 13] - truth[:, 13])) * 700.0)
    return {
        "self_mae": round(float(np.mean(np.abs(predicted - truth))), 4),
        "player_pos_px": round(position_px, 2),
        "ray_mae": round(rays, 4),
        "bullet_count_mae": round(bullet_count_mae, 1),
    }


def _danger_metrics(predicted_log: np.ndarray, truth: np.ndarray) -> dict:
    true_log = _danger_target(truth)
    flat_pred = predicted_log.reshape(len(predicted_log), -1)
    flat_true = true_log.reshape(len(true_log), -1)
    correlation = float(np.corrcoef(flat_pred.ravel(), flat_true.ravel())[0, 1]) if flat_pred.size > 1 else 0.0
    # "busy cell" accuracy: does it notice where bullets actually are?
    busy_true = flat_true > 0.2
    busy_pred = flat_pred > 0.2
    hit = float(np.mean(busy_true == busy_pred)) if busy_pred.size else 0.0
    return {
        "danger_mae_log": round(float(np.mean(np.abs(flat_pred - flat_true))), 4),
        "danger_corr": round(correlation, 4),
        "danger_cell_acc": round(hit, 4),
    }


def _nearest_bullet(bullets: np.ndarray) -> np.ndarray:
    dx = bullets[:, :, 0] * 192.0
    dy = bullets[:, :, 1] * 224.0
    valid = bullets[:, :, 5] > 0.5
    distance = np.sqrt(dx * dx + dy * dy)
    return np.where(valid, distance, np.inf).min(axis=1)


def _policy_metrics(predicted: np.ndarray, teacher: np.ndarray, bullets: np.ndarray) -> dict:
    chosen = predicted.argmax(axis=1)
    top1 = (chosen == teacher.argmax(axis=1)).astype(np.float64)
    regret = teacher.max(axis=1) - teacher[np.arange(len(teacher)), chosen]
    danger = _nearest_bullet(bullets)
    buckets = []
    for low, high in ((0.0, 15.0), (15.0, 30.0), (30.0, 60.0), (60.0, 120.0), (120.0, np.inf)):
        mask = (danger >= low) & (danger < high)
        if not mask.any():
            continue
        buckets.append(
            {
                "bucket": f"{int(low)}-{int(high)}" if np.isfinite(high) else f"{int(low)}+",
                "frames": int(mask.sum()),
                "top1": round(float(top1[mask].mean()), 4),
                "regret_px": round(float(regret[mask].mean()), 3),
            }
        )
    return {
        "top1": round(float(top1.mean()), 4),
        "regret_px": round(float(regret.mean()), 3),
        "by_danger": buckets,
    }


def train(cfg: VisionTrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dataset = VisionDataset(cfg.dataset)
    total = len(dataset)
    if total < 4:
        raise RuntimeError(f"dataset too small: {total} frames")
    stats = dataset.label_stats()

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(total)
    n_val = max(1, int(total * cfg.val_fraction))
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()

    net_config = VisionNetConfig(depth=cfg.depth, width=cfg.width, trunk_hidden=cfg.trunk_hidden)
    if cfg.init_from is not None:
        model, _extra = load_vision_checkpoint(cfg.init_from, map_location=device)
        print(f"[vision] warm start from {cfg.init_from}", flush=True)
    else:
        model = VisionScorer(net_config)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": str(cfg.dataset),
        "dataset_stats": stats,
        "frames": total,
        "train_frames": len(train_idx),
        "val_frames": len(val_idx),
        "device": str(device),
        "depth": cfg.depth,
        "parameters": model.num_parameters(),
        "frame_shape": [vision.VISION_H, vision.VISION_W, vision.VISION_CHANNELS],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stages": {},
    }

    def batches(indices: list[int], shuffle: bool, max_batches: int = 0):
        order = list(indices)
        if shuffle:
            rng.shuffle(order)
        count = 0
        for start in range(0, len(order) - cfg.batch_size + 1, cfg.batch_size):
            yield order[start : start + cfg.batch_size]
            count += 1
            if max_batches and count >= max_batches:
                return

    def run_state_epoch(training: bool, indices: list[int]) -> dict:
        model.train(training)
        total_loss = 0.0
        seen = 0
        preds: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        pred_danger: list[np.ndarray] = []
        true_danger: list[np.ndarray] = []
        for index in batches(indices, training, cfg.max_batches_per_epoch):
            frames, labels = _gather(dataset, index, cfg.depth)
            frames = frames_to_batch(frames.numpy(), device)
            with torch.set_grad_enabled(training):
                out = model.decode_state(frames)
                target_self = torch.as_tensor(labels["self_feat"], device=device)
                target_danger = torch.as_tensor(_danger_target(labels["danger"]), device=device)
                loss = torch.nn.functional.smooth_l1_loss(out["self_feat"], target_self) + torch.nn.functional.smooth_l1_loss(
                    out["danger_log"], target_danger.reshape(len(index), -1)
                )
                if training:
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
            total_loss += float(loss.detach()) * len(index)
            seen += len(index)
            preds.append(out["self_feat"].detach().cpu().numpy())
            truths.append(labels["self_feat"])
            pred_danger.append(out["danger_log"].detach().cpu().numpy().reshape(len(index), obs.DANGER_ROWS, obs.DANGER_COLS))
            true_danger.append(labels["danger"])
        if not preds:
            return {}
        result = {"loss": round(total_loss / max(1, seen), 5), "frames": seen}
        if not training:
            result.update(_state_metrics(np.concatenate(preds), np.concatenate(truths)))
            result.update(_danger_metrics(np.concatenate(pred_danger), np.concatenate(true_danger)))
        return result

    def run_policy_epoch(training: bool, indices: list[int], values: np.ndarray) -> dict:
        model.train(training)
        total_loss = 0.0
        seen = 0
        preds: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        bullets: list[np.ndarray] = []
        for index in batches(indices, training, cfg.max_batches_per_epoch):
            frames, labels = _gather(dataset, index, cfg.depth)
            frames = frames_to_batch(frames.numpy(), device)
            target = np.clip(values[index].astype(np.float32), *VALUE_CLIP) / VALUE_SCALE
            with torch.set_grad_enabled(training):
                predicted = model.score(frames)
                loss = torch.nn.functional.smooth_l1_loss(
                    predicted, torch.as_tensor(target, device=device)
                )
                if training:
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
            total_loss += float(loss.detach()) * len(index)
            seen += len(index)
            preds.append(predicted.detach().cpu().numpy())
            truths.append(values[index].astype(np.float32))
            bullets.append(labels["bullets"])
        if not preds:
            return {}
        result = {"loss": round(total_loss / max(1, seen), 5), "frames": seen}
        if not training:
            result.update(_policy_metrics(np.concatenate(preds), np.concatenate(truths), np.concatenate(bullets)))
        return result

    try:
        if cfg.stage in ("state", "both"):
            best = float("inf")
            history = []
            for epoch in range(1, cfg.epochs_state + 1):
                started = time.perf_counter()
                train_row = run_state_epoch(True, train_idx)
                with torch.no_grad():
                    val_row = run_state_epoch(False, val_idx)
                row = {
                    "epoch": epoch,
                    "train_loss": train_row.get("loss"),
                    "val_loss": val_row.get("loss"),
                    "seconds": round(time.perf_counter() - started, 1),
                    **{f"val_{key}": value for key, value in val_row.items() if key not in ("loss", "frames")},
                }
                history.append(row)
                print(f"[vision:state] {json.dumps(row)}", flush=True)
                if val_row.get("loss", float("inf")) < best:
                    best = float(val_row.get("loss", 0.0))
                    save_vision_checkpoint(
                        cfg.out_dir / "state.pt", model, {"stage": "state", "epoch": epoch, **val_row}
                    )
                    save_vision_checkpoint(cfg.out_dir / "best.pt", model, {"stage": "state", "epoch": epoch})
            summary["stages"]["state"] = {"history": history, "best_val_loss": best}

        if cfg.stage in ("policy", "both"):
            if not dataset.has_value:
                cache = cfg.teacher_values or (Path(cfg.dataset).parent / f"{Path(cfg.dataset).name}-teacher-values.npz")
                values = teacher_values_for(dataset, cache)
            else:
                values = np.zeros((total, 18), dtype=np.float16)
                for start in range(0, total, 4096):
                    stop = min(total, start + 4096)
                    values[start:stop] = dataset.batch(list(range(start, stop)))["value"]
            best = float("inf")
            history = []
            for epoch in range(1, cfg.epochs_policy + 1):
                started = time.perf_counter()
                train_row = run_policy_epoch(True, train_idx, values)
                with torch.no_grad():
                    val_row = run_policy_epoch(False, val_idx, values)
                row = {
                    "epoch": epoch,
                    "train_loss": train_row.get("loss"),
                    "val_loss": val_row.get("loss"),
                    "val_top1": val_row.get("top1"),
                    "val_regret_px": val_row.get("regret_px"),
                    "seconds": round(time.perf_counter() - started, 1),
                }
                history.append(row)
                print(f"[vision:policy] {json.dumps(row)}", flush=True)
                # Model selection uses **regret**, not loss.  Measured on this
                # dataset: loss kept falling while top-1 agreement and regret got
                # worse after epoch 4, so selecting by loss picks a strictly
                # worse policy (it is more confident about the wrong action).
                score = val_row.get("regret_px", float("inf"))
                if score < best:
                    best = float(score)
                    save_vision_checkpoint(
                        cfg.out_dir / "policy.pt",
                        model,
                        {"stage": "policy", "epoch": epoch, "val_top1": val_row.get("top1")},
                    )
                    save_vision_checkpoint(cfg.out_dir / "best.pt", model, {"stage": "policy", "epoch": epoch})
            summary["stages"]["policy"] = {
                "history": history,
                "best_val_regret_px": best,
                "final": val_row,
            }
    finally:
        summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (cfg.out_dir / "vision-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the vision student")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--stage", default="both", choices=["state", "policy", "both"])
    parser.add_argument("--epochs-state", type=int, default=12)
    parser.add_argument("--epochs-policy", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--trunk-hidden", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-from", default="")
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    args = parser.parse_args(argv)
    cfg = VisionTrainConfig(
        dataset=Path(args.dataset),
        out_dir=Path(args.out) if args.out else Path(args.dataset) / "vision-run",
        stage=args.stage,
        epochs_state=args.epochs_state,
        epochs_policy=args.epochs_policy,
        batch_size=args.batch_size,
        lr=args.lr,
        depth=args.depth,
        width=args.width,
        trunk_hidden=args.trunk_hidden,
        device=args.device,
        init_from=Path(args.init_from) if args.init_from else None,
        max_batches_per_epoch=args.max_batches_per_epoch,
    )
    summary = train(cfg)
    print(json.dumps({k: v for k, v in summary.items() if k != "stages"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
