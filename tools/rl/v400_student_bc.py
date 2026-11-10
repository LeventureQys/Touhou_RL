"""V0 behaviour-cloning smoke for the obs-v1 pixel student (P3 S9 slice).

Trains a small CNN on the **P2 collection policies' own button masks**
(``random`` / ``still`` / ``dive``) for at most ``--max-seconds``.  This is a
contract smoke test for the observation/dataset/checkpoint path, **not** P3
progress: the labels are not ``TeacherLabel`` scores and the resulting
``checkpoint-v0`` may never be reported as a P3 or G3 result.  Every artefact
carries :data:`touhou_rl.v400.student.dataset.V0_DISCLOSURE`.

Run inside WSL (the dataset lives there)::

    /root/v400-venv/bin/python tools/rl/v400_student_bc.py \
        --index /root/v400-data/p2det/out-main/index.json \
        --split /root/v400-data/p2det/split.json \
        --checkpoint .tmp_v400_p3/checkpoint-v0.pt \
        --out .tmp_v400_p3/bc-metrics.json --max-seconds 240
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from touhou_rl.v400.student import obs_spec  # noqa: E402
from touhou_rl.v400.student.dataset import (  # noqa: E402
    V0_DISCLOSURE,
    BehaviorCloneDataset,
    build_dataset,
)
from touhou_rl.v400.student.model import (  # noqa: E402
    StudentNet,
    checkpoint_digest,
    save_checkpoint,
)

ACTION_COUNT = obs_spec.N_MOVE_CLASSES


def _torch():
    import torch

    return torch


def batch_from_samples(samples: list[dict], device: str):
    """Stack dataset items into normalised tensors."""

    torch = _torch()
    play = np.stack([sample["play"] for sample in samples]).astype(np.uint8)
    hud = np.stack([sample["hud"] for sample in samples]).astype(np.uint8)[:, None]
    move = np.asarray([sample["move"] for sample in samples], dtype=np.int64)
    bomb = np.asarray([sample["bomb"] for sample in samples], dtype=np.int64)
    play_t = torch.from_numpy(play).to(device).float().div_(255.0)
    hud_t = torch.from_numpy(hud).to(device).float().div_(255.0)
    return play_t, hud_t, torch.from_numpy(move).to(device), torch.from_numpy(bomb).to(device)


def evaluate(model, dataset: BehaviorCloneDataset, indices, device: str, batch_size: int) -> dict:
    """Top-1/top-3 move accuracy, bomb accuracy and per-source breakdown."""

    torch = _torch()
    model.eval()
    correct1 = correct3 = bomb_correct = 0
    total = 0
    loss_sum = 0.0
    per_source: dict[str, list[int]] = {}
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            samples = [dataset[int(index)] for index in chunk]
            play, hud, move, bomb = batch_from_samples(samples, device)
            move_logits, bomb_logits = model(play, hud)
            loss = (torch.nn.functional.cross_entropy(move_logits, move)
                    + torch.nn.functional.cross_entropy(bomb_logits, bomb))
            loss_sum += float(loss.item()) * len(chunk)
            top3 = move_logits.topk(3, dim=1).indices
            hit1 = top3[:, 0].eq(move)
            hit3 = top3.eq(move[:, None]).any(dim=1)
            hit_bomb = bomb_logits.argmax(dim=1).eq(bomb)
            correct1 += int(hit1.sum())
            correct3 += int(hit3.sum())
            bomb_correct += int(hit_bomb.sum())
            total += len(chunk)
            for sample, a, b, c in zip(samples, hit1.tolist(), hit3.tolist(), hit_bomb.tolist()):
                entry = per_source.setdefault(sample["source"], [0, 0, 0, 0])
                entry[0] += 1
                entry[1] += int(a)
                entry[2] += int(b)
                entry[3] += int(c)
    report = {
        "samples": total,
        "move_top1": correct1 / total if total else 0.0,
        "move_top3": correct3 / total if total else 0.0,
        "bomb_accuracy": bomb_correct / total if total else 0.0,
        "loss": loss_sum / total if total else 0.0,
        "by_source": {
            source: {
                "samples": values[0],
                "move_top1": values[1] / values[0] if values[0] else 0.0,
                "move_top3": values[2] / values[0] if values[0] else 0.0,
                "bomb_accuracy": values[3] / values[0] if values[0] else 0.0,
            }
            for source, values in sorted(per_source.items())
        },
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint", default=".tmp_v400_p3/checkpoint-v0.pt")
    parser.add_argument("--out", default=".tmp_v400_p3/bc-metrics.json")
    parser.add_argument("--max-seconds", type=float, default=240.0,
                        help="wall-clock budget for the whole training phase")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args(argv)

    torch = _torch()
    started = time.monotonic()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds = build_dataset(args.index, args.split, split="train")
    val_ds = build_dataset(args.index, args.split, split="val")
    train_report = train_ds.report()
    val_report = val_ds.report()
    print(f"train: {train_report['windows']} windows / {train_report['shards']} shards "
          f"{train_report['frames_by_source']}", flush=True)
    print(f"val:   {val_report['windows']} windows / {val_report['shards']} shards "
          f"{val_report['frames_by_source']}", flush=True)

    model = StudentNet(width=args.width, hidden=args.hidden).to(device)
    parameters = model.n_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)

    order = np.arange(len(train_ds))
    rng.shuffle(order)
    cursor = 0
    step = 0
    seen = 0
    loss_window: list[float] = []
    train_started = time.monotonic()
    deadline = started + args.max_seconds
    while time.monotonic() < deadline:
        if cursor + args.batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        indices = order[cursor:cursor + args.batch_size]
        cursor += args.batch_size
        samples = [train_ds[int(index)] for index in indices]
        play, hud, move, bomb = batch_from_samples(samples, device)
        model.train()
        move_logits, bomb_logits = model(play, hud)
        loss = (torch.nn.functional.cross_entropy(move_logits, move)
                + torch.nn.functional.cross_entropy(bomb_logits, bomb))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1
        seen += len(indices)
        loss_window.append(float(loss.item()))
        if len(loss_window) > 50:
            loss_window.pop(0)
        if step % 50 == 0:
            elapsed = time.monotonic() - train_started
            print(f"step {step} seen {seen} loss {np.mean(loss_window):.4f} "
                  f"{seen / elapsed:.0f} samples/s", flush=True)

    train_seconds = time.monotonic() - train_started
    eval_indices = np.arange(len(val_ds))
    if len(eval_indices) > args.eval_samples:
        eval_indices = np.random.default_rng(args.seed + 1).choice(
            eval_indices, size=args.eval_samples, replace=False)
    eval_indices = np.sort(eval_indices)
    val_metrics = evaluate(model, val_ds, eval_indices, device, args.eval_batch)
    train_eval_indices = np.sort(np.random.default_rng(args.seed + 2).choice(
        np.arange(len(train_ds)), size=min(args.eval_samples, len(train_ds)), replace=False))
    train_metrics = evaluate(model, train_ds, train_eval_indices, device, args.eval_batch)

    train_info = {
        "kind": "bc-v0-behaviour-cloning",
        "disclosure": V0_DISCLOSURE,
        "steps": step,
        "samples_seen": int(seen),
        "train_seconds": round(train_seconds, 3),
        "wall_seconds": round(time.monotonic() - started, 3),
        "device": device,
        "device_name": (torch.cuda.get_device_name(0) if device == "cuda" else "cpu"),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": "AdamW(weight_decay=1e-4)",
        "loss": "CE(move[18]) + CE(bomb[2]) on hard collection-policy masks",
        "final_loss_window_mean": float(np.mean(loss_window)) if loss_window else None,
        "parameters": parameters,
        "train_dataset": {key: value for key, value in train_report.items()},
        "val_dataset": {key: value for key, value in val_report.items()},
        "val": {**val_metrics, "eval_samples": int(len(eval_indices))},
        "train_eval": {**train_metrics, "eval_samples": int(len(train_eval_indices))},
        "obs_spec_version": obs_spec.OBS_SPEC_VERSION,
        "obs_spec_digest": obs_spec.obs_spec_digest(),
        "label_kind": "collection policy mask (random/still/dive)",
        "not_teacher_labels": True,
        "not_a_p3_result": True,
        "not_a_g3_result": True,
    }
    checkpoint = Path(args.checkpoint)
    save_checkpoint(
        checkpoint,
        model.cpu(),
        train=train_info,
        provenance={
            "index": str(args.index),
            "split": str(args.split),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "kind": "bc-v0-behaviour-cloning",
            "disclosure": V0_DISCLOSURE,
        },
    )
    train_info["checkpoint"] = str(checkpoint)
    train_info["checkpoint_sha256"] = checkpoint_digest(checkpoint)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(train_info, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({
        "checkpoint": str(checkpoint),
        "parameters": parameters,
        "steps": step,
        "val_move_top1": val_metrics["move_top1"],
        "val_move_top3": val_metrics["move_top3"],
        "val_bomb_accuracy": val_metrics["bomb_accuracy"],
        "val_samples": val_metrics["samples"],
        "train_seconds": round(train_seconds, 2),
        "disclosure": V0_DISCLOSURE,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
