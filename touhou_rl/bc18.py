"""Privileged behaviour cloning into the 18-way entity policy (step 2 warm start).

What it consumes
----------------
The recorded decision traces from ``Document/Update/Alpha/v2.0.0/rollouts/**`` --
72,112 decisions carry a ``raw_safety_snapshot`` with the **whole active bullet
pool**, so the entity observation of :mod:`touhou_rl.entity_obs` can be rebuilt
offline.  The label is the scripted teacher's action at that decision
(``teacher_action`` -> ``focus*9 + move``); ``shoot``/``bomb`` are not part of the
candidate space (see the step-2 plan).

Why a cache
-----------
Parsing 72k JSONL rows of ~10 KB each on every epoch is minutes of CPU per pass.
``build_cache`` writes one ``.npz`` (bullet set ``float16`` to keep it near 280 MB,
meta ``float32``, labels ``int64``, a per-directory group id for leak-free
splits).  Nothing here touches the game.

Commands
--------
    python -m touhou_rl.bc18 --build --cache step2-privileged/bc-cache.npz
    python -m touhou_rl.bc18 --train --cache step2-privileged/bc-cache.npz \
        --out step2-privileged/bc --epochs 8 --device cuda
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from . import entity_obs, entity_policy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "Document/Update/Alpha/v2.0.0/rollouts"


def _trace_files(root) -> list[tuple[str, str]]:
    """``(group, path)`` for every decision trace under one or more roots.

    Multiple roots are supported so a step-2 retry can append freshly collected
    teacher demonstrations to the v2.0.0 corpus instead of overwriting it.  Group
    names are prefixed with the root index to stay unique across roots.
    """
    roots = [Path(root)] if isinstance(root, (str, Path)) else [Path(item) for item in root]
    files: list[tuple[str, str]] = []
    for root_index, base in enumerate(roots):
        if not base.is_dir():
            raise ValueError(f"rollouts root does not exist: {base}")
        # Traces can sit directly under the root (a fresh collection) or one level
        # down (the v2.0.0 per-round layout); both are scanned.
        directories = [base] + sorted(item for item in base.iterdir() if item.is_dir())
        for directory in directories:
            for path in sorted(directory.glob("episode-*-decisions.jsonl")):
                directory_name = ("root" if directory == base else directory.name)
                if len(roots) > 1:
                    directory_name = f"r{root_index}-{directory_name}"
                # One group per trace: leak-free validation that also works when a
                # corpus is a single directory holding many episodes (the step-2
                # teacher demonstrations are collected exactly that way).
                files.append((f"{directory_name}/{path.stem}", str(path)))
    return files


def _rows(path: str):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "executed_action" not in row or row.get("teacher_action") is None:
                continue
            snapshot = row.get("raw_safety_snapshot") or {}
            bullets = snapshot.get("bullets") or {}
            if not bullets.get("position"):
                continue
            yield row


def _label(row: dict) -> int:
    move, _shoot, focus, _bomb = (int(value) for value in row["teacher_action"])
    return entity_obs.candidate_index(move, focus)


def _count(files: list[tuple[str, str]]) -> int:
    total = 0
    for _group, path in files:
        for _ in _rows(path):
            total += 1
    return total


def build_cache(root: Path, cache: Path, progress=print) -> dict:
    files = _trace_files(root)
    if not files:
        raise ValueError(f"no decision traces under {root}")
    total = _count(files)
    if total == 0:
        raise ValueError("no trace row carries a raw_safety_snapshot")
    progress(f"[bc18] {len(files)} traces, {total} usable decisions")

    bullets = np.zeros((total, entity_obs.K_BULLETS, entity_obs.BULLET_FEATURES), dtype=np.float16)
    meta = np.zeros((total, entity_obs.META_DIM), dtype=np.float32)
    grid = np.zeros(
        (total, entity_obs.GRID_CHANNELS, entity_obs.GRID_ROWS, entity_obs.GRID_COLS),
        dtype=np.float16,
    )
    labels = np.zeros((total,), dtype=np.int64)
    groups = np.zeros((total,), dtype=np.int16)
    names = [group for group, _path in files]
    index = 0
    started = time.perf_counter()
    for group_id, (group, path) in enumerate(files):
        for row in _rows(path):
            observation = entity_obs.from_snapshot(row["raw_safety_snapshot"])
            bullets[index] = observation["bullets"].astype(np.float16)
            meta[index] = observation["meta"]
            grid[index] = observation["grid"].astype(np.float16)
            labels[index] = _label(row)
            groups[index] = group_id
            index += 1
        progress(f"[bc18]   {group:<32} cumulative={index}")
    if index != total:
        raise RuntimeError(f"counted {total} decisions but filled {index}")
    counts = np.bincount(labels, minlength=entity_policy.N_CANDIDATES).tolist()

    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        bullets=bullets,
        meta=meta,
        grid=grid,
        labels=labels,
        groups=groups,
        group_names=np.array(names),
        class_counts=np.array(counts, dtype=np.int64),
    )
    summary = {
        "decisions": int(total),
        "traces": len(files),
        "class_counts": counts,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "cache": str(cache),
    }
    progress(f"[bc18] cache written: {summary}")
    return summary


def load_cache(cache: Path) -> dict[str, np.ndarray]:
    with np.load(Path(cache), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def split_by_group(groups: np.ndarray, validation_groups: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the last ``validation_groups`` directories, never single samples."""
    unique = sorted(set(int(g) for g in groups.tolist()))
    if len(unique) <= validation_groups:
        raise ValueError(f"need more than {validation_groups} groups, got {len(unique)}")
    held = set(unique[-validation_groups:])
    validation = np.array([int(g) in held for g in groups.tolist()], dtype=bool)
    return np.flatnonzero(~validation), np.flatnonzero(validation)


def candidate_labels_from_actions(actions) -> np.ndarray:
    """Vector form of :func:`_label`, for tests and for narrated tooling."""
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] < 3:
        raise ValueError(f"actions must be [N,>=3], got {actions.shape}")
    return actions[:, 2].astype(np.int64) * 9 + actions[:, 0].astype(np.int64)


def class_weights(counts: np.ndarray, scheme: str = "inv-sqrt") -> np.ndarray | None:
    """Per-class loss weights that stop the majority move from swallowing the rest.

    The teacher idles in ~44% of decisions, so plain cross-entropy drives the
    argmax to "stand still": a first run predicted idle in 91% of decisions and
    died at the still-arm duration (~1214 frames).  ``inv-sqrt`` damps the
    imbalance without amplifying classes that appear a handful of times.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if scheme == "none":
        return None
    if scheme not in ("inv", "inv-sqrt"):
        raise ValueError(f"unknown class weighting scheme: {scheme}")
    present = counts > 0
    weights = np.zeros_like(counts)
    weights[present] = 1.0 / (counts[present] if scheme == "inv" else np.sqrt(counts[present]))
    # Keep the mean weight at 1 so the reported loss stays comparable across schemes.
    weights[present] = weights[present] * (present.sum() / weights[present].sum())
    return weights


def train(    cache: Path,
    out_dir: Path,
    *,
    epochs: int = 8,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    seed: int = 11,
    class_weighting: str = "inv-sqrt",
    progress=print,
) -> dict:
    import torch

    data = load_cache(cache)
    train_index, validation_index = split_by_group(data["groups"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    model = entity_policy.EntityPolicy().to(device_obj)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    bullets = data["bullets"]
    meta = data["meta"]
    grid = data["grid"]
    labels = data["labels"]
    weight_array = class_weights(np.bincount(labels, minlength=entity_policy.N_CANDIDATES), class_weighting)
    loss_weight = (
        None
        if weight_array is None
        else torch.as_tensor(weight_array, dtype=torch.float32, device=device_obj)
    )

    def _batch(indices: np.ndarray) -> tuple[dict, "torch.Tensor"]:
        selection = np.sort(indices)
        return (
            {
                "bullets": torch.as_tensor(bullets[selection].astype(np.float32), device=device_obj),
                "meta": torch.as_tensor(meta[selection], device=device_obj),
                "grid": torch.as_tensor(grid[selection].astype(np.float32), device=device_obj),
            },
            torch.as_tensor(labels[selection], device=device_obj),
        )

    history: list[dict] = []
    best = {"accuracy": -1.0, "epoch": -1}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(train_index)
        total_loss = 0.0
        correct = 0
        for start in range(0, order.size, batch_size):
            batch, target = _batch(order[start : start + batch_size])
            logits, _value = model(batch)
            loss = torch.nn.functional.cross_entropy(logits, target, weight=loss_weight)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += float(loss.detach()) * target.numel()
            correct += int((logits.argmax(dim=-1) == target).sum())

        model.eval()
        with torch.no_grad():
            batch, target = _batch(validation_index)
            logits, _value = model(batch)
            validation_loss = float(torch.nn.functional.cross_entropy(logits, target, weight=loss_weight))
            accuracy = float((logits.argmax(dim=-1) == target).float().mean())
            entropy = float(torch.distributions.Categorical(logits=logits).entropy().mean())
            predicted = torch.bincount(logits.argmax(dim=-1), minlength=entity_policy.N_CANDIDATES)
            predicted_share = (predicted.float() / max(1, target.numel())).cpu().numpy()
        row = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(1, order.size), 5),
            "train_accuracy": round(correct / max(1, order.size), 4),
            "validation_loss": round(validation_loss, 5),
            "validation_accuracy": round(accuracy, 4),
            "validation_entropy": round(entropy, 4),
            "validation_predicted_share": [round(float(v), 4) for v in predicted_share],
        }
        history.append(row)
        progress(f"[bc18] {row}")
        if accuracy > best["accuracy"]:
            best = {"accuracy": accuracy, "epoch": epoch}
            entity_policy.save_checkpoint(
                out_dir / "policy.pt",
                model,
                extra={
                    "source": "bc18",
                    "epoch": epoch,
                    "validation_accuracy": accuracy,
                    "train_decisions": int(train_index.size),
                    "validation_decisions": int(validation_index.size),
                    "cache": str(cache),
                    "class_weighting": class_weighting,
                },

            )

    summary = {
        "schema": "v311-bc18-summary/1",
        "cache": str(cache),
        "decisions": int(data["labels"].size),
        "train_decisions": int(train_index.size),
        "validation_decisions": int(validation_index.size),
        "class_counts": data["class_counts"].tolist(),
        "class_weighting": class_weighting,
        "validation_label_share": [
            round(float(v), 4)
            for v in (
                np.bincount(labels[validation_index], minlength=entity_policy.N_CANDIDATES)
                / max(1, validation_index.size)
            )
        ],
        "best": best,
        "history": history,
        "checkpoint": str(out_dir / "policy.pt"),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 2 privileged behaviour cloning (18-way)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true", help="build the .npz cache from recorded traces")
    mode.add_argument("--train", action="store_true", help="train on an existing cache")
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="rollouts root for --build; repeat to merge several corpora (default: the v2.0.0 rollouts)",
    )
    parser.add_argument("--cache", required=True, help="cache path (.npz)")
    parser.add_argument("--out", default="", help="output directory for --train")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--class-weighting", default="inv-sqrt", choices=["none", "inv", "inv-sqrt"])
    args = parser.parse_args(argv)
    if not args.root:
        args.root = [str(DEFAULT_ROOT)]

    if args.build:
        build_cache([Path(item) for item in args.root], Path(args.cache))
        return 0
    if not args.out:
        parser.error("--train requires --out")
    summary = train(
        Path(args.cache),
        Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        class_weighting=args.class_weighting,
    )
    print(json.dumps({"best": summary["best"], "checkpoint": summary["checkpoint"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
