"""Train the four candidate heads and the pixel locator; score them on holdout.

The pre-registration (EXP-P3-010 0.3-0.6) is encoded here so a run cannot drift
from the card:

* one head, identical for every input, differing only in ``in_channels``;
* BCE on the 24x24 window cells, out-of-grid cells masked out;
* Adam, lr 3e-3, batch 256, 6 epochs, 3 training seeds (0/1/2) per input;
* F1 is micro-aggregated over holdout cells, threshold 0.5;
* the locator sees only the 64x64 plane, trains on seeds 1-14 and is scored on
  15-16 only; D is disqualified when its median error exceeds 4 px.

Nothing here reads a pixel for a label: labels come from the collector's
``labels_s*.npz``, which was produced from engine state.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from touhou_rl.v400.student.probe.inputs import INPUT_SPECS, build_input, smallest_first
from touhou_rl.v400.student.probe.labels import WINDOW_CELLS
from touhou_rl.v400.student.probe.native import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH

#: Pre-registered training configuration (EXP-P3-010 0.3).
EPOCHS = 6
BATCH = 256
LEARNING_RATE = 3e-3
TRAIN_SEEDS = (0, 1, 2)
THRESHOLD = 0.5
#: Pre-registered split: seeds 1-14 train, 15-16 holdout.
TRAIN_SPLIT = tuple(range(1, 15))
TEST_SPLIT = (15, 16)
LOCATOR_MEDIAN_GATE_PX = 4.0


class TrainError(RuntimeError):
    """A dataset or configuration precondition failed."""


# ---------------------------------------------------------------------------
# dataset over the collected native fields
# ---------------------------------------------------------------------------
def load_seed_labels(path: str | Path) -> dict[str, np.ndarray]:
    """Read one ``labels_s{seed}.npz`` into arrays with boolean windows."""

    with np.load(path) as payload:
        count = int(payload["anchors"].shape[0])
        cells = np.unpackbits(payload["cells"], axis=1)[:, :WINDOW_CELLS * WINDOW_CELLS]
        valid = np.unpackbits(payload["valid"], axis=1)[:, :WINDOW_CELLS * WINDOW_CELLS]
        danger = np.unpackbits(payload["danger"], axis=1)[:, :WINDOW_CELLS * WINDOW_CELLS]
        return {
            "count": np.asarray([count]),
            "cells": cells.astype(bool).reshape(count, WINDOW_CELLS, WINDOW_CELLS),
            "valid": valid.astype(bool).reshape(count, WINDOW_CELLS, WINDOW_CELLS),
            "danger": danger.astype(bool).reshape(count, WINDOW_CELLS, WINDOW_CELLS),
            "anchors": payload["anchors"].astype(np.int64),
            "players": payload["players"].astype(np.float32),
            "hurt": payload["hurt"].astype(np.float32),
            "would_die": payload["would_die"],
            "danger_frames": payload["danger_frames"],
        }


@dataclass
class SeedShard:
    """One collected seed: the native memmap plus its labels."""

    seed: int
    native_path: Path
    labels_path: Path
    count: int = 0
    _native: np.ndarray | None = field(default=None, repr=False)
    _labels: dict[str, np.ndarray] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.native_path.is_file():
            raise TrainError(f"missing native field {self.native_path}")
        if not self.labels_path.is_file():
            raise TrainError(f"missing labels {self.labels_path}")
        with np.load(self.labels_path) as payload:
            self.count = int(payload["anchors"].shape[0])

    @property
    def native(self) -> np.ndarray:
        if self._native is None:
            self._native = np.load(self.native_path, mmap_mode="r")
        return self._native

    @property
    def labels(self) -> dict[str, np.ndarray]:
        if self._labels is None:
            self._labels = load_seed_labels(self.labels_path)
        return self._labels

    def close(self) -> None:
        self._native = None
        self._labels = None


def open_shards(run_dir: Path, seeds: Iterable[int]) -> list[SeedShard]:
    """Open the collected shards for ``seeds``; fail closed when one is absent."""

    shards = []
    for seed in seeds:
        shards.append(SeedShard(
            seed=int(seed),
            native_path=Path(run_dir) / f"native_s{seed}.npy",
            labels_path=Path(run_dir) / f"labels_s{seed}.npz",
        ))
    return shards


def iter_samples(shards: Sequence[SeedShard], name: str, *,
                 limit: int | None = None):
    """Yield ``(tensor, anchor, cells, valid, seed, index)`` in seed order."""

    produced = 0
    for shard in shards:
        labels = shard.labels
        for index in range(shard.count):
            anchor_x = int(labels["anchors"][index, 1]) * 4
            anchor_y = int(labels["anchors"][index, 0]) * 4
            fields = {"native": np.asarray(shard.native[index])}
            tensor = build_input(name, fields, anchor_x=anchor_x, anchor_y=anchor_y)
            yield (tensor, labels["anchors"][index],
                   labels["cells"][index], labels["valid"][index],
                   shard.seed, index)
            produced += 1
            if limit is not None and produced >= int(limit):
                return


class BatchSource:
    """A shuffled in-memory index over the native memmaps.

    The inputs are rebuilt from the 384x448 native field on every batch instead
    of being cached at four resolutions: one source of truth, and 4.8 GB of
    duplicated arrays that would have to be re-derived for any variant.
    """

    def __init__(self, shards: Sequence[SeedShard], name: str, *,
                 limit: int | None = None, shuffle: bool = True,
                 seed: int = 0) -> None:
        if name not in INPUT_SPECS:
            raise TrainError(f"unknown input {name!r}")
        self.name = name
        self.shards = list(shards)
        offsets: list[tuple[int, int]] = []
        for shard_index, shard in enumerate(self.shards):
            take = shard.count if limit is None else min(shard.count, int(limit))
            offsets.extend((shard_index, row) for row in range(take))
        if not offsets:
            raise TrainError("no samples: every shard is empty")
        self._offsets = offsets
        self._rng = np.random.default_rng(seed)
        self.shuffle = bool(shuffle)
        self._order = self._rng.permutation(len(offsets)) if shuffle else np.arange(len(offsets))

    def __len__(self) -> int:
        return len(self._offsets)

    def epoch(self) -> None:
        if self.shuffle:
            self._order = self._rng.permutation(len(self._offsets))

    def batch(self, start: int, stop: int) -> dict[str, np.ndarray]:
        import torch

        indices = self._order[start:stop]
        spec = INPUT_SPECS[self.name]
        tensors = np.empty((len(indices), *spec.shape), dtype=np.float32)
        anchors = np.empty((len(indices), 2), dtype=np.int64)
        cells = np.empty((len(indices), WINDOW_CELLS, WINDOW_CELLS), dtype=np.float32)
        valid = np.empty((len(indices), WINDOW_CELLS, WINDOW_CELLS), dtype=np.float32)
        for slot, flat in enumerate(indices):
            shard_index, row = self._offsets[int(flat)]
            shard = self.shards[shard_index]
            labels = shard.labels
            anchor_x = int(labels["anchors"][row, 1]) * 4
            anchor_y = int(labels["anchors"][row, 0]) * 4
            fields = {"native": np.asarray(shard.native[row])}
            tensors[slot] = build_input(self.name, fields,
                                        anchor_x=anchor_x, anchor_y=anchor_y)
            anchors[slot] = labels["anchors"][row]
            cells[slot] = labels["cells"][row].astype(np.float32)
            valid[slot] = labels["valid"][row].astype(np.float32)
        return {
            "x": torch.from_numpy(tensors),
            "anchors": torch.from_numpy(anchors),
            "cells": torch.from_numpy(cells),
            "valid": torch.from_numpy(valid),
        }


# ---------------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------------
def _device(device: str | None):
    import torch

    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_probe(name: str, *, train_dir: Path, seeds: Sequence[int] = TRAIN_SPLIT,
                train_seed: int = 0, epochs: int = EPOCHS, batch: int = BATCH,
                lr: float = LEARNING_RATE, limit: int | None = None,
                device: str | None = None, log=print) -> dict[str, Any]:
    """Train one candidate input's head; returns weights + a training report."""

    import torch
    import torch.nn.functional as F

    from touhou_rl.v400.student.probe.net import ProbeModel

    torch.manual_seed(int(train_seed))
    np.random.seed(int(train_seed))
    dev = _device(device)
    shards = open_shards(Path(train_dir), seeds)
    source = BatchSource(shards, name, limit=limit, seed=train_seed)
    model = ProbeModel(name).to(dev)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    started = time.monotonic()
    samples_seen = 0
    for epoch in range(int(epochs)):
        model.train()
        source.epoch()
        total = 0.0
        batches = 0
        for start in range(0, len(source), int(batch)):
            payload = source.batch(start, min(start + int(batch), len(source)))
            x = payload["x"].to(dev, non_blocking=True)
            anchors = payload["anchors"].to(dev)
            target = payload["cells"].to(dev)
            valid = payload["valid"].to(dev)
            logits = model(x, anchors)
            per_cell = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none")
            denom = valid.sum().clamp_min(1.0)
            loss = (per_cell * valid).sum() / denom
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
            batches += 1
            samples_seen += int(x.shape[0])
        history.append({"epoch": float(epoch), "loss": total / max(1, batches)})
        log(f"[probe] {name} seed {train_seed} epoch {epoch}: "
            f"loss {history[-1]['loss']:.5f}")
    wall = time.monotonic() - started
    report = {
        "input": name,
        "train_seed": int(train_seed),
        "train_shards": [int(seed) for seed in seeds],
        "samples": len(source),
        "epochs": int(epochs),
        "batch": int(batch),
        "lr": float(lr),
        "device": str(dev),
        "history": history,
        "wall_s": round(wall, 3),
        "samples_per_s": round(samples_seen / wall, 2) if wall else 0.0,
        "parameters": model.head.parameter_count(),
        "final_loss": history[-1]["loss"] if history else None,
    }
    return {"model": model, "report": report, "shards": shards}


def evaluate_probe(model, name: str, *, test_dir: Path,
                   seeds: Sequence[int] = TEST_SPLIT, batch: int = 512,
                   device: str | None = None, limit: int | None = None) -> dict[str, Any]:
    """Micro-F1 plus per-seed counts on the holdout seeds."""

    import torch

    from touhou_rl.v400.student.probe.net import F1Counts, count_cells

    dev = _device(device)
    model.eval()
    shards = open_shards(Path(test_dir), seeds)
    per_seed: dict[str, dict[str, Any]] = {}
    total = F1Counts()
    empty_windows = 0
    windows = 0
    with torch.no_grad():
        for shard in shards:
            counts = F1Counts()
            labels = shard.labels
            take = shard.count if limit is None else min(shard.count, int(limit))
            for start in range(0, take, int(batch)):
                stop = min(start + int(batch), take)
                tensors = np.empty(
                    (stop - start, *INPUT_SPECS[name].shape), dtype=np.float32)
                anchors = labels["anchors"][start:stop]
                for slot, row in enumerate(range(start, stop)):
                    fields = {"native": np.asarray(shard.native[row])}
                    tensors[slot] = build_input(
                        name, fields, anchor_x=int(anchors[slot, 1]) * 4,
                        anchor_y=int(anchors[slot, 0]) * 4)
                logits = model(torch.from_numpy(tensors).to(dev),
                               torch.from_numpy(anchors.astype(np.int64)).to(dev))
                predicted = (torch.sigmoid(logits) >= THRESHOLD).cpu().numpy()
                target = labels["cells"][start:stop]
                valid = labels["valid"][start:stop]
                for slot in range(stop - start):
                    counts = counts.add(count_cells(
                        predicted[slot], target[slot], valid[slot]))
                    windows += 1
                    if not target[slot][valid[slot]].any():
                        empty_windows += 1
            per_seed[str(shard.seed)] = {**counts.as_dict(), "windows": int(take)}
            total = total.add(counts)
            shard.close()
    return {
        "input": name,
        "test_seeds": [int(seed) for seed in seeds],
        "threshold": THRESHOLD,
        "micro": total.as_dict(),
        "per_seed": per_seed,
        "windows": int(windows),
        "empty_windows": int(empty_windows),
    }


def naive_baselines(*, test_dir: Path, seeds: Sequence[int] = TEST_SPLIT,
                    limit: int | None = None) -> dict[str, Any]:
    """V5's trivial predictors: never, always, and a fixed blob at centre."""

    from touhou_rl.v400.student.probe.net import F1Counts, count_cells

    counts = {"never": F1Counts(), "always": F1Counts(), "centre_square": F1Counts()}
    centre = np.zeros((WINDOW_CELLS, WINDOW_CELLS), dtype=bool)
    centre[8:16, 8:16] = True
    for shard in open_shards(Path(test_dir), seeds):
        labels = shard.labels
        take = shard.count if limit is None else min(shard.count, int(limit))
        for row in range(take):
            target = labels["cells"][row]
            valid = labels["valid"][row]
            counts["never"] = counts["never"].add(
                count_cells(np.zeros_like(target), target, valid))
            counts["always"] = counts["always"].add(
                count_cells(np.ones_like(target), target, valid))
            counts["centre_square"] = counts["centre_square"].add(
                count_cells(centre, target, valid))
        shard.close()
    return {name: value.as_dict() for name, value in counts.items()}


# ---------------------------------------------------------------------------
# pixel player locator
# ---------------------------------------------------------------------------
def train_locator(*, train_dir: Path, seeds: Sequence[int] = TRAIN_SPLIT,
                  train_seed: int = 0, epochs: int = EPOCHS, batch: int = BATCH,
                  lr: float = LEARNING_RATE, limit: int | None = None,
                  device: str | None = None, log=print) -> dict[str, Any]:
    """Train the 64x64 heatmap locator on the training seeds only."""

    import torch
    import torch.nn.functional as F

    from touhou_rl.v400.student.probe.net import PlayerLocator, locator_target

    torch.manual_seed(int(train_seed))
    dev = _device(device)
    shards = open_shards(Path(train_dir), seeds)
    source = BatchSource(shards, "A", limit=limit, seed=train_seed)
    model = PlayerLocator().to(dev)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    target_cache: dict[tuple[int, int], np.ndarray] = {}
    history: list[dict[str, float]] = []
    started = time.monotonic()
    for epoch in range(int(epochs)):
        model.train()
        source.epoch()
        total = 0.0
        batches = 0
        for start in range(0, len(source), int(batch)):
            stop = min(start + int(batch), len(source))
            indices = source._order[start:stop]
            planes = np.empty((len(indices), 1, 64, 64), dtype=np.float32)
            targets = np.empty((len(indices), 1, 64, 64), dtype=np.float32)
            for slot, flat in enumerate(indices):
                shard_index, row = source._offsets[int(flat)]
                shard = shards[shard_index]
                fields = {"native": np.asarray(shard.native[row])}
                planes[slot] = build_input("A", fields)
                key = (shard.seed, row)
                if key not in target_cache:
                    player = shard.labels["players"][row]
                    target_cache[key] = locator_target(float(player[0]), float(player[1]))
                targets[slot, 0] = target_cache[key]
            logits = model(torch.from_numpy(planes).to(dev))
            per_pixel = F.binary_cross_entropy_with_logits(
                logits, torch.from_numpy(targets).to(dev), reduction="none")
            loss = per_pixel.mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
            batches += 1
        history.append({"epoch": float(epoch), "loss": total / max(1, batches)})
        log(f"[probe] locator seed {train_seed} epoch {epoch}: "
            f"loss {history[-1]['loss']:.5f}")
    wall = time.monotonic() - started
    return {
        "model": model,
        "shards": shards,
        "report": {
            "train_seed": int(train_seed),
            "train_shards": [int(seed) for seed in seeds],
            "epochs": int(epochs), "batch": int(batch), "lr": float(lr),
            "device": str(dev), "history": history, "wall_s": round(wall, 3),
            "parameters": int(sum(p.numel() for p in model.parameters())),
        },
    }


def evaluate_locator(model, *, test_dir: Path, seeds: Sequence[int] = TEST_SPLIT,
                     batch: int = 256, device: str | None = None,
                     limit: int | None = None) -> dict[str, Any]:
    """Median / p90 Euclidean error in native playfield pixels on holdout."""

    import torch

    from touhou_rl.v400.student.probe.net import decode_locator, locator_error_px

    dev = _device(device)
    model.eval()
    errors: list[float] = []
    per_seed: dict[str, list[float]] = {}
    for shard in open_shards(Path(test_dir), seeds):
        labels = shard.labels
        take = shard.count if limit is None else min(shard.count, int(limit))
        seed_errors: list[float] = []
        with torch.no_grad():
            for start in range(0, take, int(batch)):
                stop = min(start + int(batch), take)
                planes = np.empty((stop - start, 1, 64, 64), dtype=np.float32)
                for slot, row in enumerate(range(start, stop)):
                    planes[slot] = build_input(
                        "A", {"native": np.asarray(shard.native[row])})
                heat = torch.sigmoid(model(torch.from_numpy(planes).to(dev)))
                heat = heat.cpu().numpy()[:, 0]
                for slot, row in enumerate(range(start, stop)):
                    truth = labels["players"][row]
                    predicted = decode_locator(heat[slot])
                    error = locator_error_px(predicted, (float(truth[0]),
                                                         float(truth[1])))
                    seed_errors.append(error)
                    errors.append(error)
        per_seed[str(shard.seed)] = seed_errors
        shard.close()
    if not errors:
        raise TrainError("the locator evaluated zero holdout samples")
    array = np.asarray(errors, dtype=np.float64)
    return {
        "test_seeds": [int(seed) for seed in seeds],
        "samples": int(array.size),
        "median_px": float(np.median(array)),
        "mean_px": float(array.mean()),
        "p90_px": float(np.percentile(array, 90)),
        "max_px": float(array.max()),
        "share_within_4px": float((array <= 4.0).mean()),
        "gate_px": LOCATOR_MEDIAN_GATE_PX,
        "gate_passed": bool(float(np.median(array)) <= LOCATOR_MEDIAN_GATE_PX),
        "per_seed_median_px": {
            seed: round(float(np.median(values)), 4)
            for seed, values in per_seed.items() if values
        },
    }


# ---------------------------------------------------------------------------
# the pre-registered selection rule
# ---------------------------------------------------------------------------
def select_input(results: dict[str, dict[str, Any]], *,
                 locator: dict[str, Any] | None = None,
                 floor: float = 0.90, share: float = 0.95) -> dict[str, Any]:
    """Apply EXP-P3-010 0.6 verbatim and return the decision record."""

    f1 = {name: float(payload["micro"]["f1"]) for name, payload in results.items()}
    if not f1:
        raise TrainError("no input results to select from")
    best = max(f1.values())
    order = smallest_first()
    qualified: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    for name in order:
        value = f1[name]
        ok = value >= floor and value >= share * best
        reason = "f1>=0.90 and >=0.95*best" if ok else (
            "f1<0.90" if value < floor else "f1<0.95*best")
        if name == "D":
            gate = (locator or {}).get("gate_passed")
            if gate is False:
                ok = False
                reason = "locator median error > 4 px"
            elif gate is None:
                ok = False
                reason = "locator error not measured"
        qualified[name] = bool(ok)
        reasons[name] = reason
    chosen = next((name for name in order if qualified[name]), None)
    stop_loss = all(value < floor for value in f1.values())
    return {
        "f1": {name: round(value, 6) for name, value in f1.items()},
        "best": round(best, 6),
        "floor": floor,
        "share_of_best": share,
        "size_order": order,
        "areas": {name: INPUT_SPECS[name].pixels for name in order},
        "qualified": qualified,
        "reasons": reasons,
        "chosen": chosen,
        "stop_loss": bool(stop_loss),
        "resolution_not_the_bottleneck": bool(f1.get("A", 0.0) >= floor),
        "locator_median_px": (locator or {}).get("median_px"),
    }


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(target)


__all__ = [
    "BATCH",
    "BatchSource",
    "EPOCHS",
    "LEARNING_RATE",
    "LOCATOR_MEDIAN_GATE_PX",
    "SeedShard",
    "TEST_SPLIT",
    "THRESHOLD",
    "TRAIN_SEEDS",
    "TRAIN_SPLIT",
    "TrainError",
    "evaluate_locator",
    "evaluate_probe",
    "iter_samples",
    "load_seed_labels",
    "naive_baselines",
    "open_shards",
    "select_input",
    "train_locator",
    "train_probe",
    "write_json",
]
