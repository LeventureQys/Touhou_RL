"""Behaviour cloning of the P3 teacher on **D1** (obs-v3) with the v3 student.

This is T4's training half (13-P3 §2.4, EXP-P3-013).  It is deliberately a
separate module from ``tools/rl/v400_student_train_teacher.py``'s obs-v1 path and
from ``obs_v2_train.py``: the input contract is obs-v3 (96x112 play stack + HUD +
8x18 action history), the model is :class:`~touhou_rl.v400.student.model_v3.StudentV3`,
and the corpus can be 20 GB, so the reader is **mem-mapped** rather than copied
into RAM (``obs_v2_train`` copies; that is fine for obs-v1's 64x64 planes and not
fine for 2 M frames of 112x96).

Recipe::

    pi_t(a) = softmax(scores[18] / tau)
    L = 1.0 * KL(pi_t || pi_s)                  # 18 move-focus classes
      + 1.0 * BCE(bomb; pos_weight)             # 2 classes, one-hot target
      + 0.5 * perception aux (BCE over labelled aux maps)
      + 0.25 * Huber(value, value_target)
    verified_mask = diagnostics only, never a loss weight

Every term whose target is missing is **skipped and recorded as weight 0.0**
(:func:`touhou_rl.v400.student.model_v3.student_v3_loss`), so a run whose aux
labels are not wired yet cannot look as if it trained the aux heads.

Protocol (same discipline as EXP-P3-002/008 so the rows are comparable):

* **split** whole shards (one shard = one episode = one seed) 70/15/15 through
  ``teacher_bc.assign_shard_splits``, so no trajectory and no seed crosses
  train/val/test;
* **tau** defaults to 2.0, the value EXP-P3-008 pre-registered, so a T4 row is
  comparable with the obs-v2 arms; ``--tau`` overrides it and the value used is
  written into the checkpoint;
* the checkpoint is the best-by-val-teacher-top1 state of the run;
* dev seeds 1000-1019 are never trained on: the split is over D1 shards, whose
  seeds come from T3's collection seeds.

Gates this trainer enforces before it trains:

1. every assigned shard must load through
   :func:`touhou_rl.v400.student.obs_v3_dataset.load_teacher_shard_v3` — the
   frozen obs-v3 digest, the plane shapes, the sha256 of every plane, the label
   arithmetic and the ``act_hist`` rebuild; and
2. the aggregate decision count must reach ``--min-decisions`` (the D0/D1
   contract, :data:`touhou_rl.v400.student.dataset.MIN_TEACHER_SAMPLES`) unless
   ``--accept-partial-d1`` records a waiver with the real count.

``--gate-observations aggregate`` additionally runs T3's own gate
(``load_teacher_samples_v3(..., require_observations=True)``) over the assigned
label files.  It is off by default because that loader re-reads every label row
of the whole corpus (minutes and several GB at 2 M decisions), and the per-shard
gate already proves each training row has readable pixels: a shard with no
``obs/play96x112.npy`` cannot load at all.

Aux labels are **optional and per shard**: a shard may carry
``aux/occupancy.npy`` ``(N,112,96)``, ``aux/self_heatmap.npy`` ``(N,112,96)`` and
``aux/danger.npy`` ``(N,3,112,96)`` (uint8/bool, 1/4-resolution label grid,
danger channels in :data:`model_v3.AUX_HORIZONS` order).  Nothing in D1 produces
them yet (T3 owns the producer); a shard without them simply contributes no aux
term, and the metrics record that.

Run (GPU box, D1 on this machine)::

    python -m touhou_rl.v400.student.train_v3 \
      --d1-root /root/v400-data/d1 \
      --checkpoint .tmp_v400_p3v3/checkpoint-v3.pt \
      --metrics .tmp_v400_p3v3/train-metrics.json \
      --split-json .tmp_v400_p3v3/split.json \
      --train-seconds 5400 --batch-size 128 --accept-partial-d1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from touhou_rl.v400.student import dataset as student_dataset
from touhou_rl.v400.student import model_v3, obs_spec, obs_spec_v3, obs_v3_dataset
from touhou_rl.v400.student import teacher_bc

#: Teacher soft-label temperature this trainer uses unless overridden.  EXP-P3-008
#: pre-registered 2.0 for the obs-v2 arms; keeping the same value makes the T4 row
#: comparable with them instead of introducing a second free parameter.
DEFAULT_TAU = 2.0
#: ``verified_mask`` stays a diagnostic (11-P1 §3, teacher_bc §2.4).
VERIFIED_MASK_USED_IN_LOSS = False
#: Aux label arrays the reader looks for inside a shard, in head order.
AUX_FILES: dict[str, str] = {
    "occupancy": "aux/occupancy.npy",
    "self_heatmap": "aux/self_heatmap.npy",
    "danger": "aux/danger.npy",
}
#: Value-target rules.  Both are **proxies** for "the teacher's remaining survival
#: inside the episode"; D1 stores no return-to-go, so the real quantity is a known
#: gap (EXP-P3-013 "known missing"), recorded in every metrics file.
VALUE_TARGETS: tuple[str, ...] = ("verified-share", "margin", "none")
#: Saturating margin scale (px): ``min(margin, MARGIN_SATURATION_PX) / scale``.
MARGIN_SATURATION_PX = 16.0


class TrainV3Error(RuntimeError):
    """The D1 corpus, the split or a training argument violates the contract."""


# ---------------------------------------------------------------------------
# corpus: mem-mapped obs-v3 shards
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def complete_shard_rows(root: str | Path) -> list[dict[str, Any]]:
    """``shards/<key>`` rows whose ``shard.json`` is present and complete."""

    root = Path(root)
    shards_dir = root / "shards"
    if not shards_dir.is_dir():
        raise TrainV3Error(f"{root}: no shards/ directory")
    rows: list[dict[str, Any]] = []
    for shard_dir in sorted(shards_dir.iterdir()):
        meta_path = shard_dir / "shard.json"
        if not shard_dir.is_dir() or not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("complete") is not True:
            continue
        rows.append({
            "root": root,
            "dir": shard_dir,
            "shard": shard_dir.name,
            "decisions": int(meta.get("decision_count") or 0),
            "seed": int(meta.get("seed", -1)),
            "stage": meta.get("stage"),
            "difficulty": meta.get("difficulty"),
            "obs_spec_version": meta.get("obs_spec_version"),
            "obs_spec_digest": meta.get("obs_spec_digest"),
        })
    return rows


def _load_aux_maps(shard_dir: Path, decisions: int) -> dict[str, np.ndarray]:
    """Optional per-shard aux labels; absent files are simply not returned."""

    found: dict[str, np.ndarray] = {}
    for name, relative in AUX_FILES.items():
        path = shard_dir / relative
        if not path.is_file():
            continue
        array = np.load(path, mmap_mode="r")
        expected_channels = len(model_v3.AUX_HORIZONS) if name == "danger" else 1
        tail = ((*model_v3.AUX_SHAPE,) if expected_channels == 1
                else (expected_channels, *model_v3.AUX_SHAPE))
        if tuple(array.shape) != (int(decisions), *tail):
            raise TrainV3Error(
                f"{path}: shape {tuple(array.shape)} != {(int(decisions), *tail)}; the "
                "aux label grid is the 1/4-resolution label grid "
                f"{model_v3.AUX_SHAPE} with {expected_channels} channel(s)")
        if array.dtype not in (np.uint8, np.bool_, np.float32):
            raise TrainV3Error(f"{path}: dtype {array.dtype} is not uint8/bool/float32")
        found[name] = array
    return found


@dataclass
class ShardBlock:
    """One obs-v3 shard: mem-mapped planes + compact teacher/aux labels."""

    key: str
    dir: Path
    seed: int
    stage: int
    decision_count: int
    play: np.ndarray            # (N,4,112,96) uint8, mem-mapped
    hud: np.ndarray             # (N,56,160) uint8, mem-mapped
    act_hist: np.ndarray        # (N,8,18) uint8, mem-mapped
    scores: np.ndarray          # (N,18) float32
    bomb: np.ndarray            # (N,) int8
    verified_count: np.ndarray  # (N,) int16
    margin: np.ndarray          # (N,) float32
    chosen: np.ndarray          # (N,) int16
    value: np.ndarray           # (N,) float32, already normalised to [0,1]
    aux: dict[str, np.ndarray] = field(default_factory=dict)
    obs_spec_digest: str = ""

    def report(self) -> dict[str, Any]:
        return {
            "shard": self.key,
            "seed": int(self.seed),
            "stage": int(self.stage),
            "decisions": int(self.decision_count),
            "bomb_positives": int(self.bomb.sum()),
            "verified_mean": float(self.verified_count.mean()) if len(self) else 0.0,
            "value_mean": float(self.value.mean()) if len(self) else 0.0,
            "aux_components": sorted(self.aux),
            "obs_spec_digest": self.obs_spec_digest,
        }

    def __len__(self) -> int:
        return int(self.decision_count)


def load_shard_block(row: Mapping[str, Any], *,
                     value_target: str = "verified-share") -> ShardBlock:
    """Load one shard through the frozen obs-v3 loader, keep its planes mapped.

    The loader (:func:`obs_v3_dataset.load_teacher_shard_v3`) is the gate: it
    checks the frozen digest, every plane's shape/bytes/sha256, the label
    arithmetic and the ``act_hist`` rebuild.  Its ``records`` list is dropped
    after the compact label arrays are extracted, because a 2 M-decision corpus
    would otherwise hold every label dict in memory.
    """

    if value_target not in VALUE_TARGETS:
        raise TrainV3Error(f"value_target {value_target!r} not in {list(VALUE_TARGETS)}")
    shard = obs_v3_dataset.load_teacher_shard_v3(row["dir"])
    records = list(shard.records)
    count = len(records)
    if count != int(row.get("decisions", count)):
        raise TrainV3Error(
            f"{row['shard']}: loader returned {count} decisions, shard.json says "
            f"{row.get('decisions')}")
    scores = np.stack([np.asarray(record["scores"], dtype=np.float32)
                       for record in records])
    verified = np.asarray([int(record["verified_count"]) for record in records],
                          dtype=np.int16)
    margin = np.asarray([float(record["margin"]) for record in records], dtype=np.float32)
    chosen = np.asarray([int(record["chosen_plan"]["first_action"]) for record in records],
                        dtype=np.int16)
    bomb = np.asarray([int(record["bomb"]) for record in records], dtype=np.int8)
    stages = {int(record.get("stage", 0)) for record in records}
    if len(stages) != 1:
        raise TrainV3Error(f"{row['shard']}: {len(stages)} stages in one shard {stages}")
    if value_target == "verified-share":
        value = (verified.astype(np.float32) / float(obs_spec.N_MOVE_CLASSES))
    elif value_target == "margin":
        value = np.clip(np.nan_to_num(margin, nan=0.0, posinf=MARGIN_SATURATION_PX,
                                      neginf=0.0), 0.0, MARGIN_SATURATION_PX) \
            / MARGIN_SATURATION_PX
    else:
        value = np.zeros(count, dtype=np.float32)
    blocks_digest = str(shard.meta.get("obs_spec_digest"))
    return ShardBlock(
        key=str(row["shard"]),
        dir=Path(row["dir"]),
        seed=int(row.get("seed", -1)),
        stage=int(stages.pop()),
        decision_count=count,
        play=shard.play,
        hud=shard.hud,
        act_hist=load_shard_act_hist(shard),
        scores=scores,
        bomb=bomb,
        verified_count=verified,
        margin=margin,
        chosen=chosen,
        value=value.astype(np.float32),
        aux=_load_aux_maps(Path(row["dir"]), count),
        obs_spec_digest=blocks_digest,
    )


def load_shard_act_hist(shard: Any) -> np.ndarray:
    """The shard's ``act_hist`` plane, shape-checked (kept mem-mapped)."""

    expected = (len(shard), *obs_spec_v3.ACTION_HISTORY_SHAPE)
    array = shard.act_hist
    if tuple(array.shape) != expected:
        raise TrainV3Error(f"{shard.dir}: act_hist {tuple(array.shape)} != {expected}")
    return array


@dataclass
class SplitV3:
    """One split: its shard blocks in a stable order plus the row offsets."""

    name: str
    blocks: list[ShardBlock]

    def __post_init__(self) -> None:
        offsets: list[int] = []
        cursor = 0
        for block in self.blocks:
            offsets.append(cursor)
            cursor += len(block)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.count = int(cursor)

    def __len__(self) -> int:
        return self.count

    def locate(self, index: int) -> tuple[int, int]:
        if not 0 <= int(index) < self.count:
            raise TrainV3Error(f"row {index} outside split {self.name!r} (n={self.count})")
        position = int(np.searchsorted(self.offsets, int(index), side="right") - 1)
        return position, int(index) - int(self.offsets[position])

    def gather(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        """Gather a batch of rows from the mem-mapped planes + label arrays."""

        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size and (indices.min() < 0 or indices.max() >= self.count):
            raise TrainV3Error(f"row index out of range for split {self.name!r}")
        play = np.empty((indices.size, obs_spec_v3.FRAME_STACK,
                         *obs_spec_v3.PLAY_SHAPE), dtype=np.uint8)
        hud = np.empty((indices.size, *obs_spec_v3.HUD_SHAPE), dtype=np.uint8)
        hist = np.empty((indices.size, *obs_spec_v3.ACTION_HISTORY_SHAPE), dtype=np.uint8)
        value = np.empty(indices.size, dtype=np.float32)
        aux_keys = sorted({name for block in self.blocks for name in block.aux})
        aux: dict[str, np.ndarray] = {}
        for name in aux_keys:
            channel = (len(model_v3.AUX_HORIZONS) if name == "danger" else 1)
            tail = ((*model_v3.AUX_SHAPE,) if channel == 1
                    else (channel, *model_v3.AUX_SHAPE))
            aux[name] = np.zeros((indices.size, *tail), dtype=np.float32)
        positions = np.searchsorted(self.offsets, indices, side="right") - 1
        for position, block in enumerate(self.blocks):
            mask = positions == position
            if not mask.any():
                continue
            rows = indices[mask] - int(self.offsets[position])
            play[mask] = block.play[rows]
            hud[mask] = block.hud[rows]
            hist[mask] = block.act_hist[rows]
            value[mask] = block.value[rows]
            for name in aux_keys:
                if name in block.aux:
                    aux[name][mask] = block.aux[name][rows]
        return {
            "play": play,
            "hud": hud,
            "act_hist": hist,
            "scores": self._stack_labels("scores", indices),
            "bomb": self._stack_labels("bomb", indices),
            "verified_count": self._stack_labels("verified_count", indices),
            "value": value,
            "aux": aux,
        }

    def _stack_labels(self, name: str, indices: np.ndarray) -> np.ndarray:
        """Concatenated label arrays for ``name``, gathered in ``indices`` order."""

        arrays = [getattr(block, name) for block in self.blocks]
        if len(arrays) == 1:
            return arrays[0][indices]
        return np.concatenate(arrays, axis=0)[indices]

    def report(self) -> dict[str, Any]:
        return {
            "split": self.name,
            "decisions": self.count,
            "shards": [block.key for block in self.blocks],
            "shard_count": len(self.blocks),
            "decisions_by_shard": {block.key: len(block) for block in self.blocks},
            "seeds": sorted({int(block.seed) for block in self.blocks}),
            "aux_components": sorted({name for block in self.blocks for name in block.aux}),
            "bomb_positives": int(sum(int(block.bomb.sum()) for block in self.blocks)),
            "verified_mean": (
                float(np.mean([float(block.verified_count.mean()) for block in self.blocks]))
                if self.blocks else 0.0),
        }


def assemble_splits(rows: Sequence[Mapping[str, Any]], assignments: Mapping[str, str], *,
                    value_target: str = "verified-share") -> tuple[dict[str, SplitV3], dict]:
    """Load every assigned shard once, then build the three splits."""

    by_key = {str(row["shard"]): row for row in rows}
    missing = sorted(set(assignments) - set(by_key))
    if missing:
        raise TrainV3Error(f"assigned shard(s) are not complete/readable: {missing}")
    started = time.monotonic()
    blocks: dict[str, ShardBlock] = {}
    for key in sorted(assignments):
        blocks[key] = load_shard_block(by_key[key], value_target=value_target)
    load_seconds = time.monotonic() - started
    digests = sorted({block.obs_spec_digest for block in blocks.values()})
    if len(digests) != 1:
        raise TrainV3Error(f"shards disagree on the obs spec digest: {digests}")
    splits: dict[str, SplitV3] = {}
    for name in teacher_bc.SPLIT_NAMES:
        keys = sorted(key for key, split in assignments.items() if split == name)
        if not keys:
            raise TrainV3Error(f"split {name} has no shards")
        splits[name] = SplitV3(name=name, blocks=[blocks[key] for key in keys])
    gate = {
        "loader": "touhou_rl.v400.student.obs_v3_dataset.load_teacher_shard_v3",
        "obs_spec_version": obs_spec_v3.OBS_SPEC_VERSION,
        "obs_spec_digest": digests[0],
        "obs_spec_digest_expected": obs_spec_v3.obs_spec_v3_digest(),
        "alignment": "decision_time_before_commit",
        "load_seconds": round(load_seconds, 3),
        "shards": len(blocks),
        "decisions": int(sum(len(block) for block in blocks.values())),
        "planes_mem_mapped": True,
        "act_hist_shape": list(obs_spec_v3.ACTION_HISTORY_SHAPE),
        "play_shape": list(obs_spec_v3.PLAY_SHAPE),
        "hud_shape": list(obs_spec_v3.HUD_SHAPE),
        "verified_mask_used_in_loss": VERIFIED_MASK_USED_IN_LOSS,
    }
    return splits, gate


# ---------------------------------------------------------------------------
# tensors, losses, metrics
# ---------------------------------------------------------------------------
def batch_tensors(split: SplitV3, indices: np.ndarray, device: str, torch) -> dict:
    """One batch of torch tensors; ``aux`` holds only the labels that exist."""

    gathered = split.gather(indices)
    play = torch.from_numpy(gathered["play"]).to(device).float().div_(255.0)
    hud = torch.from_numpy(gathered["hud"]).to(device).float().div_(255.0).unsqueeze(1)
    act_hist = torch.from_numpy(gathered["act_hist"]).to(device).float()
    batch = {
        "play": play,
        "hud": hud,
        "act_hist": act_hist,
        "scores": torch.from_numpy(gathered["scores"]).to(device),
        "bomb": torch.from_numpy(gathered["bomb"].astype(np.int64)).to(device),
        "value": torch.from_numpy(gathered["value"]).to(device),
        "aux": {name: torch.from_numpy(array).to(device)
                for name, array in gathered["aux"].items() if array is not None},
    }
    if not batch["aux"]:
        batch["aux"] = None
    return batch


def macro_recall(predicted, teacher, *,
                 n_classes: int = obs_spec.N_MOVE_CLASSES) -> dict[str, Any]:
    """Macro-averaged recall over the classes present in the teacher target."""

    pred = np.asarray(predicted, dtype=np.int64).reshape(-1)
    truth = np.asarray(teacher, dtype=np.int64).reshape(-1)
    if pred.shape != truth.shape:
        raise TrainV3Error(f"predicted shape {pred.shape} != teacher shape {truth.shape}")
    if pred.size == 0:
        raise TrainV3Error("macro recall needs at least one sample")
    if pred.min() < 0 or truth.min() < 0 or pred.max() >= n_classes or truth.max() >= n_classes:
        raise TrainV3Error(f"labels must be in [0, {n_classes})")
    support = np.bincount(truth, minlength=n_classes)
    hits = np.bincount(truth[pred == truth], minlength=n_classes)
    present = support > 0
    recalls = np.zeros(n_classes, dtype=np.float64)
    recalls[present] = hits[present] / support[present]
    return {
        "macro_recall": float(recalls[present].mean()) if present.any() else 0.0,
        "classes_present": int(present.sum()),
        "per_class_recall": [round(float(value), 6) for value in recalls],
        "per_class_support": support.tolist(),
        "majority_class_baseline": float(support.max() / truth.size),
        "majority_class": int(support.argmax()),
        "predicted_classes": sorted(int(value) for value in np.unique(pred)),
    }


def evaluate(split: SplitV3, model: model_v3.StudentV3, indices: np.ndarray, *,
             tau: float, device: str, batch_size: int, torch) -> dict[str, Any]:
    """Offline metrics: KL, teacher top-1/top-3 agreement, macro recall, bomb, value."""

    model.eval()
    totals = {key: 0.0 for key in
              ("kl", "ce_argmax", "teacher_entropy", "top1", "top3", "bomb_correct",
               "bomb_pred_positive", "value_abs", "value_sq")}
    count = 0
    predicted: list[np.ndarray] = []
    teacher_argmax: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            batch = batch_tensors(split, chunk, device, torch)
            out = model(batch["play"], batch["hud"], batch["act_hist"], aux=False)
            log_probs = torch.nn.functional.log_softmax(out.move, dim=-1)
            teacher = model_v3.teacher_soft_labels(batch["scores"], tau)
            safe = torch.where(teacher > 0, torch.log(torch.clamp(teacher, min=1e-30)),
                               torch.zeros_like(teacher))
            kl = (teacher * (safe - log_probs)).sum(dim=-1)
            argmax = teacher.argmax(dim=-1)
            top3 = out.move.topk(3, dim=1).indices
            ce = -log_probs.gather(1, argmax[:, None]).squeeze(1)
            entropy = -(teacher * safe).sum(dim=-1)
            totals["kl"] += float(kl.sum())
            totals["ce_argmax"] += float(ce.sum())
            totals["teacher_entropy"] += float(entropy.sum())
            totals["top1"] += float(top3[:, 0].eq(argmax).sum())
            totals["top3"] += float(top3.eq(argmax[:, None]).any(dim=1).sum())
            totals["bomb_correct"] += float(out.bomb.argmax(dim=1).eq(batch["bomb"]).sum())
            totals["bomb_pred_positive"] += float(out.bomb.argmax(dim=1).sum())
            deltas = (out.value - batch["value"]).detach()
            totals["value_abs"] += float(deltas.abs().sum())
            totals["value_sq"] += float((deltas ** 2).sum())
            predicted.append(top3[:, 0].detach().cpu().numpy())
            teacher_argmax.append(argmax.detach().cpu().numpy())
            count += len(chunk)
    if count == 0:
        raise TrainV3Error("evaluated zero samples")
    pred = np.concatenate(predicted)
    truth = np.concatenate(teacher_argmax)
    recall = macro_recall(pred, truth)
    pred_hist = np.bincount(pred, minlength=obs_spec.N_MOVE_CLASSES)
    truth_hist = np.bincount(truth, minlength=obs_spec.N_MOVE_CLASSES)
    moves_per_cand = obs_spec.N_MOVE_CLASSES // 2
    return {
        "samples": int(count),
        "kl": totals["kl"] / count,
        "ce_argmax": totals["ce_argmax"] / count,
        "teacher_entropy": totals["teacher_entropy"] / count,
        "top1": totals["top1"] / count,
        "top3": totals["top3"] / count,
        "bomb_accuracy": totals["bomb_correct"] / count,
        "bomb_pred_positive_rate": totals["bomb_pred_positive"] / count,
        "value_mae": totals["value_abs"] / count,
        "value_rmse": math.sqrt(totals["value_sq"] / count),
        "macro_recall": recall["macro_recall"],
        "macro_recall_classes_present": recall["classes_present"],
        "macro_recall_per_class": recall["per_class_recall"],
        "predicted_classes": recall["predicted_classes"],
        "majority_class_baseline": recall["majority_class_baseline"],
        "pred_histogram": [int(value) for value in pred_hist],
        "teacher_argmax_histogram": [int(value) for value in truth_hist],
        "pred_action0_share": float(pred_hist[0] / pred.size),
        "pred_idle_share": float((pred_hist[0] + pred_hist[moves_per_cand]) / pred.size),
        "pred_unique_classes": int((pred_hist > 0).sum()),
        "pred_focus_classes_covered": int(sum(
            1 for value in range(moves_per_cand + 1, obs_spec.N_MOVE_CLASSES)
            if pred_hist[value] > 0)),
        "teacher_action0_share": float(truth_hist[0] / truth.size),
    }


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    steps: int = 0
    samples: int = 0
    train_seconds: float = 0.0
    losses: list[float] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    best_metrics: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    final_metrics: dict[str, Any] | None = None
    first_breakdown: dict[str, Any] | None = None
    last_breakdown: dict[str, Any] | None = None
    aborted: bool = False

    def curve_summary(self) -> dict[str, Any]:
        if not self.history:
            return {}
        first, last = self.history[0], self.history[-1]
        best = max(self.history, key=lambda row: (row["top1"], -row["ce_argmax"]))
        return {
            "evaluations": len(self.history),
            "first": first,
            "last": last,
            "best": best,
            "train_loss_first": self.losses[0] if self.losses else None,
            "train_loss_last": self.losses[-1] if self.losses else None,
            "train_loss_best": min(self.losses) if self.losses else None,
            "samples_per_second": round(self.samples / max(self.train_seconds, 1e-9), 3),
        }


def train_run(
    *,
    model: model_v3.StudentV3,
    train_split: SplitV3,
    val_split: SplitV3,
    val_indices: np.ndarray,
    seconds: float,
    batch_size: int,
    lr: float,
    weight_decay: float,
    tau: float,
    weights: Mapping[str, float],
    bomb_pos_weight: float,
    eval_every: int,
    device: str,
    seed: int,
    torch,
    max_steps: int = 0,
) -> RunResult:
    """Time-budgeted BC of one model; keeps the best-by-val state."""

    if int(batch_size) < 2:
        raise TrainV3Error("batch_size must be >= 2 (BatchNorm needs a batch)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    result = RunResult()
    order = rng.permutation(len(train_split))
    cursor = 0
    deadline = time.monotonic() + float(seconds)
    step = 0
    while True:
        if time.monotonic() >= deadline:
            break
        if cursor + batch_size > len(order):
            order = rng.permutation(len(train_split))
            cursor = 0
        indices = order[cursor:cursor + batch_size]
        cursor += batch_size
        if len(indices) < batch_size:  # pragma: no cover - order is reshuffled above
            continue
        batch = batch_tensors(train_split, indices, device, torch)
        model.train()
        out = model(batch["play"], batch["hud"], batch["act_hist"], aux=True)
        loss, breakdown = model_v3.student_v3_loss(
            out, batch, tau=tau, bomb_pos_weight=bomb_pos_weight, weights=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        step += 1
        result.steps = step
        result.samples += len(indices)
        result.losses.append(float(loss.detach()))
        if result.first_breakdown is None:
            result.first_breakdown = breakdown
        result.last_breakdown = breakdown
        if (step % eval_every == 0 or step == 1) and not (max_steps and step >= max_steps):
            metrics = evaluate(val_split, model, val_indices, tau=tau, device=device,
                               batch_size=max(batch_size, 256), torch=torch)
            metrics.update(step=step, samples=result.samples,
                           train_loss=float(np.mean(result.losses[-eval_every:])))
            result.history.append(metrics)
            better = (result.best_metrics is None
                      or (metrics["top1"], -metrics["ce_argmax"])
                      > (result.best_metrics["top1"], -result.best_metrics["ce_argmax"]))
            if better:
                result.best_metrics = dict(metrics)
                result.best_state = {key: value.detach().cpu().clone()
                                     for key, value in model.state_dict().items()}
            elapsed = time.monotonic() - deadline + seconds
            print(f"[v3] step {step} seen {result.samples} "
                  f"loss {np.mean(result.losses[-eval_every:]):.4f} "
                  f"val_kl {metrics['kl']:.4f} val_top1 {metrics['top1']:.4f} "
                  f"ce_argmax {metrics['ce_argmax']:.4f} "
                  f"{result.samples / max(elapsed, 1e-6):.0f} samples/s", flush=True)
        if max_steps and step >= max_steps:
            break
    result.train_seconds = max(float(seconds) - max(deadline - time.monotonic(), 0.0), 1e-9)
    result.aborted = time.monotonic() >= deadline
    result.final_metrics = evaluate(val_split, model, val_indices, tau=tau, device=device,
                                    batch_size=max(batch_size, 256), torch=torch)
    if result.best_state is not None:
        model.load_state_dict(result.best_state)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_channels(text: str, name: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in str(text).split(","):
        piece = part.strip()
        if not piece:
            continue
        if not piece.isdigit() or int(piece) <= 0:
            raise TrainV3Error(f"--{name} needs positive integers, got {part!r}")
        values.append(int(piece))
    if not values:
        raise TrainV3Error(f"--{name} is empty")
    return tuple(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-root", default=".tmp_v400_d1/d1")
    parser.add_argument("--extra-root", action="append", default=None, metavar="DIR",
                        help="further D1-shaped root whose complete shards join the "
                             "same aggregate and the same whole-episode split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--train-seconds", type=float, default=1800.0)
    parser.add_argument("--max-total-seconds", type=float, default=2400.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bomb-weight", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--value-target", choices=VALUE_TARGETS, default="verified-share",
                        help="proxy for the auxiliary value target; 'none' disables the "
                             "term (the effective weight is recorded either way)")
    parser.add_argument("--bomb-pos-weight-max", type=float, default=100.0)
    parser.add_argument("--grid-pos-weight", type=float, default=1.0,
                        help="pos_weight of the sparse aux BCE (occupancy/danger)")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--val-eval-samples", type=int, default=2048)
    parser.add_argument("--stage-channels", default=",".join(str(v) for v in
                                                             model_v3.STAGE_CHANNELS))
    parser.add_argument("--residual-blocks", type=int, default=model_v3.RESIDUAL_BLOCKS)
    parser.add_argument("--trunk", type=int, default=model_v3.TRUNK)
    parser.add_argument("--readout-channels", type=int, default=model_v3.READOUT_CHANNELS)
    parser.add_argument("--hud-width", type=int, default=16)
    parser.add_argument("--hud-hidden", type=int, default=model_v3.HUD_HIDDEN)
    parser.add_argument("--hist-hidden", type=int, default=model_v3.HIST_HIDDEN)
    parser.add_argument("--aux-channels", default=",".join(str(v) for v in
                                                           model_v3.AUX_CHANNELS))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--min-decisions", type=int,
                        default=int(student_dataset.MIN_TEACHER_SAMPLES))
    parser.add_argument("--accept-partial-d1", action="store_true",
                        help="train on a D1 below the decision contract, recording the "
                             "real count and a waiver in the metrics")
    parser.add_argument("--gate-observations", choices=("shard", "aggregate"),
                        default="shard",
                        help="'shard' (default) gates every shard through the frozen "
                             "obs-v3 loader; 'aggregate' additionally runs T3's "
                             "load_teacher_samples_v3(require_observations=True) over "
                             "the assigned label files")
    return parser


def _model_from_args(args) -> model_v3.StudentV3:
    return model_v3.StudentV3(
        stage_channels=_parse_channels(args.stage_channels, "stage-channels"),
        trunk=int(args.trunk),
        readout_channels=int(args.readout_channels),
        hud_width=int(args.hud_width),
        hud_hidden=int(args.hud_hidden),
        hist_hidden=int(args.hist_hidden),
        aux_channels=_parse_channels(args.aux_channels, "aux-channels"),
        residual_blocks=int(args.residual_blocks),
        dropout=float(args.dropout),
    )


def _aggregate_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """T3's loader gate over the assigned label files (expensive but complete)."""

    by_root: dict[Path, list[Path]] = {}
    for row in rows:
        by_root.setdefault(Path(row["root"]), []).append(Path(row["dir"]) / "labels.jsonl")
    reports: list[dict[str, Any]] = []
    for root, paths in sorted(by_root.items()):
        loaded = obs_v3_dataset.load_teacher_samples_v3(
            paths, min_samples=1, obs_root=root, require_observations=True)
        reports.append({
            "root": str(root),
            "files": len(paths),
            "decisions": int(loaded["count"]),
            "obs_spec_version": loaded["obs_spec_version"],
            "obs_spec_digest": loaded["obs_spec_digest"],
            "obs_shards": len(loaded.get("obs_shards") or {}),
            "unresolved_observations": len(loaded.get("obs_unresolved") or []),
        })
    return {
        "gate": "touhou_rl.v400.student.obs_v3_dataset.load_teacher_samples_v3",
        "require_observations": True,
        "report": reports,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    started = time.monotonic()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    if not float(args.tau) > 0:
        raise SystemExit(f"--tau must be positive, got {args.tau!r}")

    roots = [Path(args.d1_root), *[Path(root) for root in (args.extra_root or [])]]
    rows: list[dict[str, Any]] = []
    for root in roots:
        found = complete_shard_rows(root)
        print(f"{root}: {len(found)} complete obs-v3 shard(s)", flush=True)
        rows.extend(found)
    if not rows:
        raise SystemExit(f"no complete obs-v3 shard under {roots[0]}")
    keys = [row["shard"] for row in rows]
    if len(set(keys)) != len(keys):
        duplicated = sorted({key for key in keys if keys.count(key) > 1})
        raise SystemExit(f"duplicate shard key(s) across roots: {duplicated[:5]}")
    total_decisions = sum(int(row["decisions"]) for row in rows)

    contract = int(args.min_decisions)
    waived = total_decisions < contract
    if waived and not args.accept_partial_d1:
        raise SystemExit(
            f"D1 holds {total_decisions} decisions < the {contract} contract; pass "
            "--accept-partial-d1 to train on the real count (recorded as a waiver)")

    assignments = teacher_bc.assign_shard_splits(keys, ratios=teacher_bc.SPLIT_RATIOS,
                                                 seed=int(args.split_seed))
    # D1 covers stage_index 0..7 and re-uses small seed numbers per stage, so the
    # episode identity is (stage, seed) -- not the seed alone.  Two shards are the
    # same episode only when both agree; the duplicate (stage, seed) pairs are
    # quarantined before training, so this stays a strict isolation check.
    seeds = {
        row["shard"]: f"{int(row['stage'])}:{int(row['seed'])}" for row in rows
    }
    decisions = {row["shard"]: int(row["decisions"]) for row in rows}
    split_report = teacher_bc.split_report(assignments, decisions=decisions, seeds=seeds)
    split_report["split_seed"] = int(args.split_seed)
    split_report["signature"] = teacher_bc.split_signature(assignments)
    print(json.dumps({name: {"shards": split_report[name]["shard_count"],
                             "decisions": split_report[name]["decisions"]}
                      for name in teacher_bc.SPLIT_NAMES}, indent=1), flush=True)

    splits, gate = assemble_splits(rows, assignments, value_target=args.value_target)
    gate.update({
        "d1_root": str(roots[0].resolve()),
        "extra_roots": [str(root.resolve()) for root in roots[1:]],
        "complete_shards": len(rows),
        "decisions_in_complete_shards": total_decisions,
        "required_decisions": contract,
        "gate_waived": bool(waived),
        "gate_waiver_reason": (
            f"the corpus holds {total_decisions} decisions < {contract} when training "
            "started (collection still running); the real count is recorded and no "
            "contract size is claimed" if waived else None),
        "observation_gate": args.gate_observations,
        "require_observations": True,
    })
    if args.gate_observations == "aggregate":
        aggregate = _aggregate_gate(rows)
        gate["aggregate_gate"] = aggregate
        print(f"aggregate observation gate: "
              f"{sum(row['decisions'] for row in aggregate['report'])} decisions, "
              f"{sum(row['unresolved_observations'] for row in aggregate['report'])} "
              "unresolved", flush=True)

    train_split, val_split, test_split = splits["train"], splits["val"], splits["test"]
    if len(train_split) < int(args.batch_size):
        raise SystemExit(f"train split has only {len(train_split)} decisions")
    val_indices = np.arange(len(val_split))
    if args.val_eval_samples and len(val_indices) > args.val_eval_samples:
        val_indices = np.sort(np.random.default_rng(int(args.seed) + 11).choice(
            val_indices, size=int(args.val_eval_samples), replace=False))
    train_eval_indices = np.arange(len(train_split))
    if args.val_eval_samples and len(train_eval_indices) > args.val_eval_samples:
        train_eval_indices = np.sort(np.random.default_rng(int(args.seed) + 12).choice(
            train_eval_indices, size=int(args.val_eval_samples), replace=False))

    bomb = np.concatenate([block.bomb for block in train_split.blocks])
    pos_weight_info = teacher_bc.bomb_pos_weight(bomb,
                                                 max_weight=float(args.bomb_pos_weight_max))
    weights = {"kl": 1.0, "bomb": float(args.bomb_weight), "aux": float(args.aux_weight),
               "value": 0.0 if args.value_target == "none" else float(args.value_weight)}
    aux_pos_weight = {name: float(args.grid_pos_weight) for name in model_v3.AUX_COMPONENTS}
    aux_pos_weight["self_heatmap"] = 1.0

    model = _model_from_args(args)
    parameters = model.n_parameters()
    # The batches are materialised straight onto ``device`` (batch_tensors), so the
    # module has to live there too; without this the first conv raises
    # "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)".
    if hasattr(model, "to"):
        model = model.to(device)
    print(f"[v3] {parameters:,} parameters, canonical={model.is_canonical()}, "
          f"device={device}", flush=True)

    deadline = started + float(args.max_total_seconds)
    budget = max(1.0, min(float(args.train_seconds), deadline - time.monotonic() - 10.0))
    run = train_run(
        model=model, train_split=train_split, val_split=val_split,
        val_indices=val_indices, seconds=budget, batch_size=int(args.batch_size),
        lr=float(args.lr), weight_decay=float(args.weight_decay), tau=float(args.tau),
        weights=weights, bomb_pos_weight=float(pos_weight_info["pos_weight"]),
        eval_every=int(args.eval_every), device=device, seed=int(args.seed) + 1,
        torch=torch, max_steps=int(args.max_steps))

    metrics = {
        "train": evaluate(train_split, model, train_eval_indices, tau=float(args.tau),
                          device=device, batch_size=int(args.eval_batch), torch=torch),
        "val": evaluate(val_split, model, np.arange(len(val_split)), tau=float(args.tau),
                        device=device, batch_size=int(args.eval_batch), torch=torch),
        "test": evaluate(test_split, model, np.arange(len(test_split)), tau=float(args.tau),
                         device=device, batch_size=int(args.eval_batch), torch=torch),
    }
    aux_status = {
        "shards_with_aux": sorted(block.key for split in splits.values()
                                  for block in split.blocks if block.aux),
        "components_present": sorted({name for split in splits.values()
                                      for block in split.blocks for name in block.aux}),
        "expected_files": AUX_FILES,
        "note": ("no shard carries aux/*.npy yet, so the perception heads are built but "
                 "their loss is skipped with effective weight 0.0 (T3 owns the label "
                 "producer; see EXP-P3-013 'known missing')"
                 if not any(block.aux for split in splits.values()
                            for block in split.blocks)
                 else "aux labels are wired; only the components present are trained"),
    }
    train_info = {
        "bc_version": "bc-v3-teacher-soft-obs-v3",
        "disclosure": (
            "BC v3 = behaviour cloning of the P3 teacher's soft labels on the obs-v3 "
            "(96x112 + act_hist) D1 corpus with the IMPALA-ResNet student. It is an "
            "offline student baseline: its S1N episodes are not a G3 pass, not PSN and "
            "not the confirm split."),
        "tau": float(args.tau),
        "loss": ("KL(softmax(scores/tau) || pi_s) + bomb_weight*BCE + aux_weight*aux + "
                 "value_weight*Huber"),
        "loss_weights_requested": dict(weights),
        "loss_weights_effective": (run.last_breakdown or {}).get("effective_weights"),
        "loss_first_breakdown": run.first_breakdown,
        "loss_last_breakdown": run.last_breakdown,
        "aux_labels": aux_status,
        "value_target": {
            "rule": args.value_target,
            "proxy": args.value_target != "none",
            "definition": {
                "verified-share": "verified_count(18) / 18, i.e. the share of the 18 "
                                  "candidate actions the teacher verified as surviving "
                                  "at this decision; a survival-capacity proxy",
                "margin": f"min(max(margin,0), {MARGIN_SATURATION_PX}) / "
                          f"{MARGIN_SATURATION_PX} of the teacher's chosen plan",
                "none": "no value target; the term is skipped",
            }.get(args.value_target, ""),
            "known_gap": "D1 stores no return-to-go; both proxies are recorded as "
                         "approximations, not as the spec's 'remaining survival/reward'",
        },
        "d1": gate,
        "splits": split_report,
        "metrics": metrics,
        "optimizer": f"AdamW(lr={args.lr}, weight_decay={args.weight_decay})",
        "batch_size": int(args.batch_size),
        "steps": run.steps,
        "samples_seen": run.samples,
        "train_seconds": round(run.train_seconds, 3),
        "wall_seconds": round(time.monotonic() - started, 3),
        "samples_per_second": round(run.samples / max(run.train_seconds, 1e-9), 3),
        "parameters": parameters,
        "canonical_architecture": bool(model.is_canonical()),
        "model": model.config(),
        "device": device,
        "device_name": (torch.cuda.get_device_name(0) if device == "cuda" else "cpu"),
        "torch": torch.__version__,
        "bomb_pos_weight": pos_weight_info,
        "curve": run.curve_summary(),
        "best_val_at_step": (run.best_metrics or {}).get("step"),
        "aborted_at_deadline": bool(run.aborted),
        "obs_spec_version": obs_spec_v3.OBS_SPEC_VERSION,
        "obs_spec_digest": obs_spec_v3.obs_spec_v3_digest(),
        "verified_mask_used_in_loss": VERIFIED_MASK_USED_IN_LOSS,
        "not_a_g3_result": True,
        "not_the_confirm_split": True,
    }
    checkpoint_path = Path(args.checkpoint)
    model_v3.save_checkpoint_v3(
        checkpoint_path, model.cpu(), train=train_info,
        provenance={
            "kind": "bc-v3-teacher-soft-obs-v3",
            "d1_root": str(roots[0]),
            "split_signature": split_report["signature"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "controller": "touhou_rl/v400/student/train_v3.py",
            "disclosure": train_info["disclosure"],
        })
    train_info["checkpoint"] = str(checkpoint_path)
    train_info["checkpoint_sha256"] = model_v3.checkpoint_digest(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_info["checkpoint_verified"] = {
        "schema": payload.get("schema"),
        "arch": payload.get("arch"),
        "obs_spec_digest": payload.get("obs_spec_digest"),
        "parameters": payload.get("parameters"),
        "tau": (payload.get("train") or {}).get("tau"),
    }
    if int(payload["parameters"]) != parameters:
        raise TrainV3Error(
            f"the checkpoint records {payload['parameters']} parameters, the run built "
            f"{parameters}")

    outputs: list[tuple[Path, dict[str, Any]]] = [(Path(args.metrics), train_info)]
    if args.split_json:
        outputs.append((Path(args.split_json), {
            "schema": "v400-teacher-bc-split-3",
            "split_seed": int(args.split_seed),
            "signature": split_report["signature"],
            "assignments": dict(sorted(assignments.items())),
            "report": split_report,
            "d1": gate,
        }))
    for path, payload_json in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload_json, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": train_info["checkpoint_sha256"],
        "parameters": parameters,
        "tau": float(args.tau),
        "device": device,
        "steps": run.steps,
        "samples_seen": run.samples,
        "samples_per_second": train_info["samples_per_second"],
        "loss_weights_effective": train_info["loss_weights_effective"],
        "aux_labels": aux_status["components_present"],
        "split_decisions": {name: split_report[name]["decisions"]
                            for name in teacher_bc.SPLIT_NAMES},
        "d1_waived": bool(waived),
        "val": metrics["val"],
        "test": metrics["test"],
    }, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
