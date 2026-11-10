"""Train the two obs-v2 arms (M2 MLP, M3 GRU) on the same corpus/split as obs-v1.

This is the EXP-P3-008 trainer.  It deliberately re-uses the frozen pieces:

* the corpus is loaded through :func:`touhou_rl.v400.student.dataset.load_teacher_shard`
  (the D0 gate: schema, obs-v1 digest, alignment, plane sizes and plane sha256);
* the split is :func:`touhou_rl.v400.student.teacher_bc.assign_shard_splits` with the
  same ``ratios``/``seed`` the obs-v1 baseline used, so M1/M2/M3 see the same shards;
* the label contract is :func:`teacher_bc.bomb_pos_weight` + ``KL(softmax(scores/tau)
  || pi_s)``; ``verified_mask`` stays a diagnostic, never a loss weight.

The only additions are the obs-v2 history planes (``act_hist``, ``stage``) from an
obs-v2 root built by :mod:`touhou_rl.v400.student.obs_v2_dataset`, and the sequence
sampler the GRU arm needs:

* a **window** is 8 consecutive decisions of one shard *and* one stage; the sampler
  never crosses a shard or a stage-clear boundary (same rule the history uses);
* the GRU forward runs the 8 steps, the loss is the mean over those steps, and BPTT is
  truncated at ``bptt`` (``bptt == seq_len`` by default: the whole window, no shorter);
* at evaluation the GRU is scored in *deployment mode* — the 8-step window ending at
  the decision, exactly the arithmetic :class:`GRUCache` performs live — so the
  offline number and the closed-loop policy use the same function.

Run::

    python -m touhou_rl.v400.student.obs_v2_train \
      --d0-root .tmp_v400_d0/d0 --extra-root .tmp_v400_dagger/d0_extra \
      --obsv2-root .tmp_v400_obsv2/obsv2 --arch gru \
      --checkpoint .tmp_v400_obsv2/arms/m3/checkpoint.pt \
      --metrics .tmp_v400_obsv2/arms/m3/metrics.json \
      --split-json .tmp_v400_obsv2/arms/m3/split.json \
      --tau 2.0 --train-seconds 690
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from touhou_rl.v400.gym.actions import MOVES_PER_CAND
from touhou_rl.v400.student import obs_spec, obs_spec_v2, teacher_bc
from touhou_rl.v400.student.dataset import (
    DatasetError,
    load_teacher_manifest,
    load_teacher_shard,
)
from touhou_rl.v400.student.model_v2 import (
    ARCH_GRU,
    ARCH_MLP,
    GRUCache,
    StudentGRU,
    StudentNetV2,
    checkpoint_digest,
    save_checkpoint_v2,
)
from touhou_rl.v400.student.obs_v2_dataset import load_obs_v2_manifest, load_obs_v2_shards

DEFAULT_TAU = 2.0
#: ``verified_mask`` is a diagnostic in every arm (11-P1 §3, teacher_bc docstring).
VERIFIED_MASK_USED_IN_LOSS = False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------
def complete_shard_rows(root: str | Path) -> list[dict[str, Any]]:
    """``shards/<key>`` rows whose ``shard.json`` is present and complete."""

    root = Path(root)
    out: list[dict[str, Any]] = []
    for shard_dir in sorted((root / "shards").iterdir()):
        meta_path = shard_dir / "shard.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("complete") is not True:
            continue
        out.append({
            "root": root,
            "dir": shard_dir,
            "shard": shard_dir.name,
            "decisions": int(meta.get("decision_count") or 0),
            "seed": int(meta.get("seed", -1)),
            "stage": meta.get("stage"),
            "difficulty": meta.get("difficulty"),
            "dagger": meta.get("dagger"),
        })
    return out


def _load_shard_arrays(row: dict[str, Any]) -> dict[str, Any]:
    """One shard's obs-v1 planes + teacher labels, validated by the frozen loader."""

    shard = load_teacher_shard(row["dir"], verify_sha=True)
    records = list(shard.records)
    count = len(records)
    play = np.array(shard.play[:], dtype=np.uint8)
    hud = np.array(shard.hud[:], dtype=np.uint8)
    scores = np.stack([np.asarray(record["scores"], dtype=np.float32) for record in records])
    bomb = np.asarray([int(record["bomb"]) for record in records], dtype=np.int8)
    verified = np.asarray([int(record["verified_count"]) for record in records],
                          dtype=np.int16)
    return {"play": play, "hud": hud, "scores": scores, "bomb": bomb,
            "verified": verified, "count": count,
            "frames_per_decision": shard.frames_per_decision}


@dataclass
class SplitArraysV2:
    """One split in RAM: obs-v1 planes + obs-v2 history + labels + segment ids."""

    name: str
    play: np.ndarray          # (N,4,64,64) uint8
    hud: np.ndarray           # (N,56,160) uint8
    hist: np.ndarray          # (N,8,18) uint8
    stage: np.ndarray         # (N,) int32
    scores: np.ndarray        # (N,18) float32
    bomb: np.ndarray          # (N,) int8
    verified_count: np.ndarray  # (N,)
    shard_index: np.ndarray   # (N,)
    segment: np.ndarray       # (N,) int32: shard/stage segment id (windows never cross)
    shard_keys: list[str]
    decisions_by_shard: dict[str, int]

    def __len__(self) -> int:
        return int(self.scores.shape[0])

    def report(self) -> dict[str, Any]:
        return {
            "split": self.name,
            "decisions": len(self),
            "shards": list(self.shard_keys),
            "shard_count": len(self.shard_keys),
            "decisions_by_shard": dict(sorted(self.decisions_by_shard.items())),
            "segments": int(self.segment.max() + 1) if len(self) else 0,
            "bomb_positives": int(self.bomb.sum()),
            "history_nonzero_rows": float((self.hist.sum(axis=2) > 0).mean()) if len(self) else 0.0,
        }


def assemble_splits(
    rows: Sequence[dict[str, Any]],
    assignments: dict[str, str],
    obsv2_root: str | Path,
) -> tuple[dict[str, SplitArraysV2], dict[str, Any]]:
    """Load every assigned shard once, then materialise the three splits."""

    by_key = {row["shard"]: row for row in rows}
    missing = sorted(set(assignments) - set(by_key))
    if missing:
        raise DatasetError(f"assigned shard(s) are not complete/readable: {missing}")
    v2_shards = load_obs_v2_shards(obsv2_root)
    per_key: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    for key in sorted(assignments):
        row = by_key[key]
        arrays = _load_shard_arrays(row)
        if key not in v2_shards:
            raise DatasetError(f"{key}: no obs-v2 shard in {obsv2_root}")
        added = v2_shards[key]
        if added.decisions != arrays["count"]:
            raise DatasetError(
                f"{key}: obs-v2 shard has {added.decisions} decisions but the v1 shard "
                f"has {arrays['count']}")
        arrays["hist"] = added.history()
        arrays["stage"] = np.array(added.stage, dtype=np.int32)
        expected_frames = added.meta.get("frames_per_decision")
        if int(expected_frames) != arrays["frames_per_decision"]:
            raise DatasetError(
                f"{key}: frames_per_decision {expected_frames!r} != "
                f"{arrays['frames_per_decision']}")
        per_key[key] = arrays
    load_seconds = time.monotonic() - started

    splits: dict[str, SplitArraysV2] = {}
    for name in teacher_bc.SPLIT_NAMES:
        keys = sorted(key for key, split in assignments.items() if split == name)
        if not keys:
            raise DatasetError(f"split {name} has no shards")
        play = np.concatenate([per_key[key]["play"] for key in keys], axis=0)
        hud = np.concatenate([per_key[key]["hud"] for key in keys], axis=0)
        hist = np.concatenate([per_key[key]["hist"] for key in keys], axis=0)
        stage = np.concatenate([per_key[key]["stage"] for key in keys], axis=0)
        scores = np.concatenate([per_key[key]["scores"] for key in keys], axis=0)
        bomb = np.concatenate([per_key[key]["bomb"] for key in keys], axis=0)
        verified = np.concatenate([per_key[key]["verified"] for key in keys], axis=0)
        shard_index = np.concatenate(
            [np.full(per_key[key]["count"], position, dtype=np.int16)
             for position, key in enumerate(keys)])
        # A window may not cross a shard or a stage clear: segment ids encode both.
        boundary = np.zeros(len(play), dtype=bool)
        if len(play):
            boundary[0] = True
            boundary[1:] = (shard_index[1:] != shard_index[:-1]) | (stage[1:] != stage[:-1])
        segment = np.cumsum(boundary) - 1
        decisions_by_shard = {key: int(per_key[key]["count"]) for key in keys}
        splits[name] = SplitArraysV2(
            name=name, play=play, hud=hud, hist=hist, stage=stage, scores=scores,
            bomb=bomb, verified_count=verified, shard_index=shard_index,
            segment=segment.astype(np.int32), shard_keys=keys,
            decisions_by_shard=decisions_by_shard)
        for key in keys:
            per_key[key]["play"] = None
            per_key[key]["hud"] = None
    gate = {
        "loader": "touhou_rl.v400.student.dataset.load_teacher_shard",
        "obsv2_root": str(Path(obsv2_root).resolve()),
        "obs_spec_version": obs_spec_v2.OBS_SPEC_VERSION,
        "obs_spec_digest": obs_spec_v2.obs_spec_v2_digest(),
        "base_obs_spec_version": obs_spec.OBS_SPEC_VERSION,
        "base_obs_spec_digest": obs_spec.obs_spec_digest(),
        "alignment": "decision_time_before_commit",
        "load_seconds": round(load_seconds, 3),
        "shards": len(per_key),
        "decisions": int(sum(per_key[key]["count"] for key in per_key)),
        "history_rows_per_decision": obs_spec_v2.ACTION_HISTORY_DECISIONS,
        "history_dims": obs_spec_v2.ACTION_HISTORY_DIM,
        "verified_mask_used_in_loss": VERIFIED_MASK_USED_IN_LOSS,
    }
    return splits, gate


# ---------------------------------------------------------------------------
# tensors, loss, metrics
# ---------------------------------------------------------------------------
def batch_decision(arrays: SplitArraysV2, indices: np.ndarray, device: str, torch):
    play = torch.from_numpy(arrays.play[indices]).to(device).float().div_(255.0)
    hud = torch.from_numpy(arrays.hud[indices]).to(device).float().div_(255.0).unsqueeze(1)
    hist = torch.from_numpy(arrays.hist[indices]).to(device).float()
    scores = torch.from_numpy(arrays.scores[indices]).to(device)
    bomb = torch.from_numpy(arrays.bomb[indices].astype(np.int64)).to(device)
    return play, hud, hist, scores, bomb


def valid_window_ends(arrays: SplitArraysV2, seq_len: int) -> np.ndarray:
    """Positions whose ``seq_len``-long window stays inside one shard/stage segment."""

    count = len(arrays)
    if count < seq_len:
        return np.zeros(0, dtype=np.int64)
    positions = np.arange(seq_len - 1, count, dtype=np.int64)
    same = arrays.segment[positions] == arrays.segment[positions - (seq_len - 1)]
    return positions[same]


def batch_sequence(arrays: SplitArraysV2, ends: np.ndarray, seq_len: int, device: str,
                   torch, *, bptt: int | None = None):
    """``(B,)`` window ends -> ``(S,B,...)`` tensors for the GRU arm."""

    offsets = np.arange(-(seq_len - 1), 1, dtype=np.int64)
    index = ends[:, None] + offsets[None, :]           # (B,S)
    play = torch.from_numpy(arrays.play[index]).to(device).float().div_(255.0)
    hud = torch.from_numpy(arrays.hud[index]).to(device).float().div_(255.0).unsqueeze(2)
    hist = torch.from_numpy(arrays.hist[index]).to(device).float()
    scores = torch.from_numpy(arrays.scores[index]).to(device)
    bomb = torch.from_numpy(arrays.bomb[index].astype(np.int64)).to(device)
    # (B,S,...) -> (S,B,...)
    return (play.permute(1, 0, 2, 3, 4).contiguous(),
            hud.permute(1, 0, 2, 3, 4).contiguous(),
            hist.permute(1, 0, 2, 3).contiguous(),
            scores.permute(1, 0, 2).contiguous(),
            bomb.permute(1, 0).contiguous())


def truncate_bptt(sequence, bptt: int, seq_len: int):
    """Truncated BPTT: detach the steps older than ``bptt``.

    The GRU still *consumes* every step of the window as input (so the recurrent state
    at the loss steps is the real 8-decision state), but no gradient flows into the
    encoder of the detached prefix.  ``bptt == seq_len`` detaches nothing, i.e. the
    whole window is backpropagated — the default for EXP-P3-008.
    """

    import torch

    if not 1 <= int(bptt) <= int(seq_len):
        raise ValueError(f"bptt {bptt} must be in [1, seq_len={seq_len}]")
    if int(bptt) == int(seq_len):
        return sequence
    keep = int(seq_len) - int(bptt)
    return torch.cat([sequence[:keep].detach(), sequence[keep:]], dim=0)


def obs_v2_loss(move_logits, bomb_logits, scores, bomb, *, tau: float, pos_weight: float,
                bomb_weight: float, torch):
    """``KL(softmax(scores/tau) || pi_s) + bomb_weight * BCE``, mean over every sample.

    Same target and same direction as obs-v1's BC v1 loss; only the reduction is
    spelled out (mean over steps x batch) so the GRU arm's 8-step loss is per-decision
    comparable to the MLP arm's single-step loss.
    """

    functional = torch.nn.functional
    log_probs = functional.log_softmax(move_logits, dim=-1)
    teacher = torch.softmax(scores / float(tau), dim=-1)
    safe = torch.where(teacher > 0, torch.log(torch.clamp(teacher, min=1e-30)),
                       torch.zeros_like(teacher))
    kl_per = (teacher * (safe - log_probs)).sum(dim=-1)
    targets = functional.one_hot(bomb, num_classes=obs_spec.N_BOMB_CLASSES).float()
    weight = torch.tensor([1.0, float(pos_weight)], device=bomb_logits.device)
    bce = functional.binary_cross_entropy_with_logits(
        bomb_logits, targets, weight=weight, reduction="mean")
    kl = kl_per.mean()
    return kl + float(bomb_weight) * bce, kl.detach(), bce.detach(), teacher


def metrics_from_logits(move_logits: np.ndarray, bomb_logits: np.ndarray,
                        arrays: SplitArraysV2, indices: np.ndarray, *, tau: float) -> dict:
    """The v1 metric set (KL / top-1 / top-3 / macro recall / collapse diagnostics)."""

    scores = arrays.scores[indices].astype(np.float64)
    teacher = np.exp((scores - scores.max(axis=-1, keepdims=True)) / float(tau))
    teacher /= teacher.sum(axis=-1, keepdims=True)
    logits = np.asarray(move_logits, dtype=np.float64)
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs -= np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
    safe = np.where(teacher > 0, np.log(np.clip(teacher, 1e-30, None)), 0.0)
    kl = float((teacher * (safe - log_probs)).sum(axis=-1).mean())
    teacher_argmax = teacher.argmax(axis=-1)
    pred = logits.argmax(axis=-1)
    top3 = np.argsort(-logits, axis=-1)[:, :3]
    ce = float(-log_probs[np.arange(len(pred)), teacher_argmax].mean())
    entropy = float(-(teacher * safe).sum(axis=-1).mean())
    support = np.bincount(teacher_argmax, minlength=obs_spec.N_MOVE_CLASSES)
    hits = np.bincount(teacher_argmax[pred == teacher_argmax],
                       minlength=obs_spec.N_MOVE_CLASSES)
    present = support > 0
    recalls = np.zeros(obs_spec.N_MOVE_CLASSES)
    recalls[present] = hits[present] / support[present]
    pred_hist = np.bincount(pred, minlength=obs_spec.N_MOVE_CLASSES)
    truth_hist = np.bincount(teacher_argmax, minlength=obs_spec.N_MOVE_CLASSES)
    bomb_pred = np.asarray(bomb_logits).argmax(axis=-1)
    bomb_true = arrays.bomb[indices]
    focus_directions = list(range(MOVES_PER_CAND + 1, obs_spec.N_MOVE_CLASSES))
    return {
        "samples": int(len(indices)),
        "kl": kl,
        "ce_argmax": ce,
        "teacher_entropy": entropy,
        "top1": float((pred == teacher_argmax).mean()),
        "top3": float((top3 == teacher_argmax[:, None]).any(axis=1).mean()),
        "bomb_accuracy": float((bomb_pred == bomb_true).mean()),
        "bomb_pred_positive_rate": float((bomb_pred > 0).mean()),
        "bomb_positives": int(bomb_true.sum()),
        "macro_recall": float(recalls[present].mean()) if present.any() else 0.0,
        "macro_recall_classes_present": int(present.sum()),
        "macro_recall_per_class": [round(float(value), 6) for value in recalls],
        "predicted_classes": sorted(int(value) for value in np.unique(pred)),
        "majority_class_baseline": float(support.max() / support.sum()),
        "pred_histogram": [int(value) for value in pred_hist],
        "teacher_argmax_histogram": [int(value) for value in truth_hist],
        "pred_action0_share": float(pred_hist[0] / pred.size),
        "pred_idle_share": float((pred_hist[0] + pred_hist[MOVES_PER_CAND]) / pred.size),
        "pred_unique_classes": int((pred_hist > 0).sum()),
        "pred_focus_classes_covered": int(sum(1 for value in focus_directions
                                              if pred_hist[value] > 0)),
        "teacher_action0_share": float(truth_hist[0] / truth_hist.sum()),
    }


def predict_mlp(model, arrays: SplitArraysV2, indices: np.ndarray, *, device: str,
                batch_size: int, torch) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    moves: list[np.ndarray] = []
    bombs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            play, hud, hist, _scores, _bomb = batch_decision(arrays, chunk, device, torch)
            move_logits, bomb_logits = model(play, hud, hist)
            moves.append(move_logits.detach().cpu().numpy())
            bombs.append(bomb_logits.detach().cpu().numpy())
    return np.concatenate(moves), np.concatenate(bombs)


def predict_gru(model, arrays: SplitArraysV2, indices: np.ndarray, *, device: str,
                batch_size: int, seq_len: int, torch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deployment-mode scoring: the ``seq_len`` window ending at each decision.

    Returns ``(move_logits, bomb_logits, used_indices)``.  Windows that would cross a
    shard/stage boundary are refused and reported, never padded with another episode.
    """

    model.eval()
    if len(indices) == 0:
        raise RuntimeError("evaluated zero samples")
    # ``indices - (seq_len-1)`` would wrap around for the first decisions of the split,
    # so those windows are refused explicitly instead of scored on wrapped data.
    early = indices >= (seq_len - 1)
    valid = np.zeros(len(indices), dtype=bool)
    valid[early] = arrays.segment[indices[early]] == arrays.segment[
        indices[early] - (seq_len - 1)]
    used = indices[valid]
    moves: list[np.ndarray] = []
    bombs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(used), batch_size):
            chunk = used[start:start + batch_size]
            play, hud, hist, _scores, _bomb = batch_sequence(
                arrays, chunk, seq_len, device, torch)
            move_logits, bomb_logits = model(play, hud, hist)
            moves.append(move_logits[-1].detach().cpu().numpy())
            bombs.append(bomb_logits[-1].detach().cpu().numpy())
    return np.concatenate(moves), np.concatenate(bombs), used


def evaluate(model, arrays: SplitArraysV2, indices: np.ndarray, *, tau: float,
             device: str, batch_size: int, arch: str, seq_len: int, torch) -> dict:
    """Offline metrics for either arm, on the same metric code."""

    if arch == ARCH_MLP:
        move_logits, bomb_logits = predict_mlp(
            model, arrays, indices, device=device, batch_size=batch_size, torch=torch)
        used = indices
    elif arch == ARCH_GRU:
        move_logits, bomb_logits, used = predict_gru(
            model, arrays, indices, device=device, batch_size=batch_size,
            seq_len=seq_len, torch=torch)
    else:  # pragma: no cover - argparse restricts arch
        raise ValueError(f"unknown arch {arch!r}")
    if len(used) == 0:
        raise RuntimeError("evaluated zero samples")
    metrics = metrics_from_logits(move_logits, bomb_logits, arrays, used, tau=tau)
    metrics["unscored_windows"] = int(len(indices) - len(used))
    metrics["window_scored"] = len(used) < len(indices)
    return metrics


# ---------------------------------------------------------------------------
# training loops
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    steps: int = 0
    samples: int = 0
    labels: int = 0
    seconds: float = 0.0
    losses: list[float] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    best_metrics: dict | None = None
    best_state: dict | None = None
    final_metrics: dict | None = None

    def curve_summary(self) -> dict:
        if not self.history:
            return {}
        best = max(self.history, key=lambda row: (row["top1"], -row["ce_argmax"]))
        return {
            "evaluations": len(self.history),
            "first": self.history[0],
            "last": self.history[-1],
            "best": best,
            "train_loss_first": self.losses[0] if self.losses else None,
            "train_loss_last": self.losses[-1] if self.losses else None,
            "train_loss_best": min(self.losses) if self.losses else None,
        }


def _train_step_mlp(model, optimizer, arrays, indices, *, device, tau, pos_weight,
                    bomb_weight, torch):
    play, hud, hist, scores, bomb = batch_decision(arrays, indices, device, torch)
    model.train()
    move_logits, bomb_logits = model(play, hud, hist)
    loss, kl, bce, _ = obs_v2_loss(move_logits, bomb_logits, scores, bomb, tau=tau,
                                   pos_weight=pos_weight, bomb_weight=bomb_weight, torch=torch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.item()), float(kl.item()), float(bce.item()), len(indices)


def train(
    *,
    arch: str,
    model,
    train_arrays: SplitArraysV2,
    val_arrays: SplitArraysV2,
    val_indices: np.ndarray,
    seconds: float,
    batch_size: int,
    seq_len: int,
    bptt: int,
    lr: float,
    weight_decay: float,
    pos_weight: float,
    bomb_weight: float,
    eval_every: int,
    device: str,
    seed: int,
    tau: float,
    torch,
) -> RunResult:
    """Time-budgeted training; keeps the best-by-val ``(top1, -ce)`` state."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    result = RunResult()
    ends = valid_window_ends(train_arrays, seq_len) if arch == ARCH_GRU else None
    if arch == ARCH_GRU and len(ends) == 0:
        raise RuntimeError("no valid GRU window in the train split")
    order = rng.permutation(len(train_arrays) if arch == ARCH_MLP else len(ends))
    cursor = 0
    deadline = time.monotonic() + float(seconds)
    step = 0
    while True:
        if time.monotonic() >= deadline:
            break
        source_len = len(train_arrays) if arch == ARCH_MLP else len(ends)
        if cursor + batch_size > source_len:
            order = rng.permutation(source_len)
            cursor = 0
        selection = order[cursor:cursor + batch_size]
        cursor += batch_size
        if arch == ARCH_MLP:
            loss, kl, bce, labels = _train_step_mlp(
                model, optimizer, train_arrays, selection, device=device, tau=tau,
                pos_weight=pos_weight, bomb_weight=bomb_weight, torch=torch)
            result.samples += labels
            result.labels += labels
        else:
            batch = ends[selection]
            play, hud, hist, scores, bomb = batch_sequence(
                train_arrays, batch, seq_len, device, torch)
            sequence = model.encode(play, hud, hist)
            sequence = truncate_bptt(sequence, bptt, seq_len)
            model.train()
            output, _ = model.gru(sequence)
            move_logits = model.move_head(output)
            bomb_logits = model.bomb_head(output)
            if bptt < seq_len:
                move_logits = move_logits[seq_len - bptt:]
                bomb_logits = bomb_logits[seq_len - bptt:]
                scores = scores[seq_len - bptt:]
                bomb = bomb[seq_len - bptt:]
            loss, kl, bce, _ = obs_v2_loss(
                move_logits, bomb_logits, scores, bomb, tau=tau, pos_weight=pos_weight,
                bomb_weight=bomb_weight, torch=torch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss, kl, bce = float(loss.item()), float(kl.item()), float(bce.item())
            result.samples += int(len(batch))
            result.labels += int(scores.shape[0] * scores.shape[1])
        step += 1
        result.steps = step
        result.losses.append(loss)
        if step % eval_every == 0 or time.monotonic() >= deadline:
            metrics = evaluate(model, val_arrays, val_indices, tau=tau, device=device,
                               batch_size=max(batch_size, 512), arch=arch, seq_len=seq_len,
                               torch=torch)
            metrics.update(step=step, samples=result.samples, labels=result.labels,
                           train_loss=float(np.mean(result.losses[-eval_every:])),
                           kl_loss=kl, bce_loss=bce)
            result.history.append(metrics)
            better = (result.best_metrics is None
                      or (metrics["top1"], -metrics["ce_argmax"])
                      > (result.best_metrics["top1"], -result.best_metrics["ce_argmax"]))
            if better:
                result.best_metrics = dict(metrics)
                result.best_state = {key: value.detach().cpu().clone()
                                     for key, value in model.state_dict().items()}
            elapsed = time.monotonic() - deadline + seconds
            print(f"[{arch}] step {step} seen {result.samples} labels {result.labels} "
                  f"loss {np.mean(result.losses[-eval_every:]):.4f} kl {kl:.4f} "
                  f"bce {bce:.4f} val_kl {metrics['kl']:.4f} "
                  f"val_top1 {metrics['top1']:.4f} ce {metrics['ce_argmax']:.4f} "
                  f"{result.labels / max(elapsed, 1e-6):.0f} labels/s", flush=True)
    result.seconds = float(seconds) - max(deadline - time.monotonic(), 0.0)
    result.final_metrics = evaluate(model, val_arrays, val_indices, tau=tau, device=device,
                                    batch_size=max(batch_size, 512), arch=arch,
                                    seq_len=seq_len, torch=torch)
    if result.best_state is not None:
        model.load_state_dict(result.best_state)
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d0-root", default=".tmp_v400_d0/d0")
    parser.add_argument("--extra-root", action="append", default=None, metavar="DIR",
                        help="further v1 root (EXP-P3-003 DAgger round-1 shards)")
    parser.add_argument("--obsv2-root", default=".tmp_v400_obsv2/obsv2")
    parser.add_argument("--arch", choices=("mlp", "gru"), default="mlp")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--train-seconds", type=float, default=690.0)
    parser.add_argument("--max-total-seconds", type=float, default=1200.0)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="decisions per step (MLP) or sequences per step (GRU)")
    parser.add_argument("--seq-len", type=int, default=obs_spec_v2.ACTION_HISTORY_DECISIONS)
    parser.add_argument("--bptt", type=int, default=obs_spec_v2.ACTION_HISTORY_DECISIONS)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bomb-weight", type=float, default=1.0)
    parser.add_argument("--bomb-pos-weight-max", type=float, default=100.0)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--val-eval-samples", type=int, default=2048)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--hist-hidden", type=int, default=64)
    parser.add_argument("--gru-input", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    started = time.monotonic()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    if args.arch == ARCH_GRU and not (1 <= args.bptt <= args.seq_len):
        raise SystemExit(f"bptt {args.bptt} must be in [1, seq_len={args.seq_len}]")

    d0_root = Path(args.d0_root)
    extra_roots = [Path(root) for root in (args.extra_root or [])]
    manifest = load_teacher_manifest(d0_root)
    rows = complete_shard_rows(d0_root)
    for root in extra_roots:
        if root.resolve() == d0_root.resolve():
            raise SystemExit(f"--extra-root {root} is the --d0-root; list it once")
        load_teacher_manifest(root)
        rows.extend(complete_shard_rows(root))
    if not rows:
        raise SystemExit(f"no complete shard under {d0_root}")
    keys = [row["shard"] for row in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate shard key(s) across roots")
    decisions = {row["shard"]: int(row["decisions"]) for row in rows}
    seeds = {row["shard"]: int(row["seed"]) for row in rows}
    total_decisions = sum(decisions.values())

    obs_v2_manifest = load_obs_v2_manifest(args.obsv2_root)
    if int(obs_v2_manifest["totals"]["decisions"]) != total_decisions:
        raise SystemExit(
            f"obs-v2 root holds {obs_v2_manifest['totals']['decisions']} decisions but the "
            f"v1 corpus totals {total_decisions}; the roots are not the same corpus")

    assignments = teacher_bc.assign_shard_splits(keys, ratios=teacher_bc.SPLIT_RATIOS,
                                                 seed=args.split_seed)
    split_report = teacher_bc.split_report(assignments, decisions=decisions, seeds=seeds)
    split_report["split_seed"] = args.split_seed
    split_report["signature"] = teacher_bc.split_signature(assignments)
    splits, gate = assemble_splits(rows, assignments, args.obsv2_root)
    print(json.dumps({name: {"shards": split_report[name]["shard_count"],
                             "decisions": split_report[name]["decisions"]}
                      for name in teacher_bc.SPLIT_NAMES}, indent=1), flush=True)

    train_arrays, val_arrays, test_arrays = splits["train"], splits["val"], splits["test"]
    if len(train_arrays) < args.batch_size:
        raise SystemExit(f"train split has only {len(train_arrays)} decisions")

    val_indices = np.arange(len(val_arrays))
    if args.val_eval_samples and len(val_indices) > args.val_eval_samples:
        val_indices = np.sort(np.random.default_rng(args.seed + 11).choice(
            val_indices, size=args.val_eval_samples, replace=False))
    train_eval_indices = np.arange(len(train_arrays))
    if args.val_eval_samples and len(train_eval_indices) > args.val_eval_samples:
        train_eval_indices = np.sort(np.random.default_rng(args.seed + 12).choice(
            train_eval_indices, size=args.val_eval_samples, replace=False))

    bomb_stats = teacher_bc.bomb_pos_weight(train_arrays.bomb,
                                            max_weight=args.bomb_pos_weight_max)
    if args.arch == ARCH_MLP:
        model = StudentNetV2(width=args.width, hidden=args.hidden,
                             hist_hidden=args.hist_hidden).to(device)
    else:
        model = StudentGRU(width=args.width, hidden=args.hidden,
                           hist_hidden=args.hist_hidden, gru_input=args.gru_input,
                           seq_len=args.seq_len).to(device)
    parameters = model.n_parameters()

    hard_deadline = started + args.max_total_seconds
    budget = max(10.0, min(args.train_seconds, hard_deadline - time.monotonic() - 30.0))
    run = train(
        arch=args.arch, model=model, train_arrays=train_arrays, val_arrays=val_arrays,
        val_indices=val_indices, seconds=budget, batch_size=args.batch_size,
        seq_len=args.seq_len, bptt=args.bptt, lr=args.lr,
        weight_decay=args.weight_decay, pos_weight=bomb_stats["pos_weight"],
        bomb_weight=args.bomb_weight, eval_every=args.eval_every, device=device,
        seed=args.seed + 1, tau=args.tau, torch=torch)

    metrics = {
        "train": evaluate(model, train_arrays, train_eval_indices, tau=args.tau,
                          device=device, batch_size=args.eval_batch, arch=args.arch,
                          seq_len=args.seq_len, torch=torch),
        "val": evaluate(model, val_arrays, np.arange(len(val_arrays)), tau=args.tau,
                        device=device, batch_size=args.eval_batch, arch=args.arch,
                        seq_len=args.seq_len, torch=torch),
        "test": evaluate(model, test_arrays, np.arange(len(test_arrays)), tau=args.tau,
                         device=device, batch_size=args.eval_batch, arch=args.arch,
                         seq_len=args.seq_len, torch=torch),
    }
    train_info = {
        "bc_version": teacher_bc.BC_VERSION,
        "disclosure": teacher_bc.TEACHER_BC_DISCLOSURE,
        "experiment": "EXP-P3-008",
        "obs_spec_version": obs_spec_v2.OBS_SPEC_VERSION,
        "obs_spec_digest": obs_spec_v2.obs_spec_v2_digest(),
        "base_obs_spec_version": obs_spec.OBS_SPEC_VERSION,
        "base_obs_spec_digest": obs_spec.obs_spec_digest(),
        "tau": float(args.tau),
        "arch": args.arch,
        "model": model.config(),
        "parameters": parameters,
        "obs_v2": {
            "root": str(Path(args.obsv2_root).resolve()),
            "manifest_sha256": _sha256_file(Path(args.obsv2_root) / "manifest.json"),
            "adds": "act_hist (8x18 one-hot, a[t-8..t-1])",
            "self_state": None,
            "danger_summary": None,
        },
        "d0": {**gate,
               "d0_root": str(d0_root.resolve()),
               "extra_roots": [str(root.resolve()) for root in extra_roots],
               "manifest_sha256": _sha256_file(d0_root / "manifest.json"),
               "manifest_totals": manifest.get("totals"),
               "complete_shards": len(rows),
               "decisions_in_complete_shards": total_decisions},
        "splits": split_report,
        "splits_report": {name: splits[name].report() for name in teacher_bc.SPLIT_NAMES},
        "metrics": metrics,
        "loss": "mean over (steps x batch) of KL(softmax(scores/tau) || pi_s) + "
                f"{args.bomb_weight} * BCE(bomb)",
        "loss_weights": {"kl": 1.0, "bomb": float(args.bomb_weight),
                         "bomb_pos_weight": float(bomb_stats["pos_weight"])},
        "optimizer": f"AdamW(lr={args.lr}, weight_decay={args.weight_decay})",
        "batch_size": args.batch_size,
        "sequence": {
            "arch_is_sequential": args.arch == ARCH_GRU,
            "seq_len": args.seq_len,
            "bptt": args.bptt,
            "bptt_is_truncated": args.bptt < args.seq_len,
            "window_rule": "8 consecutive decisions of one shard AND one stage; "
                           "boundaries are never crossed",
            "train_windows": int(len(valid_window_ends(train_arrays, args.seq_len)))
            if args.arch == ARCH_GRU else None,
        },
        "steps": run.steps,
        "samples_seen": run.samples,
        "labels_seen": run.labels,
        "train_seconds": round(run.seconds, 3),
        "wall_seconds": round(time.monotonic() - started, 3),
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "curve": run.curve_summary(),
        "best_val_at_step": run.best_metrics.get("step") if run.best_metrics else None,
        "metric_definitions": {
            "macro_recall": "mean over classes present in the teacher argmax of "
                            "|pred==c and teacher==c| / |teacher==c|",
            "top1": "argmax(pi_s) == argmax(pi_t) with pi_t = softmax(scores/tau)",
            "kl": "sum_a pi_t(a) (log pi_t(a) - log pi_s(a)), mean over decisions",
        },
        "not_a_g3_result": True,
        "not_the_confirm_split": True,
    }
    checkpoint_path = Path(args.checkpoint)
    save_checkpoint_v2(
        checkpoint_path, model.cpu(), train=train_info,
        provenance={
            "kind": teacher_bc.BC_VERSION,
            "experiment": "EXP-P3-008",
            "obs_spec_version": obs_spec_v2.OBS_SPEC_VERSION,
            "obs_spec_digest": obs_spec_v2.obs_spec_v2_digest(),
            "d0_root": str(d0_root),
            "split_signature": split_report["signature"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "controller": "touhou_rl.v400.student.obs_v2_train",
            "disclosure": teacher_bc.TEACHER_BC_DISCLOSURE,
        })
    train_info["checkpoint"] = str(checkpoint_path)
    train_info["checkpoint_sha256"] = checkpoint_digest(checkpoint_path)
    split_payload = {
        "schema": "v400-teacher-bc-split-1",
        "split_seed": args.split_seed,
        "signature": split_report["signature"],
        "assignments": dict(sorted(assignments.items())),
        "report": split_report,
        "d0": train_info["d0"],
        "obs_v2": train_info["obs_v2"],
    }
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(train_info, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
    if args.split_json:
        Path(args.split_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.split_json).write_text(
            json.dumps(split_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": train_info["checkpoint_sha256"],
        "arch": args.arch,
        "parameters": parameters,
        "tau": args.tau,
        "split_signature": split_report["signature"],
        "split_decisions": {name: split_report[name]["decisions"]
                            for name in teacher_bc.SPLIT_NAMES},
        "steps": run.steps,
        "labels_seen": run.labels,
        "train_seconds": train_info["train_seconds"],
        "wall_seconds": train_info["wall_seconds"],
        "train": metrics["train"],
        "val": metrics["val"],
        "test": metrics["test"],
        "disclosure": teacher_bc.TEACHER_BC_DISCLOSURE,
    }, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())


__all__ = [
    "SplitArraysV2",
    "assemble_splits",
    "batch_decision",
    "batch_sequence",
    "complete_shard_rows",
    "evaluate",
    "main",
    "metrics_from_logits",
    "obs_v2_loss",
    "predict_gru",
    "predict_mlp",
    "train",
    "valid_window_ends",
]
