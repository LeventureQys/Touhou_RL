"""Train **BC v1**: the obs-v1 pixel student on the P3 teacher's soft labels.

This is the first real P3 student step (EXP-P3-002): the labels are the
teacher's ``scores[18]`` (D0 / EXP-P1-018), not the P2 collection policies'
button masks that V0 cloned.

Recipe (13-P3 §2.4, frozen by :mod:`touhou_rl.v400.student.teacher_bc`)::

    pi_t(a) = softmax(scores[18] / tau)
    loss    = KL(pi_t || pi_s) + BCE(bomb; pos_weight)
    verified_mask = diagnostics only, never a loss weight

Protocol:

* **split** whole D0 shards (one shard = one episode = one seed) 70/15/15, so
  no trajectory and no seed can cross train/val/test;
* **tau** is chosen on val from {1, 2, 4} by a short equal-budget probe per tau,
  ranked by val teacher top-1 agreement (val KL is reported, not ranked: the
  target entropy grows with tau);
* the selected tau is then retrained from scratch at the full budget, and the
  checkpoint is the best-by-val-top1 state of that run;
* test shards are only ever scored offline; the closed-loop S1N evaluation uses
  seeds 1000-1004 (``tools/rl/v400_student_eval.py``), which appear in no split.

Run (GPU box, dataset on this machine)::

    python tools/rl/v400_student_train_teacher.py \
      --d0-root .tmp_v400_d0/d0 \
      --checkpoint .tmp_v400_p3v1/checkpoint-v1.pt \
      --metrics .tmp_v400_p3v1/train-metrics.json \
      --sweep-seconds 110 --train-seconds 420

T4 (obs-v3) is a third route through the same entry point, added without
changing either existing one::

    python tools/rl/v400_student_train_teacher.py \
      --obs v3 --arch v3 --taus 2.0 --d1-root /root/v400-data/d1 \
      --checkpoint .tmp_v400_p3v3/checkpoint-v3.pt \
      --metrics .tmp_v400_p3v3/train-metrics.json \
      --split-json .tmp_v400_p3v3/split.json \
      --train-seconds 5400 --batch-size 128 --accept-partial-d1

``--obs v3``/``--arch v3`` forwards to :mod:`touhou_rl.v400.student.train_v3`
(see :func:`route_obs`); ``--obs v3`` alone is enough, and every other flag
combination behaves exactly as it did before T4.  The default ``--taus`` is the
obs-v1 sweep (``1.0,2.0,4.0``), and a sweep is refused by both the obs-v2 and the
obs-v3 routes for the same reason (one pre-registered tau), so a v3 invocation
passes ``--taus 2.0``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from touhou_rl.v400.gym.actions import MOVES_PER_CAND  # noqa: E402
from touhou_rl.v400.student import obs_spec  # noqa: E402
from touhou_rl.v400.student import teacher_bc  # noqa: E402
from touhou_rl.v400.student.dataset import (  # noqa: E402
    MIN_TEACHER_SAMPLES,
    load_teacher_manifest,
    load_teacher_samples,
)
from touhou_rl.v400.student.model import StudentNet, checkpoint_digest, save_checkpoint  # noqa: E402
from touhou_rl.v400.teacher import config as teacher_config  # noqa: E402


#: 11-P1 §3's "unverified candidates are discounted" constant, read from the
#: teacher's own ``Weights`` so the geometric restore can never drift from the
#: value the labels were written with (``teacher/label.py:161-163`` subtracts
#: exactly this from every candidate whose ``verified_mask`` is False).
UNVERIFIED_DISCOUNT: float = float(teacher_config.Weights().unverified_discount)


def geo_restore_scores(
    scores,
    verified_mask,
    *,
    discount: float = UNVERIFIED_DISCOUNT,
):
    """Undo 11-P1 §3's unverified discount: ``scores + discount * (1 - mask)``.

    EXP-P3-004 §2: the label a decision carries is
    ``scores[c] = geo[c]`` when ``c`` was engine-verified and
    ``scores[c] = geo[c] - 5.0`` otherwise (``teacher/label.py``).  Restoring is
    a **value-exact inverse**, because the discount is a constant: the same
    ``+5.0`` is added to every unverified candidate and to nothing else.

    With ``fast`` (``verify_k=1``) exactly one candidate is verified, so this
    adds 5.0 to 17 of the 18 numbers and the softmax target flattens to almost
    uniform (EXP-P3-003 §8.3/§8.4) — that is why the flag is an A/B arm and not
    a default.

    The mask is *not* re-derived from the scores: a verified candidate whose
    score happens to sit 5.0 below another is left alone, which is exactly what
    ``verified_mask`` is for.
    """

    array = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(verified_mask, dtype=bool)
    if array.shape != mask.shape:
        raise ValueError(
            f"scores shape {array.shape} != verified_mask shape {mask.shape}; the mask "
            "is per-candidate")
    if array.shape[-1] != obs_spec.N_MOVE_CLASSES:
        raise ValueError(
            f"the last axis must hold {obs_spec.N_MOVE_CLASSES} candidates, got {array.shape}")
    if not np.isfinite(discount):
        raise ValueError(f"discount must be finite, got {discount!r}")
    restored = array + float(discount) * (~mask)
    if not np.all(np.isfinite(restored)):
        raise ValueError("restoring produced non-finite scores")
    return restored


def macro_recall(
    predicted,
    teacher,
    *,
    n_classes: int = obs_spec.N_MOVE_CLASSES,
) -> dict:
    """Macro-averaged recall of ``predicted`` against ``teacher`` (both hard labels).

    Macro recall is averaged over the classes **present in the teacher target**
    (``recall_c = |pred == c and teacher == c| / |teacher == c|``).  It is the
    metric the P3 cards quote (EXP-P3-002/003: 0.2121 / 0.1912) and it is the
    one that exposes "69 % of the predictions are action 0": top-1 can look
    healthy while the model never emits a rare class.
    """

    pred = np.asarray(predicted, dtype=np.int64).reshape(-1)
    truth = np.asarray(teacher, dtype=np.int64).reshape(-1)
    if pred.shape != truth.shape:
        raise ValueError(f"predicted shape {pred.shape} != teacher shape {truth.shape}")
    if pred.size == 0:
        raise ValueError("macro recall needs at least one sample")
    if pred.min() < 0 or truth.min() < 0 or pred.max() >= n_classes or truth.max() >= n_classes:
        raise ValueError(f"labels must be in [0, {n_classes})")
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



def _torch():
    import torch

    return torch


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# dataset assembly
# ---------------------------------------------------------------------------
@dataclass
class SplitArrays:
    """One split held flat in RAM: uint8 observations + teacher labels."""

    name: str
    play: np.ndarray
    hud: np.ndarray
    scores: np.ndarray
    bomb: np.ndarray
    verified_count: np.ndarray
    shard_index: np.ndarray
    shard_keys: list[str]
    decisions_by_shard: dict[str, int]
    #: The labels exactly as they sit on disk (``-5.0`` on every unverified
    #: candidate).  Equal to ``scores`` unless ``--geo-restore`` rewrote it, so
    #: the run can always report both distributions instead of only the arm's.
    scores_raw: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.scores.shape[0])

    def report(self) -> dict:
        return {
            "split": self.name,
            "decisions": len(self),
            "shards": list(self.shard_keys),
            "shard_count": len(self.shard_keys),
            "decisions_by_shard": dict(sorted(self.decisions_by_shard.items())),
            "bomb_positives": int(self.bomb.sum()),
            "verified_mean": float(self.verified_count.mean()) if len(self) else 0.0,
        }


def _complete_shards(d0_root: Path, manifest: dict) -> list[dict]:
    """Manifest rows whose ``shard.json`` is present and marked complete."""

    rows = []
    root = Path(d0_root)
    for row in manifest.get("shards") or []:
        key = str(row["shard"])
        shard_dir = root / "shards" / key
        shard_json = shard_dir / "shard.json"
        if not shard_json.is_file():
            print(f"skip shard {key}: no shard.json", flush=True)
            continue
        payload = json.loads(shard_json.read_text(encoding="utf-8"))
        if payload.get("complete") is not True:
            print(f"skip shard {key}: incomplete", flush=True)
            continue
        if not (shard_dir / "labels.jsonl").is_file():
            print(f"skip shard {key}: no labels.jsonl", flush=True)
            continue
        rows.append({
            "shard": key,
            "dir": shard_dir,
            "root": root,
            "decisions": int(payload.get("decision_count") or row.get("decisions") or 0),
            "seed": int(payload.get("seed", row.get("seed", -1))),
            "stage": payload.get("stage", row.get("stage")),
            "difficulty": payload.get("difficulty", row.get("difficulty")),
            "stop_reason": payload.get("stop_reason"),
            "generated_by": payload.get("generated_by"),
            "dagger": payload.get("dagger"),
        })
    return rows


def _load_shard_arrays(shard: object, *, geo_restore: bool = False) -> dict:
    """Copy one loaded ``TeacherShard``'s planes/labels into flat arrays.

    ``geo_restore`` (EXP-P3-004) undoes 11-P1 §3's ``-5.0`` on every candidate
    the teacher did not verify: :func:`geo_restore_scores` adds the discount back
    value by value, so the training target becomes ``softmax(geo/tau)`` instead
    of ``softmax((geo - 5 * ~mask)/tau)``.  The stored ``scores`` array is never
    modified in place — the fixture on disk stays the evidence.
    """

    records = list(shard.records)
    count = len(records)
    if count == 0:
        raise RuntimeError(f"shard {shard.meta.get('shard')!r} has no rows")
    rows = [int(record.get("row", index)) for index, record in enumerate(records)]
    if rows != list(range(count)):
        raise RuntimeError(f"shard {shard.meta.get('shard')!r}: label rows are not 0..N-1")
    play = np.array(shard.play[:], dtype=np.uint8)
    hud = np.array(shard.hud[:], dtype=np.uint8)
    if play.shape != (count, obs_spec.FRAME_STACK, *obs_spec.PLAY_SHAPE):
        raise RuntimeError(f"play shape {play.shape} != {(count, 4, 64, 64)}")
    if hud.shape != (count, *obs_spec.HUD_SHAPE):
        raise RuntimeError(f"hud shape {hud.shape} != {(count, 56, 160)}")
    scores = np.stack([np.asarray(record["scores"], dtype=np.float32) for record in records])
    mask = np.stack([np.asarray(record["verified_mask"], dtype=bool) for record in records])
    unverified = int((~mask).sum())
    restored_delta = 0.0
    if geo_restore:
        # float64 for the arithmetic, float32 back into the batch tensors: the
        # discount is 5.0 and the scores are ~1e2, so float32 is exact here.
        before = scores.astype(np.float64)
        after = geo_restore_scores(before, mask)
        restored_delta = float((after - before).sum())
        scores_raw = scores.copy()
        scores = after.astype(np.float32)
    else:
        scores_raw = scores.copy()
    bomb = np.asarray([int(record["bomb"]) for record in records], dtype=np.int8)
    verified = np.asarray([int(record["verified_count"]) for record in records], dtype=np.int16)
    return {"play": play, "hud": hud, "scores": scores, "bomb": bomb,
            "verified": verified, "count": count,
            "scores_raw": scores_raw,
            "unverified_candidates": unverified,
            "restored_delta": restored_delta}



def build_splits(
    d0_root: Path,
    manifest: dict | None,
    assignments: dict[str, str],
    *,
    min_samples: int,
    require_observations: bool = True,
    extra_roots: Sequence[str | Path] = (),
    shard_rows: list[dict] | None = None,
    geo_restore: bool = False,
) -> tuple[dict[str, SplitArrays], dict]:
    """Validate D0 through the frozen loader, then materialise the three splits.

    ``extra_roots`` are further D0-shaped roots (EXP-P3-003: the DAgger round-1
    shards) whose complete shards join the same split assignment.  Every root is
    loaded through the same frozen loader and every shard keeps its own episode
    identity, so "one shard = one episode = one seed" still holds across roots.
    """

    roots = [Path(d0_root), *[Path(root) for root in extra_roots]]
    if shard_rows is None:
        shard_rows = _complete_shards(Path(d0_root), manifest or {})
        for root in roots[1:]:
            shard_rows.extend(_complete_shards(root, load_teacher_manifest(root)))
    by_key = {row["shard"]: row for row in shard_rows}
    missing = sorted(set(assignments) - set(by_key))
    if missing:
        raise RuntimeError(f"assigned shard(s) are not complete/readable: {missing}")
    print(f"loading {len(assignments)} shard(s) from {len(roots)} root(s) through "
          "load_teacher_samples ...", flush=True)
    started = time.monotonic()
    samples: list[dict] = []
    loaded_reports: list[dict] = []
    for root in roots:
        keys = [key for key in sorted(assignments) if by_key[key]["root"] == root]
        if not keys:
            continue
        loaded = load_teacher_samples(
            [by_key[key]["dir"] / "labels.jsonl" for key in keys],
            min_samples=1,
            obs_root=root,
            require_observations=require_observations,
        )
        loaded_reports.append({
            "root": str(root),
            "shards": len(keys),
            "decisions": int(loaded["count"]),
            "obs_spec_version": loaded["obs_spec_version"],
            "obs_spec_digest": loaded["obs_spec_digest"],
            "unresolved_observations": len(loaded.get("obs_unresolved") or []),
        })
        samples.extend(loaded["samples"])
    load_seconds = time.monotonic() - started
    if len(samples) < int(min_samples):
        raise RuntimeError(
            f"the aggregate holds {len(samples)} teacher samples < required {min_samples}")
    if loaded_reports:
        obs_spec_version = loaded_reports[0]["obs_spec_version"]
        obs_spec_digest = loaded_reports[0]["obs_spec_digest"]
    else:  # pragma: no cover - the assignments check above already refused this
        raise RuntimeError("no root contributed a shard")
    print(f"loaded {len(samples)} teacher samples in {load_seconds:.1f}s "
          f"(obs spec {obs_spec_version} {obs_spec_digest[:8]}…)", flush=True)

    per_key: dict[str, dict] = {}
    for record in samples:
        observation = record.get("obs_arrays") or {}
        shard = observation.get("shard")
        if shard is None:
            raise RuntimeError("loader returned a sample without resolved observations")
        key = str(shard.meta.get("shard"))
        if key not in per_key:
            per_key[key] = _load_shard_arrays(shard, geo_restore=geo_restore)
        elif per_key[key]["count"] != len(shard.records):
            raise RuntimeError(f"shard {key} changed between loader reads")
    for key, row in by_key.items():
        if key not in per_key:
            raise RuntimeError(f"shard {key} is assigned but the loader returned no samples")
        if per_key[key]["count"] != row["decisions"]:
            raise RuntimeError(
                f"shard {key}: {per_key[key]['count']} loaded rows != shard.json "
                f"decision_count {row['decisions']}")

    splits: dict[str, SplitArrays] = {}
    for name in teacher_bc.SPLIT_NAMES:
        keys = sorted(key for key, split in assignments.items() if split == name)
        if not keys:
            raise RuntimeError(f"split {name} has no shards")
        play = np.concatenate([per_key[key]["play"] for key in keys], axis=0)
        hud = np.concatenate([per_key[key]["hud"] for key in keys], axis=0)
        scores = np.concatenate([per_key[key]["scores"] for key in keys], axis=0)
        scores_raw = np.concatenate([per_key[key]["scores_raw"] for key in keys], axis=0)
        bomb = np.concatenate([per_key[key]["bomb"] for key in keys], axis=0)
        verified = np.concatenate([per_key[key]["verified"] for key in keys], axis=0)
        shard_index = np.concatenate(
            [np.full(per_key[key]["count"], position, dtype=np.int16)
             for position, key in enumerate(keys)])
        decisions_by_shard = {key: int(per_key[key]["count"]) for key in keys}
        for key in keys:
            per_key[key]["play"] = None  # release the copy we just concatenated
        splits[name] = SplitArrays(
            name=name, play=play, hud=hud, scores=scores, bomb=bomb,
            verified_count=verified, shard_index=shard_index,
            shard_keys=keys, decisions_by_shard=decisions_by_shard,
            scores_raw=scores_raw)
    gate = {
        "loader": "touhou_rl.v400.student.dataset.load_teacher_samples",
        "require_observations": bool(require_observations),
        "min_samples_requested": int(min_samples),
        "d0_min_samples_contract": int(MIN_TEACHER_SAMPLES),
        "samples": int(len(samples)),
        "obs_spec_version": obs_spec_version,
        "obs_spec_digest": obs_spec_digest,
        "alignment": "decision_time_before_commit",
        "load_seconds": round(load_seconds, 3),
        "unresolved_observations": sum(row["unresolved_observations"]
                                       for row in loaded_reports),
        "shards_loaded": len(samples) and len(per_key),
        "roots": loaded_reports,
        "root_count": len(roots),
        "geo_restore": bool(geo_restore),
        "geo_restore_discount": float(UNVERIFIED_DISCOUNT),
        "geo_restore_unverified_values": int(sum(per_key[key]["unverified_candidates"]
                                                 for key in per_key)),
        "geo_restore_total_delta": float(sum(per_key[key]["restored_delta"]
                                             for key in per_key)),
    }
    return splits, gate


# ---------------------------------------------------------------------------
# tensors and losses
# ---------------------------------------------------------------------------
def batch_tensors(arrays: SplitArrays, indices: np.ndarray, device: str, torch):
    play = torch.from_numpy(arrays.play[indices]).to(device, non_blocking=False).float()
    play = play.div_(255.0)
    hud = torch.from_numpy(arrays.hud[indices]).to(device, non_blocking=False).float()
    hud = hud.div_(255.0).unsqueeze(1)
    scores = torch.from_numpy(arrays.scores[indices]).to(device, non_blocking=False)
    bomb = torch.from_numpy(arrays.bomb[indices].astype(np.int64)).to(device)
    return play, hud, scores, bomb


def soft_labels_torch(scores, tau: float, torch):
    """``softmax(scores / tau)`` as a torch tensor (teacher targets)."""

    return torch.softmax(scores / float(tau), dim=-1)


def bc_loss(move_logits, bomb_logits, scores, bomb, *, tau, pos_weight, bomb_weight, torch):
    """KL(pi_t || pi_s) + bomb_weight * BCE(one-hot bomb, pos_weight)."""

    functional = torch.nn.functional
    log_probs = functional.log_softmax(move_logits, dim=-1)
    teacher = soft_labels_torch(scores, tau, torch)
    # F.kl_div(input=log pi_s, target=pi_t) == sum pi_t (log pi_t - log pi_s)
    kl = functional.kl_div(log_probs, teacher, reduction="batchmean")
    targets = functional.one_hot(bomb, num_classes=obs_spec.N_BOMB_CLASSES).float()
    weight = torch.tensor([1.0, float(pos_weight)], device=bomb_logits.device)
    bce = functional.binary_cross_entropy_with_logits(
        bomb_logits, targets, pos_weight=weight)
    loss = kl + float(bomb_weight) * bce
    return loss, kl.detach(), bce.detach(), teacher


def evaluate(arrays: SplitArrays, model, indices: np.ndarray, *, tau: float, device: str,
             batch_size: int, torch) -> dict:
    """Offline metrics: KL, top-1/top-3 teacher agreement, macro recall, bomb."""

    model.eval()
    totals = {key: 0.0 for key in
              ("kl", "ce_argmax", "teacher_entropy", "top1", "top3", "bomb_correct",
               "bomb_pred_positive")}
    count = 0
    predicted_all: list[np.ndarray] = []
    teacher_all: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            play, hud, scores, bomb = batch_tensors(arrays, chunk, device, torch)
            move_logits, bomb_logits = model(play, hud)
            log_probs = torch.nn.functional.log_softmax(move_logits, dim=-1)
            teacher = soft_labels_torch(scores, tau, torch)
            safe = torch.where(teacher > 0, torch.log(torch.clamp(teacher, min=1e-30)),
                               torch.zeros_like(teacher))
            kl = (teacher * (safe - log_probs)).sum(dim=-1)
            teacher_argmax = teacher.argmax(dim=-1)
            top3 = move_logits.topk(3, dim=1).indices
            ce_argmax = -log_probs.gather(1, teacher_argmax[:, None]).squeeze(1)
            entropy = -(teacher * safe).sum(dim=-1)
            bomb_correct = bomb_logits.argmax(dim=1).eq(bomb)
            totals["kl"] += float(kl.sum())
            totals["ce_argmax"] += float(ce_argmax.sum())
            totals["teacher_entropy"] += float(entropy.sum())
            totals["top1"] += float(top3[:, 0].eq(teacher_argmax).sum())
            totals["top3"] += float(top3.eq(teacher_argmax[:, None]).any(dim=1).sum())
            totals["bomb_correct"] += float(bomb_correct.sum())
            totals["bomb_pred_positive"] += float(bomb_logits.argmax(dim=1).sum())
            predicted_all.append(top3[:, 0].detach().cpu().numpy())
            teacher_all.append(teacher_argmax.detach().cpu().numpy())
            count += len(chunk)
    if count == 0:
        raise RuntimeError("evaluated zero samples")
    predicted = np.concatenate(predicted_all)
    truth = np.concatenate(teacher_all)
    recall = macro_recall(predicted, truth)
    pred_hist = np.bincount(predicted, minlength=obs_spec.N_MOVE_CLASSES)
    truth_hist = np.bincount(truth, minlength=obs_spec.N_MOVE_CLASSES)
    # EXP-P3-006 §5: the collapse diagnostics the card reports.  ``cand = focus*9 +
    # move`` (gym/actions.py), so 0 is idle-without-focus, MOVES_PER_CAND (9) is
    # idle-with-focus and 10..17 are the eight focus directions ("focus 类").
    assert MOVES_PER_CAND * 2 == obs_spec.N_MOVE_CLASSES, (
        f"the action table changed: {MOVES_PER_CAND} moves x 2 focus states "
        f"!= {obs_spec.N_MOVE_CLASSES} candidates")
    focus_directions = list(range(MOVES_PER_CAND + 1, obs_spec.N_MOVE_CLASSES))
    return {
        "samples": int(count),
        "kl": totals["kl"] / count,
        "ce_argmax": totals["ce_argmax"] / count,
        "teacher_entropy": totals["teacher_entropy"] / count,
        "top1": totals["top1"] / count,
        "top3": totals["top3"] / count,
        "bomb_accuracy": totals["bomb_correct"] / count,
        "bomb_pred_positive_rate": totals["bomb_pred_positive"] / count,
        "bomb_positives": int(arrays.bomb[indices].sum()),
        "macro_recall": recall["macro_recall"],
        "macro_recall_classes_present": recall["classes_present"],
        "macro_recall_per_class": recall["per_class_recall"],
        "predicted_classes": recall["predicted_classes"],
        "majority_class_baseline": recall["majority_class_baseline"],
        "pred_histogram": [int(value) for value in pred_hist],
        "teacher_argmax_histogram": [int(value) for value in truth_hist],
        "pred_action0_share": float(pred_hist[0] / predicted.size),
        "pred_idle_share": float((pred_hist[0] + pred_hist[MOVES_PER_CAND])
                                 / predicted.size),
        "pred_unique_classes": int((pred_hist > 0).sum()),
        "pred_focus_classes_covered": int(sum(1 for value in focus_directions
                                              if pred_hist[value] > 0)),
        "teacher_action0_share": float(truth_hist[0] / truth.size),
    }


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    tau: float
    steps: int = 0
    samples: int = 0
    train_seconds: float = 0.0
    losses: list[float] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    best_metrics: dict | None = None
    best_state: dict | None = None
    final_metrics: dict | None = None
    aborted: bool = False

    def curve_summary(self) -> dict:
        if not self.history:
            return {}
        first = self.history[0]
        last = self.history[-1]
        best = max(self.history, key=lambda row: (row["top1"], -row["ce_argmax"]))
        return {
            "evaluations": len(self.history),
            "first": first,
            "last": last,
            "best": best,
            "train_loss_first": self.losses[0] if self.losses else None,
            "train_loss_last": self.losses[-1] if self.losses else None,
            "train_loss_best": min(self.losses) if self.losses else None,
        }


def train_run(
    *,
    name: str,
    tau: float,
    model,
    train_arrays: SplitArrays,
    val_arrays: SplitArrays,
    val_indices: np.ndarray,
    seconds: float,
    batch_size: int,
    lr: float,
    weight_decay: float,
    pos_weight: float,
    bomb_weight: float,
    eval_every: int,
    device: str,
    seed: int,
    torch,
    log_prefix: str = "",
) -> RunResult:
    """Time-budgeted training of one model; keeps the best-by-val state."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    result = RunResult(tau=tau)
    order = rng.permutation(len(train_arrays))
    cursor = 0
    deadline = time.monotonic() + float(seconds)
    step = 0
    while True:
        now = time.monotonic()
        if now >= deadline:
            break
        if cursor + batch_size > len(order):
            order = rng.permutation(len(train_arrays))
            cursor = 0
        indices = order[cursor:cursor + batch_size]
        cursor += batch_size
        play, hud, scores, bomb = batch_tensors(train_arrays, indices, device, torch)
        model.train()
        move_logits, bomb_logits = model(play, hud)
        loss, kl, bce, _ = bc_loss(
            move_logits, bomb_logits, scores, bomb, tau=tau, pos_weight=pos_weight,
            bomb_weight=bomb_weight, torch=torch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1
        result.steps = step
        result.samples += len(indices)
        result.losses.append(float(loss.item()))
        if step % eval_every == 0 or time.monotonic() >= deadline:
            metrics = evaluate(val_arrays, model, val_indices, tau=tau, device=device,
                               batch_size=max(batch_size, 512), torch=torch)
            metrics.update(step=step, samples=result.samples,
                           train_loss=float(np.mean(result.losses[-eval_every:])),
                           kl_loss=float(kl.item()), bce_loss=float(bce.item()))
            result.history.append(metrics)
            better = (result.best_metrics is None
                      or (metrics["top1"], -metrics["ce_argmax"])
                      > (result.best_metrics["top1"], -result.best_metrics["ce_argmax"]))
            if better:
                result.best_metrics = dict(metrics)
                result.best_state = {key: value.detach().cpu().clone()
                                     for key, value in model.state_dict().items()}
            elapsed = time.monotonic() - deadline + seconds
            print(f"{log_prefix}{name} step {step} seen {result.samples} "
                  f"loss {np.mean(result.losses[-eval_every:]):.4f} "
                  f"kl {kl.item():.4f} bce {bce.item():.4f} "
                  f"val_kl {metrics['kl']:.4f} val_top1 {metrics['top1']:.4f} "
                  f"ce_argmax {metrics['ce_argmax']:.4f} "
                  f"{result.samples / max(elapsed, 1e-6):.0f} samples/s", flush=True)
    result.train_seconds = float(seconds) - max(deadline - time.monotonic(), 0.0)
    result.aborted = time.monotonic() >= deadline
    result.final_metrics = evaluate(val_arrays, model, val_indices, tau=tau, device=device,
                                    batch_size=max(batch_size, 512), torch=torch)
    if result.best_state is not None:
        model.load_state_dict(result.best_state)
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d0-root", default=".tmp_v400_d0/d0")
    parser.add_argument("--d1-root", default=None,
                        help="T4: the obs-v3 / D1 corpus root for --obs v3 (falls back "
                             "to --d0-root when unset); unused by the v1/v2 arms")
    parser.add_argument("--extra-root", action="append", default=None, metavar="DIR",
                        help="further D0-shaped root whose complete shards join the same "
                             "aggregate and the same whole-episode split (EXP-P3-003: the "
                             "DAgger round-1 shards). Repeatable.")
    parser.add_argument("--checkpoint", default=".tmp_v400_p3v1/checkpoint-v1.pt")
    parser.add_argument("--metrics", default=".tmp_v400_p3v1/train-metrics.json")
    parser.add_argument("--split-json", default=".tmp_v400_p3v1/split.json")
    parser.add_argument("--taus", default="1.0,2.0,4.0")
    parser.add_argument("--sweep-seconds", type=float, default=110.0,
                        help="equal wall-clock probe budget per tau candidate")
    parser.add_argument("--train-seconds", type=float, default=420.0,
                        help="budget of the final run at the selected tau")
    parser.add_argument("--max-total-seconds", type=float, default=900.0,
                        help="hard budget for all training phases (13-P3 wants <= 15 min)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bomb-weight", type=float, default=1.0)
    parser.add_argument("--bomb-pos-weight-max", type=float, default=100.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--val-eval-samples", type=int, default=2048)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--accept-partial-d0", action="store_true",
                        help="train on a D0 below the 50k contract, recording the real count")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="optional step cap per run (0 = time budget only)")
    parser.add_argument("--geo-restore", action="store_true",
                        help="EXP-P3-004: undo 11-P1 §3's -5.0 discount on every "
                             "candidate the teacher did not verify "
                             "(scores + 5*(1-verified_mask)) before building "
                             "softmax(scores/tau).  Value-exact, but with verify_k=1 "
                             "it flattens 17 of 18 candidates per decision, so it is "
                             "an A/B arm, not a default")
    parser.add_argument("--obs-v2", action="store_true",
                        help="EXP-P3-008: train an obs-v2 arm (adds the 8x18 act_hist) "
                             "through touhou_rl.v400.student.obs_v2_train, on the same "
                             "corpus/split as the obs-v1 baseline")
    parser.add_argument("--obs-v2-root", default=".tmp_v400_obsv2/obsv2",
                        help="obs-v2 root built by "
                             "python -m touhou_rl.v400.student.obs_v2_dataset build")
    parser.add_argument("--arch", choices=("mlp", "gru", "v3"), default=None,
                        help="obs-v2 architecture: mlp (feed-forward, default) or gru "
                             "(single-layer GRU over 8 decisions, BPTT 8); 'v3' routes to "
                             "the obs-v3 IMPALA-ResNet student (T4)")
    parser.add_argument("--obs", choices=("v1", "v2", "v3"), default="v1",
                        help="observation contract: v1 (default, unchanged), v2 (adds the "
                             "8x18 act_hist, same as --obs-v2) or v3 (96x112 + act_hist, "
                             "the T4 student)")
    parser.add_argument("--accept-partial-d1", action="store_true",
                        help="obs-v3 arms: train on a D1 below the decision contract, "
                             "recording the real count as a waiver")
    parser.add_argument("--value-target", default="verified-share",
                        choices=("verified-share", "margin", "none"),
                        help="obs-v3 arms: proxy target of the auxiliary value head")
    parser.add_argument("--aux-weight", type=float, default=0.5,
                        help="obs-v3 arms: lambda_aux of the perception heads (13-P3 §2.4)")
    parser.add_argument("--value-weight", type=float, default=0.25,
                        help="obs-v3 arms: lambda_v of the value head (13-P3 §2.4)")
    parser.add_argument("--trunk", type=int, default=512,
                        help="obs-v3 arms: trunk width (13-P3 §2.3: 512)")
    parser.add_argument("--stage-channels", default="32,64,64",
                        help="obs-v3 arms: IMPALA stage widths (13-P3 §2.3: 32/64/64)")
    parser.add_argument("--gate-observations", choices=("shard", "aggregate"),
                        default="shard",
                        help="obs-v3 arms: per-shard loader gate (default) or also run "
                             "load_teacher_samples_v3(require_observations=True)")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--bptt", type=int, default=8)
    parser.add_argument("--hist-hidden", type=int, default=64)
    parser.add_argument("--gru-input", type=int, default=128)
    return parser


def _obs_v2_argv(args) -> list[str]:
    """Translate this parser's namespace into ``obs_v2_train``'s CLI.

    The obs-v2 arms are trained by :mod:`touhou_rl.v400.student.obs_v2_train` so the
    obs-v1 code path above stays byte-for-byte the code that produced the 1,816.0 / 
    1,718.8 rows.  Only the flags that exist in both parsers are forwarded, and the
    tau sweep is refused: EXP-P3-008 pre-registers tau=2.0.
    """

    taus = [part for part in str(args.taus).split(",") if part.strip()]
    if len(taus) != 1:
        raise SystemExit(
            f"obs-v2 takes exactly one tau via --taus (got {args.taus!r}); "
            "EXP-P3-008 pre-registers tau=2.0 so no sweep is allowed")
    argv = [
        "--d0-root", str(args.d0_root),
        "--obsv2-root", str(args.obs_v2_root),
        "--arch", str(args.arch or "mlp"),
        "--checkpoint", str(args.checkpoint),
        "--metrics", str(args.metrics),
        "--tau", taus[0],
        "--train-seconds", str(args.train_seconds),
        "--max-total-seconds", str(args.max_total_seconds),
        "--batch-size", str(args.batch_size),
        "--eval-batch", str(args.eval_batch),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--bomb-weight", str(args.bomb_weight),
        "--bomb-pos-weight-max", str(args.bomb_pos_weight_max),
        "--eval-every", str(args.eval_every),
        "--val-eval-samples", str(args.val_eval_samples),
        "--width", str(args.width),
        "--hidden", str(args.hidden),
        "--hist-hidden", str(args.hist_hidden),
        "--gru-input", str(args.gru_input),
        "--seq-len", str(args.seq_len),
        "--bptt", str(args.bptt),
        "--split-seed", str(args.split_seed),
        "--seed", str(args.seed),
        "--split-json", str(args.split_json),
    ]
    for root in args.extra_root or []:
        argv.extend(["--extra-root", str(root)])
    if args.device:
        argv.extend(["--device", str(args.device)])
    return argv


def route_obs(args) -> str:
    """Which trainer this invocation routes to: ``"v3"``, ``"v2"`` or ``"v1"``.

    T4 adds obs-v3 without touching either existing path: ``--obs v3`` or
    ``--arch v3`` selects :mod:`touhou_rl.v400.student.train_v3`; every other
    combination keeps the pre-T4 behaviour (``--obs-v2``/``--arch mlp|gru`` ->
    ``obs_v2_train``, nothing set -> the obs-v1 loop in :func:`main`).
    """

    obs = str(getattr(args, "obs", "v1") or "v1")
    arch = getattr(args, "arch", None)
    if obs == "v3" or arch == "v3":
        return "v3"
    if getattr(args, "obs_v2", False) or obs == "v2" or arch is not None:
        return "v2"
    return "v1"


def _train_v3_argv(args) -> list[str]:
    """Translate this parser's namespace into ``train_v3``'s CLI.

    obs-v3 is trained by :mod:`touhou_rl.v400.student.train_v3` (the obs-v3 corpus
    is mem-mapped, which the obs-v1/v2 loaders do not do), so only the shared flags
    plus the v3-specific ones this parser exposes are forwarded.  ``--width`` /
    ``--hidden`` / ``--hist-hidden`` / ``--gru-input`` describe the obs-v1/v2
    networks and are not forwarded: the v3 width is ``--trunk`` (512 by default) and
    the HUD/history widths have their own defaults.
    """

    taus = [part for part in str(args.taus).split(",") if part.strip()]
    if len(taus) != 1:
        raise SystemExit(
            f"obs-v3 takes exactly one tau via --taus (got {args.taus!r}); T4 keeps "
            "EXP-P3-008's pre-registered tau=2.0 so the row stays comparable")
    d1_root = str(getattr(args, "d1_root", None) or args.d0_root)
    argv = [
        "--d1-root", d1_root,
        "--checkpoint", str(args.checkpoint),
        "--metrics", str(args.metrics),
        "--split-json", str(args.split_json),
        "--tau", taus[0],
        "--train-seconds", str(args.train_seconds),
        "--max-total-seconds", str(args.max_total_seconds),
        "--max-steps", str(args.max_steps),
        "--batch-size", str(args.batch_size),
        "--eval-batch", str(args.eval_batch),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--bomb-weight", str(args.bomb_weight),
        "--bomb-pos-weight-max", str(args.bomb_pos_weight_max),
        "--aux-weight", str(args.aux_weight),
        "--value-weight", str(args.value_weight),
        "--value-target", str(args.value_target),
        "--trunk", str(args.trunk),
        "--stage-channels", str(args.stage_channels),
        "--eval-every", str(args.eval_every),
        "--val-eval-samples", str(args.val_eval_samples),
        "--split-seed", str(args.split_seed),
        "--seed", str(args.seed),
        "--gate-observations", str(args.gate_observations),
    ]
    for root in args.extra_root or []:
        argv.extend(["--extra-root", str(root)])
    if args.accept_partial_d1:
        argv.append("--accept-partial-d1")
    if args.device:
        argv.extend(["--device", str(args.device)])
    return argv


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    route = route_obs(args)
    if route == "v3":
        from touhou_rl.v400.student import train_v3

        return train_v3.main(_train_v3_argv(args))
    if route == "v2":
        from touhou_rl.v400.student import obs_v2_train

        return obs_v2_train.main(_obs_v2_argv(args))
    torch = _torch()
    started = time.monotonic()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    d0_root = Path(args.d0_root)
    manifest_path = d0_root / "manifest.json"
    manifest = load_teacher_manifest(d0_root)
    shard_rows = _complete_shards(d0_root, manifest)
    extra_roots = [Path(root) for root in (args.extra_root or [])]
    for root in extra_roots:
        if root.resolve() == d0_root.resolve():
            raise SystemExit(f"--extra-root {root} is the --d0-root; list it once")
        extra_rows = _complete_shards(root, load_teacher_manifest(root))
        print(f"aggregate: {len(extra_rows)} complete shard(s) under {root}", flush=True)
        shard_rows.extend(extra_rows)
    if not shard_rows:
        raise SystemExit(f"no complete D0 shard under {d0_root}")
    shard_keys = [row["shard"] for row in shard_rows]
    if len(set(shard_keys)) != len(shard_keys):
        raise SystemExit(f"duplicate shard key(s) across roots: {sorted(shard_keys)}")
    decisions = {row["shard"]: row["decisions"] for row in shard_rows}
    seeds = {row["shard"]: row["seed"] for row in shard_rows}
    total_decisions = sum(decisions.values())

    d0_gate_required = int(MIN_TEACHER_SAMPLES)
    waived = total_decisions < d0_gate_required
    if waived and not args.accept_partial_d0:
        raise SystemExit(
            f"D0 has {total_decisions} decisions < the {d0_gate_required} contract; "
            "pass --accept-partial-d0 to train on the real count (recorded as a waiver)")
    min_samples = total_decisions if waived else d0_gate_required

    assignments = teacher_bc.assign_shard_splits(
        shard_keys, ratios=teacher_bc.SPLIT_RATIOS, seed=args.split_seed)
    split_report = teacher_bc.split_report(assignments, decisions=decisions, seeds=seeds)
    split_report["split_seed"] = args.split_seed
    split_report["signature"] = teacher_bc.split_signature(assignments)
    print(json.dumps({name: {"shards": split_report[name]["shards"],
                             "decisions": split_report[name]["decisions"]}
                      for name in teacher_bc.SPLIT_NAMES}, indent=1), flush=True)

    splits, gate = build_splits(d0_root, manifest, assignments, min_samples=min_samples,
                                extra_roots=extra_roots, shard_rows=shard_rows,
                                geo_restore=bool(args.geo_restore))
    aggregate_by_root: dict[str, int] = {str(root.resolve()): 0 for root in [d0_root, *extra_roots]}
    for row in shard_rows:
        aggregate_by_root[str(Path(row["root"]).resolve())] += int(row["decisions"])
    gate.update({
        "d0_root": str(d0_root.resolve()),
        "extra_roots": [str(root.resolve()) for root in extra_roots],
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest_updated_at": (manifest.get("totals") or {}).get("updated_at"),
        "manifest_totals": manifest.get("totals"),
        "complete_shards": len(shard_rows),
        "decisions_in_complete_shards": total_decisions,
        "required_decisions": d0_gate_required,
        "gate_waived": bool(waived),
        "gate_waiver_reason": (
            f"the aggregate holds {total_decisions} decisions < {d0_gate_required} when "
            "training started (collection still running); the real count is recorded and "
            "no 50k is claimed" if waived else None),
        "aggregate_by_root": aggregate_by_root,
    })

    train_arrays = splits["train"]
    val_arrays = splits["val"]
    test_arrays = splits["test"]
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

    def _gap_stats(scores: np.ndarray) -> dict:
        ordered = np.sort(scores, axis=-1)[:, ::-1]
        gap = ordered[:, 0] - ordered[:, 1]
        return {
            "mean": float(gap.mean()), "median": float(np.median(gap)),
            "p10": float(np.percentile(gap, 10)), "max": float(gap.max()),
            # the EXP-P3-003 §8.3 statistic: how often the label's top-1 lead is
            # exactly the 5.0 discount (i.e. "no real preference, only the mask")
            "fraction_equal_to_discount": float(np.mean(np.isclose(gap, UNVERIFIED_DISCOUNT))),
            "fraction_above_discount": float(np.mean(gap > UNVERIFIED_DISCOUNT + 1e-9)),
        }

    gap_after = _gap_stats(train_arrays.scores)
    gap_before = (_gap_stats(train_arrays.scores_raw)
                  if train_arrays.scores_raw is not None
                  and not np.array_equal(train_arrays.scores_raw, train_arrays.scores)
                  else gap_after)
    label_stats = {
        "bomb": bomb_stats,
        "top1_minus_top2": gap_after,
        "top1_minus_top2_on_disk": gap_before,
        "geo_restore": bool(args.geo_restore),
        "teacher_argmax_histogram": np.bincount(
            train_arrays.scores.argmax(axis=-1), minlength=obs_spec.N_MOVE_CLASSES).tolist(),
        "verified_mask_mean": float(train_arrays.verified_count.mean()),
        "verified_mask_used_in_loss": False,
        "scores_min": float(train_arrays.scores.min()),
        "scores_max": float(train_arrays.scores.max()),
    }

    taus = [float(part) for part in args.taus.split(",") if part.strip()]
    for tau in taus:
        if not np.isfinite(tau) or tau <= 0:
            raise SystemExit(f"invalid tau {tau}")

    hard_deadline = started + args.max_total_seconds
    probe_results: dict[float, RunResult] = {}
    for tau in taus:
        remaining = hard_deadline - time.monotonic() - 5.0
        budget = max(5.0, min(args.sweep_seconds, remaining))
        if remaining <= 10.0:
            print(f"skip tau probe {tau}: out of total budget", flush=True)
            break
        torch.manual_seed(args.seed)
        model = StudentNet(width=args.width, hidden=args.hidden).to(device)
        probe_results[tau] = train_run(
            name=f"probe(tau={tau:g})", tau=tau, model=model,
            train_arrays=train_arrays, val_arrays=val_arrays, val_indices=val_indices,
            seconds=budget, batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, pos_weight=bomb_stats["pos_weight"],
            bomb_weight=args.bomb_weight, eval_every=args.eval_every, device=device,
            seed=args.seed + int(round(tau * 100)), torch=torch, log_prefix="[sweep] ")
    if not probe_results:
        raise SystemExit("no tau probe ran; increase --max-total-seconds")

    candidate_metrics = {
        tau: {
            "val_top1": result.best_metrics["top1"],
            "val_kl": result.best_metrics["kl"],
            "val_ce_argmax": result.best_metrics["ce_argmax"],
            "val_bomb_acc": result.best_metrics["bomb_accuracy"],
        }
        for tau, result in probe_results.items()
    }
    selected_tau, tau_rationale = teacher_bc.select_tau(candidate_metrics)

    remaining = hard_deadline - time.monotonic() - 10.0
    final_budget = max(10.0, min(args.train_seconds, remaining))
    torch.manual_seed(args.seed)
    final_model = StudentNet(width=args.width, hidden=args.hidden).to(device)
    final_run = train_run(
        name=f"final(tau={selected_tau:g})", tau=selected_tau, model=final_model,
        train_arrays=train_arrays, val_arrays=val_arrays, val_indices=val_indices,
        seconds=final_budget, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, pos_weight=bomb_stats["pos_weight"],
        bomb_weight=args.bomb_weight, eval_every=args.eval_every, device=device,
        seed=args.seed + 1, torch=torch, log_prefix="[final] ")

    metrics = {
        "train": evaluate(train_arrays, final_model, train_eval_indices,
                          tau=selected_tau, device=device, batch_size=args.eval_batch,
                          torch=torch),
        "val": evaluate(val_arrays, final_model, np.arange(len(val_arrays)),
                        tau=selected_tau, device=device, batch_size=args.eval_batch,
                        torch=torch),
        "test": evaluate(test_arrays, final_model, np.arange(len(test_arrays)),
                         tau=selected_tau, device=device, batch_size=args.eval_batch,
                         torch=torch),
    }
    parameters = final_model.n_parameters()
    probe_summary = {
        f"{tau:g}": {
            "best": probe_results[tau].best_metrics,
            "final": probe_results[tau].final_metrics,
            "curve": probe_results[tau].curve_summary(),
            "steps": probe_results[tau].steps,
            "samples": probe_results[tau].samples,
            "train_seconds": round(probe_results[tau].train_seconds, 3),
        }
        for tau in probe_results
    }
    train_info = {
        "bc_version": teacher_bc.BC_VERSION,
        "disclosure": teacher_bc.TEACHER_BC_DISCLOSURE,
        "tau": float(selected_tau),
        "d0": gate,
        "splits": split_report,
        "metrics": metrics,
        "tau_selection": {
            "eta": None,
            "candidates": tau_rationale["candidates"],
            "ranking": tau_rationale["ranking"],
            "rule": tau_rationale["rule"],
            "kl_not_used_for_ranking": tau_rationale["kl_not_used_for_ranking"],
            "probes": probe_summary,
        },
        "loss": (
            "KL(softmax((scores + 5*(1-verified_mask))/tau) || pi_s) + bomb_weight * BCE"
            if args.geo_restore else
            "KL(softmax(scores/tau) || pi_s) + bomb_weight * BCE(one-hot bomb, pos_weight)"),
        "geo_restore": {
            "enabled": bool(args.geo_restore),
            "rule": "scores + discount * (1 - verified_mask)",
            "discount": float(UNVERIFIED_DISCOUNT),
            "discount_source": "touhou_rl.v400.teacher.config.Weights.unverified_discount",
            "unverified_values_restored": gate.get("geo_restore_unverified_values", 0),
            "total_delta": gate.get("geo_restore_total_delta", 0.0),
            "exact_inverse": True,
        },
        "metric_definitions": {
            "macro_recall": "mean over classes present in the teacher argmax of "
                            "|pred==c and teacher==c| / |teacher==c|",
            "top1": "argmax(pi_s) == argmax(pi_t) with pi_t = softmax(scores/tau)",
            "kl": "sum_a pi_t(a) (log pi_t(a) - log pi_s(a)), mean over decisions",
        },
        "loss_weights": {"kl": 1.0, "bomb": float(args.bomb_weight),
                         "bomb_pos_weight": float(bomb_stats["pos_weight"])},
        "optimizer": f"AdamW(lr={args.lr}, weight_decay={args.weight_decay})",
        "batch_size": args.batch_size,
        "steps": final_run.steps,
        "samples_seen": final_run.samples,
        "train_seconds": round(final_run.train_seconds, 3),
        "wall_seconds": round(time.monotonic() - started, 3),
        "parameters": parameters,
        "model": {"width": args.width, "hidden": args.hidden},
        "device": device,
        "device_name": (torch.cuda.get_device_name(0) if device == "cuda" else "cpu"),
        "torch": torch.__version__,
        "label_stats": label_stats,
        "curve": final_run.curve_summary(),
        "best_val_at_step": final_run.best_metrics.get("step") if final_run.best_metrics else None,
        "aborted_at_deadline": bool(final_run.aborted),
        "obs_spec_version": obs_spec.OBS_SPEC_VERSION,
        "obs_spec_digest": obs_spec.obs_spec_digest(),
        "not_a_g3_result": True,
        "not_the_confirm_split": True,
    }
    checkpoint_path = Path(args.checkpoint)
    save_checkpoint(
        checkpoint_path,
        final_model.cpu(),
        train=train_info,
        provenance={
            "kind": teacher_bc.BC_VERSION,
            "d0_root": str(d0_root),
            "d0_manifest_sha256": gate["manifest_sha256"],
            "split_signature": split_report["signature"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "controller": "tools/rl/v400_student_train_teacher.py",
            "disclosure": teacher_bc.TEACHER_BC_DISCLOSURE,
        },
    )
    summary = teacher_bc.checkpoint_summary(torch.load(checkpoint_path,
                                                      map_location="cpu",
                                                      weights_only=False))
    train_info["checkpoint"] = str(checkpoint_path)
    train_info["checkpoint_sha256"] = checkpoint_digest(checkpoint_path)
    train_info["checkpoint_verified"] = {
        "parameters": summary["parameters"],
        "obs_spec_digest": summary["obs_spec_digest"],
        "tau": summary["tau"],
        "schema": summary["schema"],
    }
    split_payload = {
        "schema": "v400-teacher-bc-split-1",
        "split_seed": args.split_seed,
        "signature": split_report["signature"],
        "assignments": dict(sorted(assignments.items())),
        "report": split_report,
        "d0": gate,
    }
    for path, payload in ((Path(args.metrics), train_info), (Path(args.split_json), split_payload)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": train_info["checkpoint_sha256"],
        "parameters": parameters,
        "tau": selected_tau,
        "geo_restore": bool(args.geo_restore),
        "tau_candidates": candidate_metrics,
        "split_decisions": {name: split_report[name]["decisions"]
                            for name in teacher_bc.SPLIT_NAMES},
        "train_seconds": train_info["train_seconds"],
        "wall_seconds": train_info["wall_seconds"],
        "train": metrics["train"],
        "val": metrics["val"],
        "test": metrics["test"],
        "bomb": bomb_stats,
        "d0_waived": bool(waived),
        "disclosure": teacher_bc.TEACHER_BC_DISCLOSURE,
    }, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
