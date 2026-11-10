"""Teacher soft-label behaviour cloning for P3 (**BC v1**, EXP-P3-002).

The P3 teacher interface is fixed by 13-P3 §2.4::

    pi_t(a) = softmax(scores[18] / tau)
    loss    = KL(pi_t || pi_s) + BCE(bomb)          # bomb weighted by pos_weight

``verified_mask`` is a *trust / diagnostic* signal, never a loss weight: the
teacher already discounts unverified candidates when it writes ``scores``
(11-P1 §3), so re-weighting here would double-count that discount and make the
target depend on a mask the student never sees at deployment.

This module is deliberately **torch-free**: splitting, soft labels, KL, bomb
weighting, tau selection and checkpoint summary verification are pure
numpy/JSON functions so the contract is testable without a GPU.  The training
loop lives in ``tools/rl/v400_student_train_teacher.py``.

Episode isolation is the hard rule of the split: one D0 shard is exactly one
episode of one seed (``dual-s0d1seed<k>``), so a shard-level split cannot leak a
trajectory across train/val/test.  :func:`check_episode_isolation` re-derives
that from the shard -> seed mapping instead of trusting the caller.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from touhou_rl.v400.student import obs_spec

#: Split names in allocation order; the ratios are a *declared* 70/15/15.
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
SPLIT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)

#: Teacher soft-label temperatures the val sweep is allowed to choose from.
TAU_CANDIDATES: tuple[float, ...] = (1.0, 2.0, 4.0)

#: Checkpoint schema this module can summarise.  Duplicated from
#: ``touhou_rl.v400.student.model`` on purpose (the same deliberate duplication
#: ``dataset.py`` uses for the D0 schemas): a reader of a checkpoint must fail
#: when the writer's contract changes, not silently follow it.
CHECKPOINT_SCHEMA: str = "v400-student-checkpoint-1"

#: Provenance tag written into every BC-v1 artefact.
BC_VERSION: str = "bc-v1-teacher-soft"

#: Mandatory disclosure: BC v1 is an offline student baseline, not a gate pass.
TEACHER_BC_DISCLOSURE = (
    "BC v1 = behaviour cloning of the P3 teacher's soft labels "
    "(pi_t = softmax(scores/τ)) on stage 0 (= true Stage 1). It is an offline "
    "student baseline: its S1N episodes are not a G3 pass, not PSN, and not the "
    "confirm split."
)


class SplitError(ValueError):
    """The requested shard/episode split would leak or cannot be formed."""


# ---------------------------------------------------------------------------
# shard-level split (one shard = one episode = one seed)
# ---------------------------------------------------------------------------
def allocate_split_counts(n: int, ratios: Sequence[float] = SPLIT_RATIOS) -> tuple[int, ...]:
    """Largest-remainder allocation of ``n`` shards over ``ratios``, each >= 1."""

    n = int(n)
    if n < len(ratios):
        raise SplitError(
            f"{n} shard(s) cannot fill {len(ratios)} non-empty splits; "
            "collect more episodes or declare fewer splits")
    if any(ratio <= 0 for ratio in ratios):
        raise SplitError(f"split ratios must be positive: {tuple(ratios)}")
    total_ratio = float(sum(ratios))
    raw = [float(ratio) / total_ratio * n for ratio in ratios]
    counts = [int(math.floor(value)) for value in raw]
    remainder = n - sum(counts)
    order = sorted(range(len(ratios)), key=lambda i: (-(raw[i] - counts[i]), i))
    for position in range(remainder):
        counts[order[position % len(order)]] += 1
    for index in range(len(counts)):
        while counts[index] < 1:
            donor = max(range(len(counts)), key=lambda j: (counts[j], -j))
            if counts[donor] <= 1:
                raise SplitError(f"cannot give every split >= 1 shard from {n} shards")
            counts[donor] -= 1
            counts[index] += 1
    if sum(counts) != n:
        raise SplitError(f"allocation {counts} does not sum to {n}")
    return tuple(counts)


def assign_shard_splits(
    shard_keys: Iterable[str],
    *,
    ratios: Sequence[float] = SPLIT_RATIOS,
    seed: int = 0,
) -> dict[str, str]:
    """Deterministically assign whole shards to ``train``/``val``/``test``.

    The permutation is seeded and the keys are sorted first, so the same shard
    set always produces the same split (a checkpoint's split can be re-derived
    from its recorded shard list, which is what the tests check).
    """

    keys = list(shard_keys)
    if not keys:
        raise SplitError("no shards to split")
    if any(not isinstance(key, str) or not key for key in keys):
        raise SplitError(f"shard keys must be non-empty strings: {keys[:3]!r}")
    if len(set(keys)) != len(keys):
        duplicated = sorted({key for key in keys if keys.count(key) > 1})
        raise SplitError(f"duplicate shard key(s) in the dataset: {duplicated[:3]}")
    counts = allocate_split_counts(len(keys), ratios)
    permuted = sorted(keys)
    random.Random(int(seed)).shuffle(permuted)
    assignments: dict[str, str] = {}
    cursor = 0
    for name, count in zip(SPLIT_NAMES, counts):
        for key in permuted[cursor:cursor + count]:
            assignments[key] = name
        cursor += count
    return assignments


def check_episode_isolation(
    assignments: Mapping[str, str], seeds: Mapping[str, int]
) -> dict[str, int]:
    """Re-derive the shard -> seed mapping and refuse a seed spanning two splits."""

    unknown = [key for key in assignments if key not in seeds]
    if unknown:
        raise SplitError(f"shard(s) without a seed: {sorted(unknown)[:3]}")
    seed_to_split: dict[int, str] = {}
    duplicates: dict[int, list[str]] = {}
    for key, split in assignments.items():
        if split not in SPLIT_NAMES:
            raise SplitError(f"shard {key} has unknown split {split!r}")
        seed = int(seeds[key])
        previous = seed_to_split.get(seed)
        if previous is not None and previous != split:
            raise SplitError(
                f"seed {seed} appears in both {previous!r} and {split!r} "
                f"(shard {key}); one episode may not cross splits")
        if previous is not None:
            duplicates.setdefault(seed, []).append(key)
        seed_to_split[seed] = split
    return {str(seed): split for seed, split in sorted(seed_to_split.items())}


def split_report(
    assignments: Mapping[str, str],
    *,
    decisions: Mapping[str, int],
    seeds: Mapping[str, int],
) -> dict[str, Any]:
    """Per-split shard/seed/decision counts plus the isolation statement."""

    check_episode_isolation(assignments, seeds)
    missing = [key for key in assignments if key not in decisions]
    if missing:
        raise SplitError(f"shard(s) without a decision count: {sorted(missing)[:3]}")
    report: dict[str, Any] = {"ratios": list(SPLIT_RATIOS), "split_seed": None}
    total = sum(int(decisions[key]) for key in assignments)
    if total <= 0:
        raise SplitError("dataset has no decisions")
    for name in SPLIT_NAMES:
        keys = sorted(key for key, split in assignments.items() if split == name)
        if not keys:
            raise SplitError(f"split {name!r} is empty")
        split_decisions = sum(int(decisions[key]) for key in keys)
        report[name] = {
            "shards": keys,
            "shard_count": len(keys),
            "seeds": sorted(int(seeds[key]) for key in keys),
            "decisions": split_decisions,
            "fraction": split_decisions / total,
        }
    report["total_decisions"] = total
    report["isolation"] = (
        "one D0 shard = one episode = one seed; no shard and no seed crosses a split")
    return report


def split_signature(assignments: Mapping[str, str]) -> str:
    """Stable one-line signature of a split, recorded in checkpoints/metrics."""

    body = ";".join(f"{key}={assignments[key]}" for key in sorted(assignments))
    import hashlib

    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# teacher soft labels and the KL loss
# ---------------------------------------------------------------------------
def softmax_scores(scores: np.ndarray, tau: float, *, axis: int = -1) -> np.ndarray:
    """``softmax(scores / tau)`` with a max shift for numerical stability."""

    array = np.asarray(scores, dtype=np.float64)
    if array.shape[-1] != obs_spec.N_MOVE_CLASSES:
        raise ValueError(
            f"scores must end in {obs_spec.N_MOVE_CLASSES} candidates, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("scores contain non-finite values")
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError(f"tau must be finite and positive, got {tau!r}")
    shifted = (array - array.max(axis=axis, keepdims=True)) / tau
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=axis, keepdims=True)


def teacher_soft_labels(
    scores: np.ndarray,
    tau: float,
    *,
    verified_mask: np.ndarray | None = None,
) -> np.ndarray:
    """``pi_t = softmax(scores / tau)`` — **verified_mask is not applied**.

    ``verified_mask`` is accepted only so a caller can pass it explicitly and be
    told that it is diagnostic: applying it as a weight would re-introduce the
    teacher's internal trust discount a second time (13-P3 §2.4).
    """

    if verified_mask is not None:
        mask = np.asarray(verified_mask, dtype=bool)
        if mask.shape != np.asarray(scores).shape:
            raise ValueError(
                f"verified_mask shape {mask.shape} != scores shape "
                f"{np.asarray(scores).shape}; the mask is per-candidate diagnostics")
    return softmax_scores(scores, tau)


def kl_teacher_student(
    teacher_probs: np.ndarray, student_log_probs: np.ndarray
) -> np.ndarray:
    """Per-row ``KL(pi_t || pi_s)`` in nats.

    ``KL(pi_t || pi_s)`` is the direction 13-P3 §2.4 fixes: the teacher's mass
    covers the student, so the student is pulled toward *every* teacher-preferred
    plan rather than collapsing onto the single argmax.
    """

    teacher = np.asarray(teacher_probs, dtype=np.float64)
    student = np.asarray(student_log_probs, dtype=np.float64)
    if teacher.shape != student.shape:
        raise ValueError(f"teacher {teacher.shape} != student {student.shape}")
    if teacher.ndim < 1 or teacher.shape[-1] != obs_spec.N_MOVE_CLASSES:
        raise ValueError(f"last axis must be {obs_spec.N_MOVE_CLASSES}: {teacher.shape}")
    if np.any(teacher < 0):
        raise ValueError("teacher probabilities must be non-negative")
    if not np.allclose(teacher.sum(axis=-1), 1.0, atol=1e-6):
        raise ValueError("teacher probabilities must sum to 1")
    if np.any(student > 1e-9):
        raise ValueError("student_log_probs must be log-probabilities (<= 0)")
    safe = np.where(teacher > 0, np.log(np.where(teacher > 0, teacher, 1.0)), 0.0)
    return (teacher * (safe - student)).sum(axis=-1)


def bomb_pos_weight(
    labels: np.ndarray, *, max_weight: float = 100.0
) -> dict[str, Any]:
    """``pos_weight = negatives / positives`` for the rare bomb-positive class.

    A D0 slice can legitimately contain **zero** bomb positives (the teacher
    clears Stage 1 without dying, so it never needs a bomb).  Then no finite
    pos_weight exists: the weight degenerates to 1.0 and the head is reported as
    *not identifiable* instead of being silently trained into a vacuous 100 %
    accuracy.
    """

    array = np.asarray(labels).astype(np.int64).ravel()
    if array.size == 0:
        raise ValueError("no bomb labels")
    if np.any((array != 0) & (array != 1)):
        raise ValueError("bomb labels must be 0/1")
    positives = int((array == 1).sum())
    negatives = int((array == 0).sum())
    identifiable = positives > 0 and negatives > 0
    weight = min(float(max_weight), negatives / positives) if identifiable else 1.0
    return {
        "samples": int(array.size),
        "positives": positives,
        "negatives": negatives,
        "positives_fraction": positives / array.size,
        "pos_weight": weight,
        "max_weight": float(max_weight),
        "identifiable": bool(identifiable),
        "note": (
            "no positive example exists in D0: the bomb head can only learn p(bomb)=0 "
            "and its accuracy is vacuous" if not identifiable else
            "pos_weight = negatives/positives, clamped to max_weight"),
    }


# ---------------------------------------------------------------------------
# tau selection on val
# ---------------------------------------------------------------------------
_TAU_METRIC_KEYS = ("val_top1", "val_kl", "val_ce_argmax", "val_bomb_acc")


def select_tau(candidates: Mapping[float, Mapping[str, float]]) -> tuple[float, dict[str, Any]]:
    """Pick τ from val metrics; returns ``(tau, rationale)``.

    Ranking rule: **max val teacher top-1 agreement**, tie-broken by
    **min val NLL of the teacher's argmax action**, then by the smaller τ.

    Val ``KL(pi_t || pi_s)`` is *reported but not used to rank*: the target
    entropy ``H(pi_t)`` grows with τ, so the same student scores a smaller
    absolute KL at a larger τ.  Ranking by KL would therefore pick τ=4 by
    construction rather than by transfer quality.  Top-1 agreement and the NLL
    of a fixed (τ-invariant) action are comparable across τ.
    """

    if not candidates:
        raise ValueError("no tau candidate was evaluated")
    rows = []
    for tau, metrics in candidates.items():
        tau = float(tau)
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError(f"tau must be positive: {tau!r}")
        missing = [key for key in _TAU_METRIC_KEYS if key not in metrics]
        if missing:
            raise ValueError(f"tau {tau} metrics missing {missing}")
        rows.append({
            "tau": tau,
            "val_top1": float(metrics["val_top1"]),
            "val_kl": float(metrics["val_kl"]),
            "val_ce_argmax": float(metrics["val_ce_argmax"]),
            "val_bomb_acc": float(metrics["val_bomb_acc"]),
        })
    ranking = sorted(rows, key=lambda row: (-row["val_top1"], row["val_ce_argmax"], row["tau"]))
    best = ranking[0]
    rationale = {
        "selected_tau": best["tau"],
        "rule": "max val teacher top-1 agreement; tiebreak min val NLL of the teacher "
                "argmax action; then the smaller tau",
        "kl_not_used_for_ranking": (
            "val KL is reported but not ranked: the target entropy H(pi_t) grows with "
            "tau, so absolute KL is not comparable across tau"),
        "ranking": ranking,
        "candidates": sorted(rows, key=lambda row: row["tau"]),
    }
    return best["tau"], rationale


# ---------------------------------------------------------------------------
# checkpoint summary verification
# ---------------------------------------------------------------------------
_REQUIRED_TRAIN_KEYS = ("bc_version", "tau", "d0", "splits", "metrics")

#: ``state_dict`` entries that are BatchNorm buffers, not parameters.  The
#: checkpoint's ``parameters`` field is ``model.n_parameters()`` (torch counts
#: only ``nn.Parameter``), so a summary that summed every state_dict tensor would
#: over-count by the running stats and refuse every honest checkpoint.
_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def _state_dict_tensors(state_dict: Mapping[str, Any]) -> int:
    total = 0
    for value in state_dict.values():
        numel = getattr(value, "numel", None)
        if callable(numel):
            total += int(numel())
        else:
            total += int(np.asarray(value).size)
    return total


def _state_dict_parameters(state_dict: Mapping[str, Any]) -> int:
    total = 0
    for key, value in state_dict.items():
        if any(key.endswith(suffix) for suffix in _BUFFER_SUFFIXES):
            continue
        numel = getattr(value, "numel", None)
        if callable(numel):
            total += int(numel())
        else:
            total += int(np.asarray(value).size)
    return total


def checkpoint_summary(
    payload: Mapping[str, Any], *, expected_parameters: int | None = None
) -> dict[str, Any]:
    """Verify a BC checkpoint's schema, obs-spec digest and parameter count.

    Fails closed on: another checkpoint schema, another obs spec version or
    digest (``ObsSpecMismatch``), a ``parameters`` count that does not match the
    summed ``state_dict``, a train block missing the BC-v1 provenance keys, and
    a caller-supplied ``expected_parameters`` that disagrees.
    """

    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint payload is {type(payload).__name__}, not a mapping")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"checkpoint schema {payload.get('schema')!r} != {CHECKPOINT_SCHEMA!r}")
    obs_spec.check_obs_spec(payload.get("obs_spec_version"), payload.get("obs_spec_digest"))
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint has no state_dict")
    counted = _state_dict_parameters(state_dict)
    tensors = _state_dict_tensors(state_dict)
    declared = payload.get("parameters")
    if not isinstance(declared, int) or declared != counted:
        raise ValueError(
            f"checkpoint parameters {declared!r} != state_dict parameter count {counted} "
            f"({tensors} tensors including buffers); the checkpoint is corrupt or was "
            "written by another model")
    if expected_parameters is not None and int(expected_parameters) != counted:
        raise ValueError(
            f"checkpoint has {counted} parameters but the caller expected {expected_parameters}")
    train = payload.get("train")
    if not isinstance(train, Mapping):
        raise ValueError("checkpoint has no train block")
    missing = [key for key in _REQUIRED_TRAIN_KEYS if key not in train]
    if missing:
        raise ValueError(f"train block missing BC-v1 provenance keys: {missing}")
    if train.get("bc_version") != BC_VERSION:
        raise ValueError(
            f"train bc_version {train.get('bc_version')!r} != {BC_VERSION!r}")
    return {
        "schema": payload["schema"],
        "obs_spec_version": payload["obs_spec_version"],
        "obs_spec_digest": payload["obs_spec_digest"],
        "parameters": counted,
        "model": dict(payload.get("model") or {}),
        "bc_version": train["bc_version"],
        "tau": float(train["tau"]),
        "d0": train["d0"],
        "splits": train["splits"],
        "metrics": train["metrics"],
        "provenance": dict(payload.get("provenance") or {}),
        "disclosure": train.get("disclosure"),
    }


__all__ = [
    "BC_VERSION",
    "CHECKPOINT_SCHEMA",
    "SPLIT_NAMES",
    "SPLIT_RATIOS",
    "TAU_CANDIDATES",
    "TEACHER_BC_DISCLOSURE",
    "SplitError",
    "allocate_split_counts",
    "assign_shard_splits",
    "bomb_pos_weight",
    "check_episode_isolation",
    "checkpoint_summary",
    "kl_teacher_student",
    "select_tau",
    "softmax_scores",
    "split_report",
    "split_signature",
    "teacher_soft_labels",
]
