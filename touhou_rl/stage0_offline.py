"""Stage 0 offline V/Q learnability gate for the v3.4.0 research plan.

Why this module exists
----------------------
v3.1.1 / v3.3.0 spent seven policy trainings (BC x3, PPO x4) between 1,214 and
1,894 frames while uniform random sits at 2,787.7 and the scripted search teacher
at 6,400.8.  The v3.4.0 research plan therefore stops trying to replace search
with one forward pass and asks a cheaper question first:

    Is the *existing* privileged state + label definition enough to learn which
    of the 18 candidate moves has more future survival value?

This module answers that question offline, with no game process:

1. :func:`candidate_clearance` labels all 18 candidates from a recorded
   ``raw_safety_snapshot`` -- the same constant-velocity geometry the teacher
   plans with (``options.py`` / ``chunk_safety.py``), but with 10/20/40-frame
   horizons instead of 2/6/10.
2. :func:`build_cache` rebuilds the frozen ``v313-entity-obs/1`` observation
   (``entity_obs``) for every valid decision and stores the candidate labels,
   the teacher action, the executed action and the remaining survival frames.
3. :class:`ValueNet` / :class:`CandidateQNet` are small supervised models over
   that observation -- no pixels, no PPO, no search at inference time.
4. :func:`evaluate` computes the pre-registered gates and :func:`verdict`
   decides pass/fail.  The gates are frozen in
   ``Document/Update/Alpha/v3.4.0/执行计划.md``; changing them requires a new
   version, not a rerun.

Label semantics (and their known limits)
----------------------------------------
``clearance(a, H)`` is the minimum *surface distance* between the player hurtbox
and any recorded bullet over the next ``H`` frames, assuming the player holds
candidate ``a`` at a constant per-frame step and bullets keep their recorded
velocity.  It ignores spawning, lasers, enemy bodies and invulnerability, so it
is a lower bound on danger, not a safety certificate -- the same assumption the
teacher has always made.  If the episode is game-over before ``H`` frames, the
label is forced unsafe: the player did die, whatever the geometric model says.

The label contract is deliberately auditable (:func:`label_validity`): the
executed action's H=20 clearance is scored against "died within 60 frames".  A
label family that does not point at real danger cannot support Stage 1.

Commands
--------
    python -m touhou_rl.stage0_offline --build \
        --root rollouts=Document/Update/Alpha/v2.0.0/rollouts \
        --root teacher=third_party/th08/build/rl/step2-teacher-demos \
        --cache Document/Update/Alpha/v3.4.0/stage0/cache.npz

    python -m touhou_rl.stage0_offline --train \
        --cache Document/Update/Alpha/v3.4.0/stage0/cache.npz \
        --out Document/Update/Alpha/v3.4.0/stage0 --device cuda --seed 11
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import entity_obs

SCHEMA = "v340-stage0-offline/1"
SNAPSHOT_SCHEMA = "v200-raw-safety-snapshot/1"
#: Candidate horizons in game frames, from the research plan (10/20/40).
HORIZONS: tuple[int, ...] = (10, 20, 40)
#: Clearance cap in px, identical to ``chunk_safety.ChunkSafetyConfig.clearance_cap``.
CLEARANCE_CAP = 60.0
#: Survival horizons (frames) for the V classification heads.
SURVIVAL_FRAMES: tuple[int, ...] = (60, 240)
#: ``frame stalled`` means the game-over/continue screen: the run is over.
VALID_PLAYER_STATES = (0, 3)
#: Number of candidates in the frozen ``focus*9 + move`` order.
N_CANDIDATES = 18
#: Index of the H=20 column: the headline horizon for agreement/risk.
HEADLINE_HORIZON = 20
#: Pre-registered gate thresholds (see 执行计划.md §6).  Frozen on purpose.
SPEARMAN_MIN = 0.60
Q_AUC_MIN = 0.75
AGREEMENT_MIN = 0.50
DEATH_AUC_MIN = 0.70
DEATH_P_MAX = 0.01
VALID_FRACTION_MIN = 0.90
MIN_VALIDATION_EPISODES = 10
#: Every 4th episode (by sorted name) is held out: ``index % 4 == 3``.
VALIDATION_STRIDE = 4
VALIDATION_RESIDUE = 3
#: Nominal episode length used to normalise the remaining-frames regression.
FRAME_SCALE = 8000.0


# ---------------------------------------------------------------------------
# geometry: labels for all 18 candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeometryConfig:
    """Constant-velocity geometry shared with ``options`` / ``chunk_safety``."""

    speed: float = 4.0
    speed_diag: float = 2.8284271
    focus_speed: float = 2.0
    focus_speed_diag: float = 1.4142135
    player_half: float = 0.825
    bullet_pad: float = 0.5
    clearance_cap: float = CLEARANCE_CAP

    def validate(self) -> None:
        for name in (
            "speed",
            "speed_diag",
            "focus_speed",
            "focus_speed_diag",
            "player_half",
            "bullet_pad",
            "clearance_cap",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


def candidate_steps(config: GeometryConfig = GeometryConfig()) -> np.ndarray:
    """Per-frame displacement of the 18 candidates in ``focus*9+move`` order."""
    config.validate()
    from .options import DIRECTIONS

    axis = np.repeat([config.speed, config.focus_speed], 9)
    diagonal = np.repeat([config.speed_diag, config.focus_speed_diag], 9)
    directions = np.tile(np.asarray(DIRECTIONS, dtype=np.float64), (2, 1))
    return directions * np.where(np.all(directions != 0, axis=1), diagonal, axis)[:, None]


def _bounds(config: GeometryConfig) -> tuple[np.ndarray, np.ndarray]:
    from . import spec

    low = np.array([spec.MOVE_AREA_X + config.player_half, spec.MOVE_AREA_Y + config.player_half])
    high = np.array(
        [
            spec.MOVE_AREA_X + spec.MOVE_AREA_W - config.player_half,
            spec.MOVE_AREA_Y + spec.MOVE_AREA_H - config.player_half,
        ]
    )
    return low, high


def parse_snapshot(snapshot: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Validate a recorded snapshot; return ``(player, position, velocity, half, stage)``.

    Mirrors ``chunk_safety._valid_snapshot`` on purpose: if a snapshot is not good
    enough for the shipping safety model, it is not good enough for a label here.
    Returns ``None`` for anything unusable -- callers count the reason, they do not
    silently coerce.
    """
    if not isinstance(snapshot, dict) or snapshot.get("schema") != SNAPSHOT_SCHEMA:
        return None
    if snapshot.get("complete") is not True:
        return None
    player_state = snapshot.get("player_state")
    if player_state is not None and player_state not in VALID_PLAYER_STATES:
        return None
    try:
        player = np.asarray(snapshot["player_position"], dtype=np.float64).reshape(-1)
        stage = int(snapshot["stage"])
        bullets = snapshot["bullets"]
        if player.shape != (2,) or not isinstance(bullets, dict):
            return None
        position = np.asarray(bullets["position"], dtype=np.float64)
        velocity = np.asarray(bullets["velocity"], dtype=np.float64)
        half = np.asarray(bullets["half"], dtype=np.float64).reshape(-1)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if position.size == 0:
        position = np.zeros((0, 2), dtype=np.float64)
        velocity = np.zeros((0, 2), dtype=np.float64)
        half = np.zeros((0,), dtype=np.float64)
    elif position.ndim != 2 or position.shape[1] != 2:
        return None
    if velocity.shape != position.shape or half.shape != (position.shape[0],):
        return None
    if stage < 0:
        return None
    for array in (player, position, velocity, half):
        if not np.all(np.isfinite(array)):
            return None
    if np.any(half < 0.0):
        return None
    return player, position, velocity, half, stage


def candidate_clearance(
    player: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    half: np.ndarray,
    horizons: Sequence[int] = HORIZONS,
    config: GeometryConfig = GeometryConfig(),
) -> dict[str, np.ndarray]:
    """Minimum surface distance per candidate per horizon, ``(18, len(horizons))``.

    One frame at a time, like ``chunk_safety.predict`` but with arbitrary
    horizons: the player is stepped along candidate ``a`` and clipped to the move
    area, bullets move linearly, and the distance is taken between the two moving
    segments so a fast bullet cannot tunnel through the hurtbox between frames.
    """
    config.validate()
    horizons = tuple(int(h) for h in horizons)
    if not horizons or min(horizons) < 1:
        raise ValueError("horizons must be a non-empty sequence of positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique")
    position = np.asarray(position, dtype=np.float64).reshape(-1, 2)
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1, 2)
    half = np.asarray(half, dtype=np.float64).reshape(-1)
    player = np.asarray(player, dtype=np.float64).reshape(2)
    if position.shape != velocity.shape:
        raise ValueError("position and velocity must share shape (N,2)")
    if half.shape != (position.shape[0],):
        raise ValueError("half must have shape (N,)")
    clearance = np.full((N_CANDIDATES, len(horizons)), float(config.clearance_cap), dtype=np.float64)
    if position.shape[0] == 0:
        return {"clearance": clearance, "safe": np.ones_like(clearance, dtype=bool)}
    steps = candidate_steps(config)
    low, high = _bounds(config)
    horizon_columns = {horizon: index for index, horizon in enumerate(horizons)}
    origin = np.clip(player, low, high)
    current = np.tile(origin, (len(steps), 1))
    radius = half + config.player_half + config.bullet_pad
    minimum = np.full(len(steps), float(config.clearance_cap), dtype=np.float64)
    ending = max(horizons)
    for frame in range(1, ending + 1):
        following = np.clip(current + steps, low, high)
        # Segment distance between player delta and bullet delta, frame-1 -> frame.
        offset = current[:, None, :] - (position + velocity * (frame - 1))[None, :, :]
        delta = (following - current)[:, None, :] - velocity[None, :, :]
        denominator = np.sum(delta * delta, axis=-1)
        fraction = np.clip(
            np.divide(
                -np.sum(offset * delta, axis=-1),
                denominator,
                out=np.zeros_like(denominator),
                where=denominator > 0.0,
            ),
            0.0,
            1.0,
        )
        distance = np.linalg.norm(offset + fraction[..., None] * delta, axis=-1) - radius[None, :]
        minimum = np.minimum(minimum, distance.min(axis=1))
        column = horizon_columns.get(frame)
        if column is not None:
            clearance[:, column] = minimum
        current = following
    return {"clearance": clearance, "safe": clearance > 0.0}


def rejection_reason(snapshot: Any) -> str:
    """Why :func:`parse_snapshot` would reject this row (for coverage accounting)."""
    if not isinstance(snapshot, dict):
        return "missing"
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        return "schema"
    if snapshot.get("complete") is not True:
        return "incomplete"
    if snapshot.get("player_state") not in (None, *VALID_PLAYER_STATES):
        return "player_state"
    return "malformed"


def label_snapshot(
    snapshot: Any,
    *,
    horizons: Sequence[int] = HORIZONS,
    config: GeometryConfig = GeometryConfig(),
) -> dict[str, Any]:
    """Geometry labels for one snapshot plus a machine-readable rejection reason."""
    parsed = parse_snapshot(snapshot)
    if parsed is None:
        return {"valid": False, "reason": rejection_reason(snapshot)}
    player, position, velocity, half, stage = parsed
    labelled = candidate_clearance(player, position, velocity, half, horizons, config)
    return {
        "valid": True,
        "reason": "",
        "clearance": labelled["clearance"],
        "safe": labelled["safe"],
        "parsed": (player, position, velocity, half, stage),
    }


# ---------------------------------------------------------------------------
# decision traces -> cache
# ---------------------------------------------------------------------------


def trace_files(root: str | Path) -> list[Path]:
    """Every ``episode-*-decisions.jsonl`` under ``root`` (nested one level included)."""
    base = Path(root)
    if not base.is_dir():
        raise ValueError(f"trace root does not exist: {base}")
    found = sorted(
        Path(item) for item in glob.glob(str(base / "**" / "episode-*-decisions.jsonl"), recursive=True)
    )
    if not found:
        raise ValueError(f"no decision traces under {base}")
    return found


def group_name(root_name: str, path: Path) -> str:
    """Corpus-qualified episode id: the leak-free split unit."""
    return f"{root_name}/{path.parent.name}/{path.stem}"


def candidate_of_action(action: Any) -> int:
    """``(move, shoot, focus, bomb)`` -> ``focus*9 + move``; ``-1`` when invalid."""
    if not isinstance(action, (list, tuple, np.ndarray)) or len(action) < 3:
        return -1
    try:
        move, _shoot, focus = int(action[0]), int(action[1]), int(action[2])
    except (TypeError, ValueError):
        return -1
    if not 0 <= move < 9 or focus not in (0, 1):
        return -1
    return entity_obs.candidate_index(move, focus)


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _episode_end(rows: list[dict]) -> int:
    """Last recorded frame plus its step length; the run is over there."""
    end = 0
    for row in rows:
        try:
            frame = int(row.get("frame"))
        except (TypeError, ValueError):
            continue
        try:
            delta = int(row.get("frame_delta") or 0)
        except (TypeError, ValueError):
            delta = 0
        end = max(end, frame + max(0, delta))
    return end


def held_out_groups(names: Sequence[str]) -> set[str]:
    """Deterministic whole-episode hold-out: every ``VALIDATION_STRIDE``-th name.

    Sorted names make the split reproducible from the cache alone and independent
    of filesystem ordering.  One episode -- not one decision -- is the unit,
    because consecutive decisions describe nearly the same state.
    """
    unique = sorted(set(str(name) for name in names))
    return {
        name for index, name in enumerate(unique) if index % VALIDATION_STRIDE == VALIDATION_RESIDUE
    }


def split_by_group(groups: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-row split from per-row group names (``train_index``, ``validation_index``)."""
    held = held_out_groups(groups)
    mask = np.array([str(group) in held for group in groups], dtype=bool)
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def build_cache(
    roots: Sequence[tuple[str, str | Path]],
    cache: str | Path,
    *,
    max_rows_per_group: int | None = None,
    progress=print,
) -> dict[str, Any]:
    """Rebuild observations + labels for every valid decision under ``roots``."""
    if not roots:
        raise ValueError("at least one (name, root) pair is required")
    files: list[tuple[int, str, Path]] = []
    corpus_names: list[str] = []
    for corpus_id, (name, root) in enumerate(roots):
        corpus_names.append(str(name))
        for path in trace_files(root):
            files.append((corpus_id, str(name), path))
    if not files:
        raise ValueError("no decision traces found")

    groups_sorted = sorted({group_name(name, path) for _cid, name, path in files})
    held = held_out_groups(groups_sorted)
    group_to_id = {name: index for index, name in enumerate(groups_sorted)}

    started = time.perf_counter()
    field_names = (
        "bullets",
        "meta",
        "grid",
        "clearance",
        "teacher",
        "executed",
        "group",
        "corpus",
        "frame",
        "remaining",
        "split",
    )
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in field_names}
    dropped: dict[str, int] = {}
    per_corpus: dict[str, dict[str, int]] = {name: {"parsed": 0, "kept": 0} for name in corpus_names}
    kept = 0
    for corpus_id, corpus_name, path in files:
        group = group_name(corpus_name, path)
        rows = _load_rows(path)
        per_corpus[corpus_name]["parsed"] += len(rows)
        if max_rows_per_group is not None:
            rows = rows[: int(max_rows_per_group)]
        end_frame = _episode_end(rows)
        local: dict[str, list[np.ndarray]] = {name: [] for name in field_names}
        for row in rows:
            labelled = label_snapshot(row.get("raw_safety_snapshot"))
            if not labelled["valid"]:
                reason = str(labelled["reason"])
                dropped[reason] = dropped.get(reason, 0) + 1
                continue
            teacher = candidate_of_action(row.get("teacher_action"))
            executed = candidate_of_action(row.get("executed_action"))
            if teacher < 0 or executed < 0:
                dropped["action"] = dropped.get("action", 0) + 1
                continue
            try:
                frame = int(row.get("frame"))
            except (TypeError, ValueError):
                dropped["frame"] = dropped.get("frame", 0) + 1
                continue
            player, position, velocity, half, stage = labelled["parsed"]
            observation = entity_obs.build(
                float(player[0]), float(player[1]), position, velocity, half, stage=stage
            )
            remaining = max(0, end_frame - frame)
            clearance = np.array(labelled["clearance"], dtype=np.float64)
            # Death before the horizon makes every candidate unsafe, whatever the
            # constant-velocity model believes.
            for column, horizon in enumerate(HORIZONS):
                if remaining < horizon:
                    clearance[:, column] = 0.0
            local["bullets"].append(observation["bullets"].astype(np.float16))
            local["meta"].append(observation["meta"])
            local["grid"].append(observation["grid"].astype(np.float16))
            local["clearance"].append(clearance.T.astype(np.float16))
            local["teacher"].append(teacher)
            local["executed"].append(executed)
            local["group"].append(group_to_id[group])
            local["corpus"].append(corpus_id)
            local["frame"].append(frame)
            local["remaining"].append(remaining)
            local["split"].append(1 if group in held else 0)
        if local["bullets"]:
            for name in field_names:
                chunks[name].append(np.stack(local[name]).astype(_cache_dtype(name)))
            kept += len(local["bullets"])
            per_corpus[corpus_name]["kept"] += len(local["bullets"])
            progress(
                f"[stage0] {group:<46} kept={len(local['bullets']):5d} dropped={sum(dropped.values())}"
            )

    if kept == 0:
        raise ValueError("no usable decision rows; nothing to cache")
    arrays = {name: np.concatenate(parts, axis=0) for name, parts in chunks.items() if parts}
    arrays["group_names"] = np.array(groups_sorted)
    arrays["corpus_names"] = np.array(corpus_names)
    cache_path = Path(cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)

    parsed_rows = sum(item["parsed"] for item in per_corpus.values())
    summary = {
        "schema": SCHEMA,
        "cache": str(cache_path),
        "decisions": int(kept),
        "parsed_rows": int(parsed_rows),
        "valid_fraction": round(kept / max(1, parsed_rows), 4),
        "dropped": dropped,
        "episodes": len(groups_sorted),
        "validation_episodes": len(held),
        "held_out_groups": sorted(held),
        "corpora": {name: {"parsed": item["parsed"], "kept": item["kept"]} for name, item in per_corpus.items()},
        "horizons": list(HORIZONS),
        "clearance_cap": CLEARANCE_CAP,
        "split": f"sorted episode index % {VALIDATION_STRIDE} == {VALIDATION_RESIDUE}",
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "bytes": int(cache_path.stat().st_size),
    }
    summary_path = cache_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(f"[stage0] cache written: {cache_path} ({summary['bytes']} bytes)")
    return summary


def _cache_dtype(name: str) -> Any:
    if name in ("bullets", "grid", "clearance"):
        return np.float16
    if name == "meta":
        return np.float32
    if name in ("teacher", "executed", "group", "corpus", "split"):
        return np.int16
    if name == "frame":
        return np.int32
    return np.int32


def load_cache(cache: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(cache), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


def clearance_target(clearance: np.ndarray, cap: float = CLEARANCE_CAP) -> np.ndarray:
    """``[0, 1]`` target: ``log1p(clip(clearance)) / log1p(cap)``.

    The log compresses the ``= cap`` majority (nothing threatens) and expands the
    few-pixel range that actually decides "safe or not", without moving the safety
    threshold at 0.
    """
    values = np.clip(np.asarray(clearance, dtype=np.float64), 0.0, float(cap))
    return np.log1p(values) / np.log1p(float(cap))


def clearance_from_target(target: np.ndarray, cap: float = CLEARANCE_CAP) -> np.ndarray:
    """Inverse of :func:`clearance_target` (used for reported pixel values)."""
    clipped = np.clip(np.asarray(target, dtype=np.float64), 0.0, 1.0)
    return np.expm1(clipped * np.log1p(float(cap)))


def survival_targets(remaining: np.ndarray, horizons: Sequence[int] = SURVIVAL_FRAMES) -> np.ndarray:
    """``(B, len(horizons))`` 0/1 labels: did the run survive that many frames?"""
    remaining = np.asarray(remaining, dtype=np.float64).reshape(-1)
    return np.stack([remaining >= float(horizon) for horizon in horizons], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# metrics (numpy only; scipy is not a dependency of this project)
# ---------------------------------------------------------------------------


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, 1-based, ties sharing the mean rank."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    ordered = values[order]
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and ordered[stop + 1] == ordered[start]:
            stop += 1
        if stop > start:
            ranks[order[start : stop + 1]] = (start + stop + 2) / 2.0
        start = stop + 1
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    """Spearman rank correlation; ``None`` when either side is constant."""
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size != right.size or left.size < 3:
        return None
    left_ranks = rankdata(left)
    right_ranks = rankdata(right)
    left_ranks = left_ranks - left_ranks.mean()
    right_ranks = right_ranks - right_ranks.mean()
    denominator = float(np.sqrt(np.sum(left_ranks**2) * np.sum(right_ranks**2)))
    if denominator <= 0.0:
        return None
    return float(np.sum(left_ranks * right_ranks) / denominator)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Mann-Whitney AUC with tie-aware ranks; ``None`` when a class is missing."""
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size:
        raise ValueError("labels and scores disagree in length")
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(scores)
    positive_rank_sum = float(ranks[labels].sum())
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def permutation_p(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    samples: int = 200,
    seed: int = 11,
) -> float | None:
    """One-sided permutation p for ``AUC > 0.5``; ``None`` if AUC is undefined."""
    observed = roc_auc(labels, scores)
    if observed is None:
        return None
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    generator = np.random.default_rng(seed)
    better = 0
    for _ in range(int(samples)):
        shuffled = roc_auc(generator.permutation(labels), scores)
        if shuffled is not None and shuffled >= observed:
            better += 1
    return float((better + 1) / (int(samples) + 1))


def _subsample(indices: np.ndarray, count: int, seed: int) -> np.ndarray:
    if indices.size <= count:
        return indices
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(indices, size=int(count), replace=False))


def _rounded(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def _torch():
    import torch  # imported lazily so cache building needs no torch

    return torch


def _encoder_class():
    torch = _torch()

    class Encoder(torch.nn.Module):
        """Bullet set + danger grid + meta -> ``hidden``-d state embedding."""

        def __init__(self, hidden: int = 256) -> None:
            super().__init__()
            self.bullets = torch.nn.Sequential(
                torch.nn.Linear(entity_obs.BULLET_FEATURES, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
            )
            self.grid = torch.nn.Sequential(
                torch.nn.Conv2d(entity_obs.GRID_CHANNELS, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
            )
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(128 + 64 + entity_obs.META_DIM, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.ReLU(),
            )

        def forward(self, bullets, meta, grid):
            encoded = self.bullets(bullets)
            valid = bullets[..., 5:6]
            masked = encoded + (valid - 1.0) * 1e4
            maximum = masked.max(dim=1).values
            counted = valid.sum(dim=1).clamp(min=1.0)
            average = (encoded * valid).sum(dim=1) / counted
            pooled = torch.cat([maximum, average, self.grid(grid), meta], dim=-1)
            return self.trunk(pooled)

    return Encoder


def _value_class():
    torch = _torch()
    Encoder = _encoder_class()

    class ValueNet(torch.nn.Module):
        """V(s): remaining frames (regression) + survival logits."""

        def __init__(self, hidden: int = 256, survival_frames: Sequence[int] = SURVIVAL_FRAMES) -> None:
            super().__init__()
            self.encoder = Encoder(hidden)
            self.remaining = torch.nn.Linear(hidden, 1)
            self.survival_frames = tuple(int(item) for item in survival_frames)
            self.survival = torch.nn.Linear(hidden, len(self.survival_frames))

        def forward(self, bullets, meta, grid):
            state = self.encoder(bullets, meta, grid)
            return self.remaining(state).squeeze(-1), self.survival(state)

    return ValueNet


def _q_class():
    torch = _torch()
    Encoder = _encoder_class()

    class CandidateQNet(torch.nn.Module):
        """Q(s, .): one value per candidate and horizon, shaped ``(B, H, 18)``."""

        def __init__(self, hidden: int = 256, horizons: Sequence[int] = HORIZONS) -> None:
            super().__init__()
            self.encoder = Encoder(hidden)
            self.horizons = tuple(int(item) for item in horizons)
            self.head = torch.nn.Linear(hidden, len(self.horizons) * N_CANDIDATES)

        def forward(self, bullets, meta, grid):
            state = self.encoder(bullets, meta, grid)
            flat = self.head(state)
            return flat.view(-1, len(self.horizons), N_CANDIDATES)

    return CandidateQNet


def make_value_net(hidden: int = 256, survival_frames: Sequence[int] = SURVIVAL_FRAMES):
    return _value_class()(hidden=hidden, survival_frames=survival_frames)


def make_q_net(hidden: int = 256, horizons: Sequence[int] = HORIZONS):
    return _q_class()(hidden=hidden, horizons=horizons)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _device(requested: str):
    torch = _torch()
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda requested but torch.cuda.is_available() is False")
    return torch.device(requested)


class _Batches:
    """Index the cache on demand: bullets/grid are float16, converted per batch."""

    def __init__(self, data: dict[str, np.ndarray], indices: np.ndarray, device, torch) -> None:
        self.data = data
        self.indices = np.sort(np.asarray(indices, dtype=np.int64))
        self.device = device
        self.torch = torch

    def __len__(self) -> int:
        return int(self.indices.size)

    def batch(self, rows: np.ndarray) -> tuple[Any, ...]:
        """Materialise the given **state rows** (not positional indices)."""
        selected = np.sort(np.asarray(rows, dtype=np.int64))
        torch = self.torch
        bullets = torch.as_tensor(self.data["bullets"][selected], device=self.device).float()
        meta = torch.as_tensor(self.data["meta"][selected], device=self.device)
        grid = torch.as_tensor(self.data["grid"][selected], device=self.device).float()
        return bullets, meta, grid, selected


def _iterate(indices: np.ndarray, batch_size: int):
    for start in range(0, indices.size, int(batch_size)):
        yield start, indices[start : start + int(batch_size)]


def _value_losses(model, batches: _Batches, rows: np.ndarray, remaining_target, survival_target, torch):
    bullets, meta, grid, selected = batches.batch(rows)
    predicted, logits = model(bullets, meta, grid)
    target = torch.as_tensor(remaining_target[selected], device=batches.device)
    label = torch.as_tensor(survival_target[selected], device=batches.device)
    regression = torch.nn.functional.mse_loss(predicted, target)
    classification = torch.nn.BCEWithLogitsLoss()(logits, label)
    return regression, classification


def train_value(
    data: dict[str, np.ndarray],
    train_index: np.ndarray,
    validation_index: np.ndarray,
    out_dir: str | Path,
    *,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    seed: int = 11,
    hidden: int = 256,
    progress=print,
) -> dict[str, Any]:
    """Supervised V(s): remaining frames + survival heads; checkpoint on val loss."""
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_obj = _device(device)
    model = make_value_net(hidden=hidden).to(device_obj)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train = _Batches(data, train_index, device_obj, torch)
    validation = _Batches(data, validation_index, device_obj, torch)
    remaining_target = np.clip(data["remaining"].astype(np.float64) / FRAME_SCALE, 0.0, 1.0).astype(np.float32)
    survival_target = survival_targets(data["remaining"]).astype(np.float32)

    history: list[dict[str, Any]] = []
    best = {"validation_loss": float("inf"), "epoch": -1}
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = train.indices.copy()
        np.random.shuffle(order)
        total = 0.0
        seen = 0
        for _start, rows in _iterate(order, batch_size):
            regression, classification = _value_losses(
                model, train, rows, remaining_target, survival_target, torch
            )
            loss = regression + 0.5 * classification
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += float(loss.detach()) * rows.size
            seen += int(rows.size)

        model.eval()
        with torch.no_grad():
            validation_total = 0.0
            for _start, rows in _iterate(validation.indices, 512):
                regression, classification = _value_losses(
                    model, validation, rows, remaining_target, survival_target, torch
                )
                validation_total += float(regression + 0.5 * classification) * rows.size
            validation_loss = validation_total / max(1, validation.__len__())
        row = {
            "epoch": epoch,
            "train_loss": round(total / max(1, seen), 6),
            "validation_loss": round(validation_loss, 6),
        }
        history.append(row)
        progress(f"[stage0] V {row}")
        if validation_loss < best["validation_loss"]:
            best = {"validation_loss": validation_loss, "epoch": epoch}
            torch.save(
                {
                    "schema": SCHEMA,
                    "kind": "value",
                    "hidden": int(hidden),
                    "survival_frames": list(model.survival_frames),
                    "state_dict": model.state_dict(),
                    "extra": {
                        "epoch": epoch,
                        "validation_loss": validation_loss,
                        "train_states": int(train_index.size),
                        "validation_states": int(validation_index.size),
                        "frame_scale": FRAME_SCALE,
                    },
                },
                out_path / "value.pt",
            )
    summary = {
        "kind": "value",
        "epochs": int(epochs),
        "train_states": int(train_index.size),
        "validation_states": int(validation_index.size),
        "best": best,
        "history": history,
        "checkpoint": str(out_path / "value.pt"),
    }
    (out_path / "value-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def train_q(
    data: dict[str, np.ndarray],
    train_index: np.ndarray,
    validation_index: np.ndarray,
    out_dir: str | Path,
    *,
    epochs: int = 12,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    seed: int = 11,
    hidden: int = 256,
    progress=print,
) -> dict[str, Any]:
    """Supervised Q(s, .) over the 18 candidates x 3 horizons."""
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_obj = _device(device)
    model = make_q_net(hidden=hidden).to(device_obj)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train = _Batches(data, train_index, device_obj, torch)
    validation = _Batches(data, validation_index, device_obj, torch)
    target = clearance_target(data["clearance"].astype(np.float32)).astype(np.float32)

    history: list[dict[str, Any]] = []
    best = {"validation_loss": float("inf"), "epoch": -1}
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = train.indices.copy()
        np.random.shuffle(order)
        total = 0.0
        seen = 0
        for _start, rows in _iterate(order, batch_size):
            bullets, meta, grid, selected = train.batch(rows)
            predicted = model(bullets, meta, grid)
            label = torch.as_tensor(target[selected], device=device_obj)
            if predicted.shape != label.shape:
                raise ValueError(
                    f"Q prediction {tuple(predicted.shape)} does not match label {tuple(label.shape)}"
                )
            loss = torch.nn.functional.mse_loss(predicted, label)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += float(loss.detach()) * selected.size
            seen += int(selected.size)

        model.eval()
        with torch.no_grad():
            validation_total = 0.0
            for _start, rows in _iterate(validation.indices, 512):
                bullets, meta, grid, selected = validation.batch(rows)
                predicted = model(bullets, meta, grid)
                label = torch.as_tensor(target[selected], device=device_obj)
                validation_total += float(torch.nn.functional.mse_loss(predicted, label)) * selected.size
            validation_loss = validation_total / max(1, validation.__len__())
        row = {
            "epoch": epoch,
            "train_loss": round(total / max(1, seen), 6),
            "validation_loss": round(validation_loss, 6),
        }
        history.append(row)
        progress(f"[stage0] Q {row}")
        if validation_loss < best["validation_loss"]:
            best = {"validation_loss": validation_loss, "epoch": epoch}
            torch.save(
                {
                    "schema": SCHEMA,
                    "kind": "candidate_q",
                    "hidden": int(hidden),
                    "horizons": list(model.horizons),
                    "state_dict": model.state_dict(),
                    "extra": {
                        "epoch": epoch,
                        "validation_loss": validation_loss,
                        "train_states": int(train_index.size),
                        "validation_states": int(validation_index.size),
                        "clearance_cap": CLEARANCE_CAP,
                    },
                },
                out_path / "q.pt",
            )
    summary = {
        "kind": "candidate_q",
        "epochs": int(epochs),
        "train_states": int(train_index.size),
        "validation_states": int(validation_index.size),
        "best": best,
        "history": history,
        "checkpoint": str(out_path / "q.pt"),
    }
    (out_path / "q-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _load_model(checkpoint: str | Path, kind: str, device: str):
    torch = _torch()
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    schema = payload.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"unexpected checkpoint schema: {schema}")
    if payload.get("kind") != kind:
        raise ValueError(f"expected a {kind} checkpoint, got {payload.get('kind')}")
    if kind == "value":
        model = make_value_net(
            hidden=int(payload.get("hidden", 256)),
            survival_frames=payload.get("survival_frames", SURVIVAL_FRAMES),
        )
    else:
        model = make_q_net(hidden=int(payload.get("hidden", 256)), horizons=payload.get("horizons", HORIZONS))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def _predict_value(checkpoint: str | Path, data: dict[str, np.ndarray], indices: np.ndarray, device: str):
    torch = _torch()
    model = _load_model(checkpoint, "value", device)
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    remaining = np.zeros(indices.size, dtype=np.float64)
    logits = np.zeros((indices.size, len(model.survival_frames)), dtype=np.float64)
    with torch.no_grad():
        for start, rows in _iterate(indices, 512):
            bullets = torch.as_tensor(data["bullets"][rows], device=device).float()
            meta = torch.as_tensor(data["meta"][rows], device=device)
            grid = torch.as_tensor(data["grid"][rows], device=device).float()
            prediction, survival = model(bullets, meta, grid)
            remaining[start : start + rows.size] = prediction.cpu().numpy()
            logits[start : start + rows.size] = survival.cpu().numpy()
    return remaining, logits


def _predict_q(checkpoint: str | Path, data: dict[str, np.ndarray], indices: np.ndarray, device: str):
    torch = _torch()
    model = _load_model(checkpoint, "candidate_q", device)
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    values = np.zeros((indices.size, len(model.horizons), N_CANDIDATES), dtype=np.float64)
    with torch.no_grad():
        for start, rows in _iterate(indices, 512):
            bullets = torch.as_tensor(data["bullets"][rows], device=device).float()
            meta = torch.as_tensor(data["meta"][rows], device=device)
            grid = torch.as_tensor(data["grid"][rows], device=device).float()
            values[start : start + rows.size] = model(bullets, meta, grid).cpu().numpy()
    return values


def evaluate(
    data: dict[str, np.ndarray],
    value_checkpoint: str | Path,
    q_checkpoint: str | Path,
    *,
    device: str = "cuda",
    seed: int = 11,
    progress=print,
) -> dict[str, Any]:
    """Compute the pre-registered Stage 0 metrics on the held-out episodes."""
    validation = np.flatnonzero(data["split"].astype(np.int64) == 1)
    if validation.size == 0:
        raise ValueError("cache has no held-out decisions")
    groups = data["group"][validation].astype(np.int64)
    names = [str(item) for item in data["group_names"]]
    remaining = data["remaining"][validation].astype(np.float64)
    teacher = data["teacher"][validation].astype(np.int64)
    executed = data["executed"][validation].astype(np.int64)
    corpus = data["corpus"][validation].astype(np.int64)
    corpus_names = [str(item) for item in data["corpus_names"]]

    progress("[stage0] predicting V on held-out episodes")
    remaining_prediction, survival_logits = _predict_value(value_checkpoint, data, validation, device)
    progress("[stage0] predicting Q on held-out episodes")
    q_values = _predict_q(q_checkpoint, data, validation, device)

    # -- V: ordering and survival ------------------------------------------
    per_episode: list[float] = []
    per_episode_rows: list[dict[str, Any]] = []
    for group_id in sorted(set(int(item) for item in groups.tolist())):
        mask = groups == group_id
        correlation = spearman(remaining_prediction[mask], remaining[mask])
        if correlation is not None:
            per_episode.append(correlation)
            per_episode_rows.append(
                {
                    "episode": names[group_id] if group_id < len(names) else str(group_id),
                    "states": int(mask.sum()),
                    "spearman": round(correlation, 4),
                    "remaining_mean_frames": round(float(remaining[mask].mean()), 1),
                }
            )
    pooled = spearman(remaining_prediction, remaining)
    survival_label = survival_targets(data["remaining"])[validation]
    survival_metrics: dict[str, Any] = {}
    for index, horizon in enumerate(SURVIVAL_FRAMES):
        probability = 1.0 / (1.0 + np.exp(-survival_logits[:, index]))
        survival_metrics[f"survive_{horizon}_auc"] = _rounded(
            roc_auc(survival_label[:, index] > 0.5, probability)
        )
    value_report = {
        "spearman_pooled": _rounded(pooled),
        "spearman_per_episode_mean": _rounded(float(np.mean(per_episode)) if per_episode else None),
        "spearman_per_episode_median": _rounded(float(np.median(per_episode)) if per_episode else None),
        "spearman_per_episode_min": _rounded(float(np.min(per_episode)) if per_episode else None),
        "episodes": len(per_episode),
        "episode_rows": per_episode_rows,
        "remaining_mae_frames": _rounded(
            float(
                np.mean(
                    np.abs(np.clip(remaining_prediction, 0.0, 1.0) * FRAME_SCALE - remaining)
                )
            )
        ),
        **survival_metrics,
    }

    # -- Q: per-horizon safety discrimination and teacher agreement ---------
    true_clearance = data["clearance"][validation].astype(np.float64)
    predicted_clearance = clearance_from_target(np.clip(q_values, 0.0, 1.0))
    teacher_corpus = next(
        (index for index, name in enumerate(corpus_names) if name.lower() in ("teacher", "teacher_demos")),
        None,
    )
    teacher_mask = (
        corpus == teacher_corpus if teacher_corpus is not None else np.zeros(validation.size, dtype=bool)
    )
    headline_column = HORIZONS.index(HEADLINE_HORIZON)
    q_metrics: dict[str, Any] = {}
    for index, horizon in enumerate(HORIZONS):
        safe = true_clearance[:, index, :] > 0.0
        entry: dict[str, Any] = {
            "safe_auc": _rounded(roc_auc(safe.reshape(-1), predicted_clearance[:, index, :].reshape(-1))),
            "safe_fraction": round(float(safe.mean()), 4),
            "top1_safe_rate": round(
                float(safe[np.arange(validation.size), predicted_clearance[:, index, :].argmax(axis=1)].mean()), 4
            ),
            "oracle_safe_rate": round(float(safe.max(axis=1).mean()), 4),
            "predicted_top1_clearance_px": round(float(predicted_clearance[:, index, :].max(axis=1).mean()), 2),
        }
        if teacher_mask.any():
            subset_safe = safe[teacher_mask]
            entry["safe_auc_teacher_states"] = _rounded(
                roc_auc(subset_safe.reshape(-1), predicted_clearance[teacher_mask, index, :].reshape(-1))
            )
        # Diagnostic (not a gate): the pooled AUC above can be earned by knowing
        # which *states* are calm.  The within-state AUC asks the Stage 1 question --
        # can this state rank its own 18 candidates? -- so both numbers are reported.
        within_state: list[float] = []
        for row in range(validation.size):
            row_labels = safe[row]
            if row_labels.all() or not row_labels.any():
                continue
            value = roc_auc(row_labels, predicted_clearance[row, index, :])
            if value is not None:
                within_state.append(value)
        entry["within_state_safe_auc_mean"] = _rounded(
            float(np.mean(within_state)) if within_state else None
        )
        entry["within_state_states"] = len(within_state)
        q_metrics[str(horizon)] = entry
    q_report = {"horizons": list(HORIZONS), "headline_horizon": HEADLINE_HORIZON, "metrics": q_metrics}

    agreement: dict[str, Any] = {
        "states": int(teacher_mask.sum()),
        "random_baseline": round(1 / N_CANDIDATES, 4),
        "horizon": HEADLINE_HORIZON,
    }
    if teacher_mask.any():
        predicted_best = predicted_clearance[teacher_mask, headline_column, :].argmax(axis=1)
        teacher_actions = teacher[teacher_mask]
        label_best = true_clearance[teacher_mask, headline_column, :].argmax(axis=1)
        agreement["q_argmax_vs_teacher"] = round(float((predicted_best == teacher_actions).mean()), 4)
        agreement["label_argmax_vs_teacher"] = round(float((label_best == teacher_actions).mean()), 4)
        moved = teacher_actions != 0
        agreement["teacher_moved_states"] = int(moved.sum())
        agreement["teacher_idle_share"] = round(float((~moved).mean()), 4)
        if moved.any():
            agreement["q_argmax_vs_teacher_moved"] = round(
                float((predicted_best[moved] == teacher_actions[moved]).mean()), 4
            )
            agreement["label_argmax_vs_teacher_moved"] = round(
                float((label_best[moved] == teacher_actions[moved]).mean()), 4
            )
    else:
        agreement["q_argmax_vs_teacher"] = None
        agreement["reason"] = "teacher corpus absent from the held-out set"

    # -- death proximity risk ----------------------------------------------
    survival_index = list(SURVIVAL_FRAMES).index(60)
    risk = 1.0 - 1.0 / (1.0 + np.exp(-survival_logits[:, survival_index]))
    positive = np.flatnonzero(remaining < 60)
    negative_pool = np.flatnonzero(remaining > 300)
    if positive.size == 0 or negative_pool.size == 0:
        death_report: dict[str, Any] = {
            "auc": None,
            "reason": "one class absent",
            "positives": int(positive.size),
        }
    else:
        negatives = _subsample(negative_pool, positive.size, seed)
        chosen = np.concatenate([positive, negatives])
        labels = np.zeros(chosen.size, dtype=bool)
        labels[: positive.size] = True
        death_report = {
            "auc": _rounded(roc_auc(labels, risk[chosen])),
            "p_value": _rounded(permutation_p(labels, risk[chosen], seed=seed)),
            "positives": int(positive.size),
            "negatives": int(negatives.size),
            "risk_mean_positive": _rounded(float(risk[positive].mean())),
            "risk_mean_negative": _rounded(float(risk[negatives].mean())),
        }

    # -- label validity (diagnostic, not a gate) ---------------------------
    executed_clearance = true_clearance[
        np.arange(validation.size), headline_column, np.clip(executed, 0, N_CANDIDATES - 1)
    ]
    label_report = {
        "executed_clearance_vs_death60_auc": _rounded(roc_auc(remaining < 60, -executed_clearance)),
        "note": "diagnostic only: does the geometric label point at real danger?",
    }

    summary = {
        "validation_states": int(validation.size),
        "validation_episodes": int(len(set(int(item) for item in groups.tolist()))),
        "validation_teacher_states": int(teacher_mask.sum()),
        "held_out_groups": sorted({names[int(item)] for item in groups.tolist()}),
    }
    return {
        "summary": summary,
        "value": value_report,
        "q": q_report,
        "agreement": agreement,
        "death_risk": death_report,
        "label_validity": label_report,
    }


def verdict(metrics: dict[str, Any], data_summary: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen gates; a missing metric counts as a failure, not a skip."""
    gates: dict[str, Any] = {}

    corpora = data_summary.get("corpora") or {}
    valid_fraction = float(data_summary.get("valid_fraction") or 0.0)
    validation_episodes = int(metrics["summary"]["validation_episodes"])
    gates["S0-0_data_integrity"] = {
        "valid_fraction": valid_fraction,
        "validation_episodes": validation_episodes,
        "corpora": {name: int(item.get("kept", 0)) for name, item in corpora.items()},
        "threshold": (
            f"valid_fraction >= {VALID_FRACTION_MIN}, validation_episodes >= "
            f"{MIN_VALIDATION_EPISODES}, every corpus non-empty"
        ),
        "pass": bool(
            valid_fraction >= VALID_FRACTION_MIN
            and validation_episodes >= MIN_VALIDATION_EPISODES
            and len(corpora) >= 2
            and all(int(item.get("kept", 0)) > 0 for item in corpora.values())
        ),
    }

    spearman_value = metrics["value"]["spearman_per_episode_mean"]
    gates["S0-1_value_spearman"] = {
        "value": spearman_value,
        "threshold": SPEARMAN_MIN,
        "pass": bool(spearman_value is not None and spearman_value >= SPEARMAN_MIN),
    }

    q_aucs = {horizon: entry.get("safe_auc") for horizon, entry in metrics["q"]["metrics"].items()}
    gates["S0-2_q_safe_auc"] = {
        "value": q_aucs,
        "threshold": Q_AUC_MIN,
        "pass": bool(q_aucs and all(item is not None and item >= Q_AUC_MIN for item in q_aucs.values())),
    }

    agreement_value = metrics["agreement"].get("q_argmax_vs_teacher")
    gates["S0-3_teacher_agreement"] = {
        "value": agreement_value,
        "threshold": AGREEMENT_MIN,
        "pass": bool(agreement_value is not None and agreement_value >= AGREEMENT_MIN),
    }

    death = metrics["death_risk"]
    gates["S0-4_death_ranking"] = {
        "auc": death.get("auc"),
        "p_value": death.get("p_value"),
        "threshold": f"auc >= {DEATH_AUC_MIN} and p < {DEATH_P_MAX}",
        "pass": bool(
            death.get("auc") is not None
            and death["auc"] >= DEATH_AUC_MIN
            and death.get("p_value") is not None
            and death["p_value"] < DEATH_P_MAX
        ),
    }

    passed = all(gate["pass"] for gate in gates.values())
    return {
        "gates": gates,
        "passed": passed,
        "failed": [name for name, gate in gates.items() if not gate["pass"]],
        "verdict": "pass" if passed else "fail",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_root_specs(specs: Sequence[str]) -> list[tuple[str, str]]:
    """``name=path`` (or bare ``path``) -> ``[(name, path)]``; names must be unique."""
    parsed: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, spec in enumerate(specs):
        text = str(spec)
        if "=" in text:
            name, path = text.split("=", 1)
        else:
            name, path = f"root{index}", text
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"malformed root spec: {spec}")
        if name in seen:
            raise ValueError(f"duplicate corpus name: {name}")
        seen.add(name)
        parsed.append((name, path))
    return parsed


def _data_summary(data: dict[str, np.ndarray], cache: str | Path) -> dict[str, Any]:
    summary_path = Path(cache).with_suffix(".summary.json")
    built = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    validation = data["split"].astype(np.int64) == 1
    return {
        "cache": str(cache),
        "decisions": int(data["bullets"].shape[0]),
        "episodes": int(data["group_names"].size),
        "parsed_rows": int(built.get("parsed_rows", data["bullets"].shape[0])),
        # Strict reading when the build summary is missing: the data gate must not
        # pass by default.
        "valid_fraction": float(built.get("valid_fraction", 0.0)),
        "dropped": built.get("dropped", {}),
        "corpora": built.get("corpora", {name: {"kept": 0} for name in data["corpus_names"].tolist()}),
        "validation_episodes": int(
            len(set(int(item) for item in data["group"][validation].tolist()))
        ),
        "validation_states": int(validation.sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v3.4.0 Stage 0 offline V/Q gate")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true", help="build the dataset cache from decision traces")
    mode.add_argument("--train", action="store_true", help="train V/Q and evaluate the Stage 0 gates")
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="corpus root, optionally labelled as name=path; repeat for several corpora",
    )
    parser.add_argument("--cache", required=True, help="cache path (.npz)")
    parser.add_argument("--out", default="", help="output directory (--train)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--value-epochs", type=int, default=10)
    parser.add_argument("--q-epochs", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument(
        "--max-rows-per-group",
        type=int,
        default=None,
        help="smoke-test cap on rows read per episode (never for a reported run)",
    )
    args = parser.parse_args(argv)

    if args.build:
        if not args.root:
            parser.error("--build requires at least one --root")
        summary = build_cache(
            parse_root_specs(args.root),
            Path(args.cache),
            max_rows_per_group=args.max_rows_per_group,
        )
        print(
            json.dumps(
                {"decisions": summary["decisions"], "valid_fraction": summary["valid_fraction"], "cache": summary["cache"]},
                ensure_ascii=False,
            )
        )
        return 0

    if not args.out:
        parser.error("--train requires --out")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_cache(Path(args.cache))

    # The split is rebuilt from the sorted episode names and then verified against
    # the per-row flags stored in the cache; a mismatch is a corrupt cache, not a
    # reason to proceed.
    held = held_out_groups(data["group_names"])
    names_by_row = data["group_names"][data["group"].astype(np.int64)]
    expected = np.array([str(name) in held for name in names_by_row.tolist()], dtype=bool)
    cached = data["split"].astype(np.int64) == 1
    if not np.array_equal(expected, cached):
        raise ValueError("cache split disagrees with the deterministic hold-out rule: rebuild the cache")
    train_index = np.flatnonzero(~expected)
    validation_index = np.flatnonzero(expected)
    if train_index.size == 0 or validation_index.size == 0:
        raise ValueError("empty train or validation split")

    progress = print
    started = time.perf_counter()
    progress(
        f"[stage0] {int(data['group_names'].size)} episodes, {int(data['bullets'].shape[0])} states, "
        f"train={train_index.size}, validation={validation_index.size}"
    )
    value_summary = train_value(
        data,
        train_index,
        validation_index,
        out_dir,
        epochs=args.value_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        hidden=args.hidden,
    )
    q_summary = train_q(
        data,
        train_index,
        validation_index,
        out_dir,
        epochs=args.q_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        hidden=args.hidden,
    )
    metrics = evaluate(
        data, value_summary["checkpoint"], q_summary["checkpoint"], device=args.device, seed=args.seed
    )
    data_summary = _data_summary(data, args.cache)
    gates = verdict(metrics, data_summary)
    report = {
        "schema": SCHEMA,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cache": str(args.cache),
        "config": {
            "device": args.device,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "value_epochs": args.value_epochs,
            "q_epochs": args.q_epochs,
            "hidden": args.hidden,
            "horizons": list(HORIZONS),
            "clearance_cap": CLEARANCE_CAP,
            "survival_frames": list(SURVIVAL_FRAMES),
            "frame_scale": FRAME_SCALE,
            "split": f"sorted episode index % {VALIDATION_STRIDE} == {VALIDATION_RESIDUE}",
        },
        "data": data_summary,
        "training": {"value": value_summary, "q": q_summary},
        "metrics": metrics,
        "gates": gates,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    report_path = out_dir / "stage0-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"verdict": gates["verdict"], "failed": gates["failed"], "report": str(report_path)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
