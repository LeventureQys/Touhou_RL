"""v3.6.0 T1: offline information ablation for the seven tactics.

Why this module exists
----------------------
v3.5.0 S1 measured one tactic (large danger regions) and found it added *nothing*
over local clearance: region AUC 0.8094 against clearance AUC 0.9246, increment
-0.115 at p=0.002.  That result is the reason v3.6.0 §2 (six tactics) cannot be
turned into live control before each tactic proves it carries information.

The unit of ablation here is deliberately small and falsifiable::

    one tactic = one feature block (only that tactic can see it)
               + one future label  (the quantity that tactic claims to improve)
               + one baseline      (what you get without the block)

"Does tactic X help?" then becomes "does adding block X improve out-of-fold
prediction of label Y, over the same model, folds, and samples?" -- an
information question, not a learning-capability question.  A tactic may fail
this and still be learnable later; it cannot pass it without carrying signal.

What makes the labels honest
----------------------------
Every label is read from the *recorded future* of the same episode: the death
events, the real bullet fields and the real player positions at i+30 / i+120
decisions.  Nothing is extrapolated, so a feature cannot win by agreeing with
its own forward model.  Features only ever see decision i's snapshot.

What is deliberately *not* here
------------------------------
* No forward model / no simulator: counterfactual action outcomes are not
  observable offline, so action-level paired comparison belongs to T2 (live).
* No neural network: with 21k samples and <100 features a linear model plus a
  within-episode permutation test is the pre-registered protocol; it cannot
  manufacture an increment by overfitting.
* No enemy fire attribution: ``v350-trace/1`` has no bullet->enemy field, so
  "bullets removed by killing this enemy" is recorded as ``unknown`` and only a
  global ``bullet_delta`` proxy is used (see contract §2.3).

CLI::

    python -m touhou_rl.ablate36 --probe <S0 dir> --out <dir> [--stride N]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import spec
from .bullet_track import attribute_spawns, spawns_by_slot
from .region_probe import auc as rank_auc
from .region_probe import component_from
from .trace35 import read_episodes, read_trace, sha256_file, stable_split

SCHEMA = "v360-ablation/1"

# --- geometry / horizons (frozen before results; see 执行计划 §4) -------------
PLAYER_HALF = 0.825
BULLET_PAD = 0.5
CLR_CAP = 60.0
CLR_FLOOR = -10.0
TIMES = (0, 10, 20, 40, 60)
SCALES: tuple[tuple[int, int, int], ...] = ((8, 48, 56), (16, 24, 28))
MIN_REGION_CELLS = 80  # S1's "large region" (3% of the 8 px grid), unchanged
SAFE_MIN_CELLS = 8
FRAMESKIP = 2
HORIZON_STEPS = 10  # 10 * 2 = 20 frames, the local pilot's horizon
S = 30  # short label horizon in decisions = 60 frames
M = 120  # mid label horizon in decisions = 240 frames
MAX_BULLETS = 320
# Below this many px/frame a trace does not reveal the movement model (the
# player never moved), so ``measure_speeds`` must fall back instead of returning
# a near-zero speed that would disable every horizon feature.
MIN_MEASURED_SPEED = 1.0
# Spawn->enemy attribution radius (see reports/标签来源调查.md).  Fixed before any
# D3 result; a spawn farther than this from every enemy centre is unattributed.
SPAWN_RADIUS = 32.0

BLOCKS = ("BASE", "REGION", "LURE", "ENEMY", "POWER", "BOSS", "EMERGENCY", "HISTORY")

# Block membership of the two pre-registered ablation directions (§2.2).
ALL_BLOCKS = BLOCKS

# Pre-registered label -> metric used to score its out-of-fold prediction.
LABELS: dict[str, str] = {
    "death_60": "auc",
    "death_240": "auc",
    "min_clearance_60": "spearman",
    "min_clearance_240": "spearman",
    "danger_60": "spearman",
    "in_region_60": "auc",
    "bullet_delta_60": "spearman",
    "bullet_delta_240": "spearman",
    "power_delta_60": "spearman",
    "power_delta_240": "spearman",
    "p_collected_60": "auc",
    "boss_life_delta_60": "spearman",
    "boss_life_delta_240": "spearman",
    "bomb_needed_60": "auc",
    "bomb_used_60": "auc",
    # v3.6.0 D3: bullets actually spawned by a *visible enemy* within the next 60
    # frames.  This replaces the global bullet-count proxy that made ENEMY pass its
    # increment twice (D1 +0.0699, D2 +0.1100) while failing its absolute gate
    # twice (0.439 / 0.371) - the label, not the block, was the bottleneck.
    "attributed_spawns_60": "spearman",
}

# Pre-registered gate registry, transcribed from 执行计划 §6.1.
#
# ``primary``  = (label, absolute gate).  The absolute gate is ``(metric,
#                threshold)`` or None and is always evaluated on the *primary*
#                label.
# ``increments`` = list of ``(label, metric, threshold, combine)``.  ``combine``
#                is "and" (all listed increments must hold) or "or" (any one).
#                Every satisfied branch must also have permutation p < 0.05.
# ``extras``   = diagnostic labels reported but not gated.
#
# The label a gate is evaluated on is part of the pre-registration: REGION's
# increment lives on ``death_60`` (v3.5.0 S1's failed gate), not on its own
# ``in_region_60``.
CELLS: tuple[dict[str, Any], ...] = (
    {
        "strategy": "LocalDodge",
        "block": "BASE",
        "primary": ("death_60", ("auc", 0.80)),
        "increments": (),
        "extras": (("min_clearance_60", "spearman", 0.30),),
    },
    {
        "strategy": "REGION",
        "block": "REGION",
        "primary": ("in_region_60", ("auc", 0.80)),
        "increments": (("death_60", "auc", 0.02, "and"),),
        "extras": (),
    },
    {
        "strategy": "LURE",
        "block": "LURE",
        "primary": ("min_clearance_60", ("spearman", 0.30)),
        "increments": (
            ("min_clearance_60", "spearman", 0.05, "or"),
            ("death_60", "auc", 0.02, "or"),
        ),
        "extras": (),
    },
    {
        "strategy": "ENEMY",
        "block": "ENEMY",
        "primary": ("bullet_delta_60", ("spearman", 0.50)),
        "increments": (("bullet_delta_60", "spearman", 0.05, "and"),),
        "extras": (),
    },
    {
        "strategy": "POWER",
        "block": "POWER",
        "primary": ("p_collected_60", ("auc", 0.60)),
        "increments": (
            ("p_collected_60", "auc", 0.02, "or"),
            ("power_delta_60", "spearman", 0.05, "or"),
        ),
        "extras": (),
    },
    {
        "strategy": "BOSS",
        "block": "BOSS",
        "primary": ("boss_life_delta_60", ("spearman", 0.30)),
        "increments": (("boss_life_delta_60", "spearman", 0.05, "and"),),
        "extras": (),
    },
    {
        "strategy": "EMERGENCY",
        "block": "EMERGENCY",
        "primary": ("bomb_needed_60", ("brier", 0.20)),
        "increments": (("bomb_needed_60", "auc", 0.02, "and"),),
        "extras": (("bomb_used_60", "auc", 0.02),),
    },
    {
        "strategy": "HISTORY",
        "block": "HISTORY",
        "primary": ("death_60", None),
        "increments": (("death_60", "auc", 0.02, "and"),),
        "extras": (),
    },
)


def _with_d3_enemy_gate(cells: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """D3 gate set (执行计划-D3补遗 §4): only ENEMY moves.

    Its gate moves from the global ``bullet_delta_60`` proxy to the
    attributed-spawn label with the *same* thresholds (absolute Spearman >= 0.50,
    increment >= +0.05), so comparing D1/D2/D3 compares labels, not thresholds.
    """
    out = []
    for cell in cells:
        if cell["strategy"] == "ENEMY":
            out.append(
                {
                    **cell,
                    "primary": ("attributed_spawns_60", ("spearman", 0.50)),
                    "increments": (("attributed_spawns_60", "spearman", 0.05, "and"),),
                    "extras": (("bullet_delta_60", "spearman", 0.50),),
                }
            )
        else:
            out.append(cell)
    return tuple(out)


CELLS_D3 = _with_d3_enemy_gate(CELLS)
GATE_SETS: dict[str, tuple[dict[str, Any], ...]] = {"v1": CELLS, "d3": CELLS_D3}

PERMUTATIONS = 200
SEED = 3600
PERM_SEED = 3601
RIDGE_LAMBDA = 1.0
# Below this many positive samples a binary label cannot support an AUC gate.
UNDERPOWERED_POSITIVES = 100
# 执行计划-D2补遗 §3: when ``death_60`` has too few positives (human 1CC replays do
# not die), the *survival* label falls back to the continuous executed clearance
# with fixed substitute thresholds.  Only the label changes; the direction and
# magnitude requirements of every other gate stay identical.
SURVIVAL_LABEL = "death_60"
SURVIVAL_FALLBACK_LABEL = "min_clearance_60"
SURVIVAL_FALLBACK_ABS = ("spearman", 0.30)
SURVIVAL_FALLBACK_INC = 0.05
# A permutation null whose spread is numerically zero carries no p-value: it
# happens when the ablated block is (nearly) constant inside an episode, e.g.
# the EMERGENCY block in D1 where bombs/lives/player_state barely move.  Those
# cells are flagged ``degenerate_null`` and the *effect size* is the gate.
NULL_STD_EPS = 1e-9

# Direction unit vectors for movement classes 0..8 (same order as spec.MOVES).
_MOVE_VECTORS = np.array(
    [
        [0.0, 0.0],
        [0.0, -1.0],
        [1.0, -1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 0.0],
        [-1.0, -1.0],
    ],
    dtype=np.float32,
)


def resolve_survival_fallback(dataset: "Dataset") -> dict:
    """Decide whether the survival label is testable on this dataset.

    Pre-registered in 执行计划-D2补遗 §3.  A human 1CC stage-1 replay never dies, so
    ``death_60`` can end up with zero positives and an undefined AUC; the label
    then falls back to the continuous executed clearance.  The decision is a pure
    function of the *label counts*, which is why it can be pre-registered.
    """
    y = dataset.labels[SURVIVAL_LABEL]
    positives = int((np.nan_to_num(y) > 0.5).sum())
    fallback = positives < UNDERPOWERED_POSITIVES
    return {
        "label": SURVIVAL_LABEL,
        "positives": positives,
        "n_samples": int(np.isfinite(y).sum()),
        "fallback": bool(fallback),
        "fallback_label": SURVIVAL_FALLBACK_LABEL if fallback else None,
        "fallback_absolute": list(SURVIVAL_FALLBACK_ABS) if fallback else None,
        "fallback_increment": SURVIVAL_FALLBACK_INC if fallback else None,
    }


def resolve_cells(
    fallback: dict,
    base: tuple[dict[str, Any], ...] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Apply the survival-label fallback to a pre-registered gate registry.

    ``base`` selects the gate set (``GATE_SETS``); it defaults to ``CELLS`` so
    every D1/D2 number is reproduced by the same code path.
    """
    spec = CELLS if base is None else base
    if not fallback.get("fallback"):
        return spec
    resolved = []
    for spec_cell in spec:
        new = dict(spec_cell)
        label, absolute = new["primary"]
        if label == SURVIVAL_LABEL:
            new["primary"] = (SURVIVAL_FALLBACK_LABEL, SURVIVAL_FALLBACK_ABS if absolute is not None else None)
        increments = []
        for inc_label, inc_metric, inc_threshold, combine in new["increments"]:
            if inc_label == SURVIVAL_LABEL:
                increments.append((SURVIVAL_FALLBACK_LABEL, "spearman", SURVIVAL_FALLBACK_INC, combine))
            else:
                increments.append((inc_label, inc_metric, inc_threshold, combine))
        # LURE lists both labels with "or"; after substitution they collapse.
        new["increments"] = tuple(dict.fromkeys(increments))
        extras = []
        for extra_label, extra_metric, extra_threshold in new.get("extras", ()):
            if extra_label == SURVIVAL_LABEL:
                extras.append((SURVIVAL_FALLBACK_LABEL, "spearman", SURVIVAL_FALLBACK_INC))
            else:
                extras.append((extra_label, extra_metric, extra_threshold))
        new["extras"] = tuple(dict.fromkeys(extras))
        resolved.append(new)
    return tuple(resolved)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def _grid(cell: int) -> tuple[int, int]:
    for size, gw, gh in SCALES:
        if size == cell:
            return gw, gh
    raise ValueError(f"unknown cell size {cell}")


def _cell_index(points: np.ndarray, cell: int) -> tuple[np.ndarray, int, int]:
    gw, gh = _grid(cell)
    ix = np.floor((points[:, 0] - spec.PLAYFIELD_X) / cell).astype(np.int64)
    iy = np.floor((points[:, 1] - spec.PLAYFIELD_Y) / cell).astype(np.int64)
    keep = (ix >= 0) & (ix < gw) & (iy >= 0) & (iy < gh)
    return iy[keep] * gw + ix[keep], gw, gh


def occupancy(points: np.ndarray, cell: int) -> tuple[np.ndarray, np.ndarray]:
    """(covered mask, 3x3 density) for bullet centres at one time slice.

    ``covered`` is the same conservative 3x3 splat v3.5.0 S1 used (a cell is
    dangerous if any bullet centre lies within its 3x3 neighbourhood), so
    ``covered`` values stay comparable with S1.  ``density`` additionally keeps
    *how many* centres are in that neighbourhood, which is what the union/density
    region required by v3.6.0 T0 §4.1 needs: a single connected component cannot
    express "two separate walls closing in".
    """
    gw, gh = _grid(cell)
    idx, _, _ = _cell_index(points, cell)
    counts = np.bincount(idx, minlength=gw * gh).astype(np.float32).reshape(gh, gw)
    padded = np.pad(counts, 1, mode="constant")
    density = np.zeros_like(counts)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            density += padded[dy : dy + gh, dx : dx + gw]
    return density > 0.0, density


def player_cell(px: float, py: float, cell: int) -> tuple[int, int]:
    gw, gh = _grid(cell)
    ix = int(math.floor((px - spec.PLAYFIELD_X) / cell))
    iy = int(math.floor((py - spec.PLAYFIELD_Y) / cell))
    return max(0, min(gh - 1, iy)), max(0, min(gw - 1, ix))


def exit_info(mask: np.ndarray, start: tuple[int, int], radius: int = 24) -> tuple[int, int, float, float]:
    """Chebyshev ring distance to the nearest free component, its area, and its direction.

    Returns ``(ring, area, dx, dy)`` where ``dx/dy`` point from the player cell to
    the component centroid in cells.  ``ring == 0`` means the player cell itself
    is free.  Same wavefront as ``region_probe.nearest_safe_component`` but also
    keeps the ring distance, which is the "到出口的距离" the contract asks for.
    """
    free = ~mask
    gh, gw = mask.shape
    if free[start]:
        cells = component_from(free, start, limit=2000)
        cy = sum(c[0] for c in cells) / len(cells)
        cx = sum(c[1] for c in cells) / len(cells)
        return 0, len(cells), cx - start[1], cy - start[0]
    # A handful of BFS calls per decision: the first free cell of a ring is
    # representative, and a tiny pocket (< SAFE_MIN_CELLS) just moves the search
    # outward.  Doing one BFS per *candidate* cell measured ~8 ms/decision on the
    # S0 traces, which is why candidate components are inspected from the first
    # free cell of each ring and the total number of BFS calls is capped.
    fallback: tuple[int, int, float, float] | None = None
    calls = 0
    for ring in range(1, radius + 1):
        if calls >= 6:
            break
        for dy in range(-ring, ring + 1):
            for dx in range(-ring, ring + 1):
                if max(abs(dy), abs(dx)) != ring:
                    continue
                ny, nx = start[0] + dy, start[1] + dx
                if not (0 <= ny < gh and 0 <= nx < gw and free[ny, nx]):
                    continue
                calls += 1
                cells = component_from(free, (ny, nx), limit=2000)
                cy = sum(c[0] for c in cells) / len(cells)
                cx = sum(c[1] for c in cells) / len(cells)
                candidate = (ring, len(cells), cx - start[1], cy - start[0])
                if fallback is None or candidate[1] > fallback[1]:
                    fallback = candidate
                if candidate[1] >= SAFE_MIN_CELLS:
                    return candidate
                break
    if fallback is None or fallback[1] < SAFE_MIN_CELLS:
        # No *usable* exit (a one-cell pocket is not an exit): keep the distance
        # so "far away" and "blocked" stay distinguishable, but report area 0.
        return (fallback[0] if fallback is not None else radius), 0, 0.0, 0.0
    return fallback


def clearance_now(px: float, py: float, pos: np.ndarray, half: np.ndarray, t: int = 0, vel: np.ndarray | None = None) -> float:
    """Distance from the hurtbox surface to the nearest bullet surface (capped)."""
    if pos.shape[0] == 0:
        return CLR_CAP
    bx = pos[:, 0] if vel is None or t == 0 else pos[:, 0] + vel[:, 0] * t
    by = pos[:, 1] if vel is None or t == 0 else pos[:, 1] + vel[:, 1] * t
    dist = np.sqrt((bx - px) ** 2 + (by - py) ** 2) - half - PLAYER_HALF - BULLET_PAD
    return float(max(CLR_FLOOR, min(CLR_CAP, dist.min())))


def _per_step_disp(frameskip: int = FRAMESKIP, speeds: dict[str, float] | None = None) -> np.ndarray:
    """(18, 2) displacement per decision, ordered as focus*9 + move.

    ``speeds`` overrides the pilot's constants.  TH08 stores per-shot-type
    ``axisSpeed`` / ``diagonalSpeed`` in the player SHT file, and a D2 check on
    recorded human replays measured 5.0 / 1.9 px per frame for Youmu against
    4.0 / 2.0 for Border Team.  A 25% speed error over a 20-frame plan is a 20 px
    position error, so mixing characters with the Border constants would corrupt
    the horizon features rather than just add noise.
    """
    if speeds is None:
        axis_speed = 4.0
        focus_speed = 2.0
    else:
        axis_speed = float(speeds["speed"])
        focus_speed = float(speeds["focus_speed"])
    axis = np.array([axis_speed, focus_speed], dtype=np.float32)
    # The diagonal per-axis component is speed / sqrt(2); measured |v| on both
    # Border Team and Youmu replays matches that structure to <1%.
    diag = axis / np.sqrt(2.0)
    disp = np.zeros((2, 9, 2), dtype=np.float32)
    for focus in (0, 1):
        for move in range(9):
            ux, uy = _MOVE_VECTORS[move]
            comp = diag[focus] if (ux != 0.0 and uy != 0.0) else axis[focus]
            disp[focus, move] = (ux * comp, uy * comp)
    return disp.reshape(18, 2) * float(frameskip)


_DISP = _per_step_disp()


def measure_speeds(records: list[dict], percentile: float = 95.0) -> dict[str, float]:
    """Estimate the shot type's per-frame speeds from recorded velocities.

    Uses a high percentile of ``|v|`` inside each (direction, focus) bucket: the
    game accelerates over a few frames after a direction change, so the *mean*
    under-reports the steady-state speed (Border measures 3.95 by mean and 4.0 by
    p95).  Falls back to the Border constants when a bucket is empty, and records
    which fallback was used so a report can show it instead of assuming.
    """
    buckets: dict[tuple[int, int], list[float]] = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
    for record in records:
        action = record.get("action") or [0, 0, 0, 0]
        move = int(action[0]) if len(action) > 0 else 0
        focus = int(action[2]) if len(action) > 2 else 0
        player = record.get("player") or {}
        vx, vy = player.get("vx"), player.get("vy")
        if vx is None or vy is None:
            continue
        ux, uy = _MOVE_VECTORS[move]
        axis = 1 if (abs(ux) + abs(uy)) == 1 else 0
        diagonal = 1 if (ux != 0.0 and uy != 0.0) else 0
        speed = math.hypot(float(vx), float(vy))
        if axis:
            buckets[(0, focus)].append(speed)
        elif diagonal:
            buckets[(1, focus)].append(speed)
    def p(values: list[float]) -> float | None:
        return float(np.percentile(values, percentile)) if len(values) >= 20 else None
    axis_normal = p(buckets[(0, 0)])
    axis_focus = p(buckets[(0, 1)])
    diag_normal = p(buckets[(1, 0)])
    diag_focus = p(buckets[(1, 1)])
    # Prefer axis buckets; fall back to diagonal |v| (same quantity by construction).
    speed = axis_normal if axis_normal is not None else diag_normal
    focus_speed = axis_focus if axis_focus is not None else diag_focus
    # A trace where the player barely moves does not reveal the movement model;
    # treating 0.0 as "measured 0 px/frame" would silently disable every horizon
    # feature, so below the floor it is a fallback with a named reason.
    if speed is not None and speed < MIN_MEASURED_SPEED:
        speed = None
    if focus_speed is not None and focus_speed < MIN_MEASURED_SPEED:
        focus_speed = None
    measured = {"speed": speed, "focus_speed": focus_speed}
    fallback = []
    if speed is None:
        measured["speed"] = 4.0
        fallback.append("speed")
    if focus_speed is None:
        measured["focus_speed"] = 2.0
        fallback.append("focus_speed")
    return {
        "speed": float(measured["speed"]),
        "focus_speed": float(measured["focus_speed"]),
        "speed_diag": float(measured["speed"]) / math.sqrt(2.0),
        "focus_speed_diag": float(measured["focus_speed"]) / math.sqrt(2.0),
        "samples_axis_normal": len(buckets[(0, 0)]),
        "samples_axis_focus": len(buckets[(0, 1)]),
        "samples_diag_normal": len(buckets[(1, 0)]),
        "samples_diag_focus": len(buckets[(1, 1)]),
        "fallback": fallback,
    }


def horizon_clearance(
    px: float,
    py: float,
    pos: np.ndarray,
    vel: np.ndarray,
    half: np.ndarray,
    steps: int = HORIZON_STEPS,
    frameskip: int = FRAMESKIP,
    disp: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-action min clearance along a constant-action rollout.

    Returns ``(mins, traj)`` with shapes ``(18,)`` and ``(18, steps, 2)``.  This
    is *not* the pilot's beam search: holding one action for the horizon is a
    strictly weaker planner, but it is the same information and it makes the 18
    action values independent of the pilot's tie-breaking.
    """
    step_disp = _DISP if disp is None else disp
    x_lo = spec.MOVE_AREA_X + PLAYER_HALF
    x_hi = spec.MOVE_AREA_X + spec.MOVE_AREA_W - PLAYER_HALF
    y_lo = spec.MOVE_AREA_Y + PLAYER_HALF
    y_hi = spec.MOVE_AREA_Y + spec.MOVE_AREA_H - PLAYER_HALF
    radius = half + PLAYER_HALF + BULLET_PAD
    cur = np.tile(np.array([px, py], dtype=np.float32), (18, 1))
    mins = np.full(18, CLR_CAP, dtype=np.float32)
    traj = np.zeros((18, steps, 2), dtype=np.float32)
    n = pos.shape[0]
    for step in range(1, steps + 1):
        cur[:, 0] = np.clip(cur[:, 0] + step_disp[:, 0], x_lo, x_hi)
        cur[:, 1] = np.clip(cur[:, 1] + step_disp[:, 1], y_lo, y_hi)
        traj[:, step - 1] = cur
        if n:
            t = float(step * frameskip)
            bx = pos[:, 0] + vel[:, 0] * t
            by = pos[:, 1] + vel[:, 1] * t
            d = np.sqrt((cur[:, 0:1] - bx[None, :]) ** 2 + (cur[:, 1:2] - by[None, :]) ** 2) - radius[None, :]
            mins = np.minimum(mins, d.min(axis=1))
    return mins.astype(np.float32), traj


def _path_density(start: tuple[int, int], direction: tuple[float, float], density: np.ndarray, samples: int = 8) -> float:
    """Mean danger density on the 8 samples between the player cell and its exit."""
    if direction == (0.0, 0.0):
        return 0.0
    gh, gw = density.shape
    total = 0.0
    for k in range(1, samples + 1):
        y = int(round(start[0] + direction[1] * k / samples))
        x = int(round(start[1] + direction[0] * k / samples))
        if 0 <= y < gh and 0 <= x < gw:
            total += float(density[y, x])
    return total / samples


# ---------------------------------------------------------------------------
# per-decision extraction
# ---------------------------------------------------------------------------


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _closest_bullets(
    pos: np.ndarray, vel: np.ndarray, half: np.ndarray, px: float, py: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Keep the ``MAX_BULLETS`` nearest bullets and return the kept indices.

    The index array is returned because the sidecar's ``slots`` must be permuted
    the same way; keeping them aligned is what makes spawn detection exact.
    """
    if pos.shape[0] <= MAX_BULLETS:
        return pos, vel, half, None
    d = (pos[:, 0] - px) ** 2 + (pos[:, 1] - py) ** 2
    order = np.argsort(d)[:MAX_BULLETS]
    return pos[order], vel[order], half[order], order


def extract_episode(
    records: list[dict],
    bullets,
    stride: int = 1,
    speed_mode: str = "config",
    strict_frames: bool = False,
) -> dict:
    """Build feature blocks, labels, and audit primitives for one episode.

    ``speed_mode`` selects the movement model used by the horizon features:
    ``"config"`` keeps the pilot's Border-Team constants (bit-identical to the
    D1 numbers), ``"measure"`` estimates them from this episode's own velocity
    field, which is required for non-Border characters (see ``measure_speeds``).

    ``strict_frames`` drops decisions whose sampled frame span is not exactly
    ``frameskip``.  Replay playback is *polled* and the recorder is marginally
    slower than 60 fps, so ~1.4% of D2 decisions span 3+ frames instead of 2.
    Those records are complete in every *field* sense (no ``schema_*_error``),
    only the continuity assertion fails; the flag exists so a report can show
    that the conclusions do not depend on them.
    """
    if speed_mode not in ("config", "measure"):
        raise ValueError(f"unknown speed_mode {speed_mode}")
    speeds = None if speed_mode == "config" else measure_speeds(records)
    disp = _DISP if speeds is None else _per_step_disp(FRAMESKIP, speeds)
    positions_all = bullets["positions"]
    velocities_all = bullets["velocities"]
    half_all = bullets["half"]
    offsets = bullets["offsets"]
    slots_all = bullets["slots"] if "slots" in bullets else None

    rows: list[dict[str, float]] = []
    prim: dict[str, list[float]] = {
        "decision": [],
        "frame": [],
        "clearance": [],
        "n_bullets": [],
        "danger8": [],
        "region20": [],
        "power": [],
        "bombs": [],
        "bombs_used": [],
        "boss_life": [],
        "px": [],
        "py": [],
        "item_collected": [],
        "death": [],
        "attributed_spawns": [],
        "attributed_spawns_known": [],
    }
    prev_slots: np.ndarray | None = None
    history_clear: deque[float] = deque(maxlen=4)
    history_bullets: deque[float] = deque(maxlen=4)
    history_py: deque[float] = deque(maxlen=4)
    history_power: deque[float] = deque(maxlen=4)

    for index, record in enumerate(records):
        if index % stride:
            continue
        if strict_frames:
            frame_pre, frame_post = record.get("frame_pre"), record.get("frame_post")
            if frame_pre is None or frame_post is None:
                continue
            if int(frame_post) - int(frame_pre) != FRAMESKIP:
                continue
        player = record.get("player") or {}
        px, py = player.get("x"), player.get("y")
        if px is None or py is None:
            continue
        px, py = float(px), float(py)
        start, end = int(offsets[index]), int(offsets[index + 1])
        pos = positions_all[start:end]
        vel = velocities_all[start:end]
        half = half_all[start:end]
        cur_slots = slots_all[start:end] if slots_all is not None else None
        if pos.shape[0] > 0:
            pos, vel, half, keep = _closest_bullets(pos, vel, half, px, py)
            if keep is not None and cur_slots is not None:
                cur_slots = cur_slots[keep]
        n_bullets = int(pos.shape[0])

        resources = record.get("resources") or {}
        clock = record.get("clock") or {}
        enemies = record.get("enemies") or []
        items = record.get("items") or []
        bosses = record.get("bosses") or []
        events = record.get("events") or []
        frame_events = record.get("frame_events") or {}
        shot = record.get("shot") or {}

        row: dict[str, float] = {}

        # --- BASE: the temporal proxies named by v3.6.0 §4 T1 ---------------
        clear0 = clearance_now(px, py, pos, half)
        mins, traj = horizon_clearance(px, py, pos, vel, half, disp=disp)
        boss_hp = float(sum(_finite(b.get("life")) for b in bosses)) if bosses else 0.0
        row["BASE_clearance"] = clear0
        row["BASE_n_bullets"] = float(n_bullets)
        row["BASE_stage_frames"] = _finite(clock.get("stage_frames"))
        row["BASE_power"] = _finite(resources.get("power"))
        row["BASE_boss_hp"] = boss_hp
        row["BASE_horizon_best"] = float(mins.max())
        row["BASE_horizon_worst"] = float(mins.min())
        row["BASE_horizon_mean"] = float(mins.mean())
        row["BASE_survivable_actions"] = float((mins > 0.0).sum())
        row["BASE_bullets_near_60"] = float(((pos[:, 0] - px) ** 2 + (pos[:, 1] - py) ** 2 < 3600).sum()) if n_bullets else 0.0
        row["BASE_bullets_near_120"] = float(((pos[:, 0] - px) ** 2 + (pos[:, 1] - py) ** 2 < 14400).sum()) if n_bullets else 0.0

        # --- REGION: multi-scale danger density + union region + exit -------
        maps: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for cell, _gw, _gh in SCALES:
            for t in TIMES:
                maps[(cell, t)] = occupancy(pos + vel * float(t), cell)
        pcell8 = player_cell(px, py, 8)
        pcell16 = player_cell(px, py, 16)
        for t in TIMES:
            occ8, den8 = maps[(8, t)]
            occ16, den16 = maps[(16, t)]
            row[f"REGION_occ8_t{t}"] = float(occ8.mean())
            row[f"REGION_den8_t{t}"] = float(den8[pcell8])
            row[f"REGION_occ16_t{t}"] = float(occ16.mean())
            row[f"REGION_dense16_t{t}"] = float((den16 >= 2.0).mean())
        occ20, den20 = maps[(8, 20)]
        occ40, _ = maps[(8, 40)]
        comp20 = component_from(occ20, pcell8)
        comp40 = component_from(occ40, pcell8)
        row["REGION_player_comp20"] = float(len(comp20))
        row["REGION_player_comp40"] = float(len(comp40))
        row["REGION_in_region20"] = 1.0 if len(comp20) >= MIN_REGION_CELLS else 0.0
        row["REGION_in_region40"] = 1.0 if len(comp40) >= MIN_REGION_CELLS else 0.0
        # Union/density component: what S1 was missing.
        dense8 = den20 >= 2.0
        union_cells = component_from(dense8, pcell8) if dense8[pcell8] else []
        row["REGION_union_comp"] = float(len(union_cells))
        row["REGION_union_frac"] = float(dense8.mean())
        if comp20 and comp40:
            c20 = (sum(c[1] for c in comp20) / len(comp20), sum(c[0] for c in comp20) / len(comp20))
            c40 = (sum(c[1] for c in comp40) / len(comp40), sum(c[0] for c in comp40) / len(comp40))
            row["REGION_expand_x"] = c40[0] - c20[0]
            row["REGION_expand_y"] = c40[1] - c20[1]
            row["REGION_expand_norm"] = math.hypot(c40[0] - c20[0], c40[1] - c20[1])
            row["REGION_centroid_x_rel"] = c20[0] - pcell8[1]
            row["REGION_centroid_y_rel"] = c20[1] - pcell8[0]
        else:
            row["REGION_expand_x"] = 0.0
            row["REGION_expand_y"] = 0.0
            row["REGION_expand_norm"] = 0.0
            row["REGION_centroid_x_rel"] = 0.0
            row["REGION_centroid_y_rel"] = 0.0
        ring, safe_area, ex_dx, ex_dy = exit_info(occ20, pcell8)
        row["REGION_exit_ring"] = float(ring)
        row["REGION_exit_area"] = float(safe_area)
        row["REGION_exit_dx"] = float(ex_dx)
        row["REGION_exit_dy"] = float(ex_dy)
        row["REGION_path_density"] = _path_density(pcell8, (ex_dx, ex_dy), den20)
        row["REGION_wall_left"] = float(px - spec.MOVE_AREA_X)
        row["REGION_wall_right"] = float(spec.MOVE_AREA_X + spec.MOVE_AREA_W - px)
        row["REGION_wall_bottom"] = float(spec.MOVE_AREA_Y + spec.MOVE_AREA_H - py)
        row["REGION_wall_min"] = float(min(px - spec.MOVE_AREA_X, spec.MOVE_AREA_X + spec.MOVE_AREA_W - px, spec.MOVE_AREA_Y + spec.MOVE_AREA_H - py))

        # --- LURE: aimed-bullet geometry ------------------------------------
        left_room = float(min(traj[a, :, 0].min() for a in range(9) if mins[a] > 0.0) - px) if (mins[:9] > 0).any() else 0.0
        right_room = float(max(traj[a, :, 0].max() for a in range(9) if mins[a] > 0.0) - px) if (mins[:9] > 0).any() else 0.0
        row["LURE_room_left"] = max(0.0, -left_room)
        row["LURE_room_right"] = max(0.0, right_room)
        if n_bullets:
            dx = px - pos[:, 0]
            dy = py - pos[:, 1]
            dist = np.sqrt(dx * dx + dy * dy) + 1e-6
            speed = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2) + 1e-6
            cosang = (dx * vel[:, 0] + dy * vel[:, 1]) / (dist * speed)
            aimed = cosang >= 0.985
            approaching = cosang > 0.0
            ttc = np.where(approaching, dist / speed, 1e6)
            in60 = aimed & (ttc <= 60.0)
            row["LURE_aimed_count"] = float(aimed.sum())
            row["LURE_aimed_60"] = float(in60.sum())
            row["LURE_aimed_ttc_min"] = float(min(60.0, ttc[in60].min())) if in60.any() else 60.0
            row["LURE_aimed_ttc_mean"] = float(min(60.0, ttc[in60].mean())) if in60.any() else 60.0
            if in60.any():
                ax = pos[in60, 0] + vel[in60, 0] * ttc[in60]
                row["LURE_aim_point_dx"] = float(ax.mean() - px)
                row["LURE_aim_left"] = float((ax < px).sum())
                row["LURE_aim_right"] = float((ax >= px).sum())
            else:
                row["LURE_aim_point_dx"] = 0.0
                row["LURE_aim_left"] = 0.0
                row["LURE_aim_right"] = 0.0
            cross = np.abs(vel[:, 1])
            row["LURE_crosstrack_rate"] = float(cross[approaching].mean()) if approaching.any() else 0.0
            row["LURE_approach_count"] = float(approaching.sum())
        else:
            for key in ("aimed_count", "aimed_60", "crosstrack_rate", "approach_count", "aim_left", "aim_right"):
                row[f"LURE_{key}"] = 0.0
            row["LURE_aimed_ttc_min"] = CLR_CAP
            row["LURE_aimed_ttc_mean"] = CLR_CAP
            row["LURE_aim_point_dx"] = 0.0
        row["LURE_corridor_20"] = float(safe_area)
        row["LURE_exit_progress"] = float(1.0 / (1.0 + ring))

        # --- ENEMY: predicted fire pressure --------------------------------
        n_enemies = len(enemies)
        row["ENEMY_count"] = float(n_enemies)
        if n_enemies:
            ex = np.array([_finite(e.get("x")) for e in enemies], dtype=np.float32)
            ey = np.array([_finite(e.get("y")) for e in enemies], dtype=np.float32)
            elife = np.array([_finite(e.get("life")) for e in enemies], dtype=np.float32)
            emax = np.array([max(1.0, _finite(e.get("max_life"), 1.0)) for e in enemies], dtype=np.float32)
            edist = np.sqrt((ex - px) ** 2 + (ey - py) ** 2)
            row["ENEMY_nearest_dist"] = float(edist.min())
            row["ENEMY_nearest_life"] = float(elife[int(edist.argmin())])
            row["ENEMY_life_sum"] = float(elife.sum())
            row["ENEMY_life_frac_mean"] = float((elife / emax).mean())
            row["ENEMY_near_120"] = float((edist < 120.0).sum())
            row["ENEMY_above"] = float((ey < py).sum())
            # Heuristic "predicted bullet generation": close, healthy, on-screen.
            row["ENEMY_pressure"] = float((np.sqrt(emax / 20.0) / (1.0 + edist / 100.0)).sum())
            row["ENEMY_column_threat"] = float((np.abs(ex - px) < 24.0).sum())
            row["ENEMY_boss_slots"] = float(sum(1 for e in enemies if e.get("boss")))
            row["ENEMY_item_drop"] = float(sum(max(0, int(_finite(e.get("item_drop"), -1.0))) for e in enemies))
            # v3.6.0 D3 columns.  On D1/D2 traces these keys are absent, so they
            # become constant 0 and - after centring - contribute nothing to the
            # ridge fit, which is what keeps those runs reproducible.
            row["ENEMY_damageable_count"] = float(sum(1 for e in enemies if e.get("damageable")))
            row["ENEMY_boss_damageable"] = float(any(e.get("damageable") for e in enemies if e.get("boss")))
            limits = []
            for e in enemies:
                phase_life = _finite(e.get("phase_starting_life"), 0.0)
                if phase_life > 0.0:
                    limits.append(max(0.0, min(1.0, _finite(e.get("life"), 0.0) / phase_life)))
            row["ENEMY_phase_frac_mean"] = float(np.mean(limits)) if limits else 0.0
            intervals = [_finite(e.get("shoot_interval_frames"), 0.0) for e in enemies]
            row["ENEMY_shoot_interval_mean"] = float(np.mean(intervals)) if intervals else 0.0
            row["ENEMY_unique_indices"] = float(
                len({int(_finite(e.get("enemy_index"), -1.0)) for e in enemies} - {-1})
            )
        else:
            for key in (
                "nearest_dist",
                "nearest_life",
                "life_sum",
                "life_frac_mean",
                "near_120",
                "above",
                "pressure",
                "column_threat",
                "boss_slots",
                "item_drop",
                "damageable_count",
                "boss_damageable",
                "phase_frac_mean",
                "shoot_interval_mean",
                "unique_indices",
            ):
                row[f"ENEMY_{key}"] = 0.0
            row["ENEMY_nearest_dist"] = 999.0

        # --- spawn attribution (v3.6.0 D3) ---------------------------------
        # Slot ids make "this bullet is new" exact (see reports/标签来源调查.md §1);
        # nearest-enemy within SPAWN_RADIUS attaches it to a shooter.  Both the
        # per-decision attributed count (a feature) and its forward window sum (the
        # ClearEnemy label) come from here.  Without slot ids the count is 0 and the
        # label is NaN, so old traces cannot silently produce a fake label.
        attributed = 0.0
        if cur_slots is None:
            # No slot ids in this sidecar: the label is unknown, not zero spawns.
            prim["attributed_spawns"].append(0.0)
            prim["attributed_spawns_known"].append(0.0)
        else:
            # Being known is a property of the sidecar, not of having a previous
            # snapshot: the first decision legitimately has no spawn to count but
            # still belongs to the label denominator.
            prim["attributed_spawns_known"].append(1.0)
            if prev_slots is not None:
                new_mask = spawns_by_slot(prev_slots, cur_slots)
                spawn_pos = pos[new_mask]
                if spawn_pos.shape[0]:
                    enemy_xy = np.array(
                        [[_finite(e.get("x")), _finite(e.get("y"))] for e in enemies], dtype=np.float32
                    ).reshape(-1, 2)
                    attributed = float(attribute_spawns(spawn_pos, enemy_xy, SPAWN_RADIUS)["counts"].sum())
            prim["attributed_spawns"].append(attributed)
        row["ENEMY_attributed_spawns"] = attributed
        prev_slots = None if cur_slots is None else cur_slots.copy()

        # --- POWER: collection opportunity ---------------------------------
        power = _finite(resources.get("power"))
        deficit = max(0.0, spec.POWER_MAX - power)
        p_items = [it for it in items if int(_finite(it.get("type"), -1)) in spec.ITEM_POWER_TYPES]
        row["POWER_power"] = power
        row["POWER_deficit"] = deficit
        row["POWER_deficit_frac"] = deficit / spec.POWER_MAX
        row["POWER_p_count"] = float(len(p_items))
        if p_items:
            ix = np.array([_finite(it.get("x")) for it in p_items], dtype=np.float32)
            iy = np.array([_finite(it.get("y")) for it in p_items], dtype=np.float32)
            idist = np.sqrt((ix - px) ** 2 + (iy - py) ** 2)
            row["POWER_p_nearest"] = float(idist.min())
            row["POWER_p_above_line"] = float((iy < 128.0).sum())
            row["POWER_p_onscreen"] = float(sum(1 for it in p_items if it.get("onscreen")))
            row["POWER_p_max_value"] = float(sum(1 for it in p_items if it.get("is_max_value")))
            row["POWER_p_mean_y"] = float(iy.mean())
        else:
            for key in ("p_nearest", "p_above_line", "p_onscreen", "p_max_value", "p_mean_y"):
                row[f"POWER_{key}"] = 0.0
            row["POWER_p_nearest"] = 999.0
        line = _finite(shot.get("point_item_value_line"), 128.0)
        row["POWER_y_to_line"] = float(py - line)
        row["POWER_up_clearance"] = float(mins[1])
        if len(prim["power"]) > 0:
            row["POWER_rate"] = power - prim["power"][-1]
        else:
            row["POWER_rate"] = 0.0

        # --- BOSS: damage window -------------------------------------------
        if bosses:
            bx = np.array([_finite(b.get("x")) for b in bosses], dtype=np.float32)
            row["BOSS_present"] = 1.0
            row["BOSS_dx"] = float(np.abs(bx - px).min())
            row["BOSS_life"] = boss_hp
            row["BOSS_life_log"] = float(math.log1p(max(0.0, boss_hp)))
            row["BOSS_count"] = float(len(bosses))
            # ``clock.current_spell`` used to sit here.  It is a dead field (0 in
            # D1, -1 in D2): the game does not populate currentSpellCardNumber
            # during normal play, and the old 4-byte read also leaked the retry
            # menu into it.  ``flags1.damageable`` and ``phase_starting_life`` are
            # the real invulnerability/phase source (EnemyManager.hpp C_ASSERTs).
            row["BOSS_damageable"] = float(any(b.get("damageable") for b in bosses))
            row["BOSS_accepts_damage"] = float(any(b.get("accepts_damage") for b in bosses))
            phase_limit = 0.0
            for b in bosses:
                phase_life = _finite(b.get("phase_starting_life"), 0.0)
                if phase_life > 0.0:
                    phase_limit = max(phase_limit, max(0.0, min(1.0, _finite(b.get("life"), 0.0) / phase_life)))
            row["BOSS_phase_frac"] = phase_limit
            row["BOSS_shoot_interval"] = float(
                np.mean([_finite(b.get("shoot_interval_frames"), 0.0) for b in bosses])
            )
            aligned = np.abs(traj[:, -1, 0] - bx.min()) < 8.0
            safe = mins > 12.0
            row["BOSS_align_safe"] = 1.0 if (aligned & safe).any() else 0.0
            row["BOSS_align_safe_margin"] = float(mins[aligned].max()) if aligned.any() else 0.0
            row["BOSS_damage_window"] = 1.0 if clear0 > 20.0 else 0.0
        else:
            for key in (
                "dx",
                "life",
                "life_log",
                "count",
                "damageable",
                "accepts_damage",
                "phase_frac",
                "shoot_interval",
                "align_safe",
                "align_safe_margin",
                "damage_window",
            ):
                row[f"BOSS_{key}"] = 0.0
            row["BOSS_present"] = 0.0

        # --- EMERGENCY: bomb / resource ------------------------------------
        bombs = _finite(resources.get("bombs"))
        lives = _finite(resources.get("lives"))
        row["EMERGENCY_bombs"] = bombs
        row["EMERGENCY_lives"] = lives
        row["EMERGENCY_routes"] = float((mins > 0.0).sum())
        row["EMERGENCY_all_blocked"] = 1.0 if float(mins.max()) < 0.0 else 0.0
        row["EMERGENCY_best_route"] = float(mins.max())
        row["EMERGENCY_player_state"] = _finite(player.get("state"))
        row["EMERGENCY_deaths"] = _finite(resources.get("deaths"))

        # --- HISTORY: temporal encoder -------------------------------------
        hist_clear = list(history_clear) or [clear0]
        hist_bullets = list(history_bullets) or [float(n_bullets)]
        hist_py = list(history_py) or [py]
        hist_power = list(history_power) or [power]
        for k in range(4):
            back = -(k + 1)
            row[f"HISTORY_clear{ k + 1 }"] = float(hist_clear[back]) if len(hist_clear) >= k + 1 else clear0
            row[f"HISTORY_bullets{ k + 1 }"] = float(hist_bullets[back]) if len(hist_bullets) >= k + 1 else float(n_bullets)
            row[f"HISTORY_py{ k + 1 }"] = float(hist_py[back]) if len(hist_py) >= k + 1 else py
            row[f"HISTORY_power{ k + 1 }"] = float(hist_power[back]) if len(hist_power) >= k + 1 else power
        row["HISTORY_d_clear"] = clear0 - row["HISTORY_clear1"]
        row["HISTORY_d_bullets"] = float(n_bullets) - row["HISTORY_bullets1"]
        row["HISTORY_d_py"] = py - row["HISTORY_py1"]
        row["HISTORY_d_power"] = power - row["HISTORY_power1"]
        history_clear.append(clear0)
        history_bullets.append(float(n_bullets))
        history_py.append(py)
        history_power.append(power)

        rows.append(row)
        prim["decision"].append(float(record.get("decision", index + 1)))
        prim["frame"].append(float(record.get("frame_post", 0)))
        prim["clearance"].append(clear0)
        prim["n_bullets"].append(float(n_bullets))
        prim["danger8"].append(float(maps[(8, 0)][0].mean()))
        prim["region20"].append(float(len(comp20)))
        prim["power"].append(power)
        prim["bombs"].append(bombs)
        prim["bombs_used"].append(_finite(resources.get("bombs_used")))
        prim["boss_life"].append(boss_hp if bosses else float("nan"))
        prim["px"].append(px)
        prim["py"].append(py)
        prim["item_collected"].append(1.0 if frame_events.get("item_collected") else 0.0)
        prim["death"].append(1.0 if "death" in events else 0.0)

    columns: dict[str, list[str]] = {}
    for name in BLOCKS:
        prefix = name + "_"
        columns[name] = sorted({key for row in rows for key in row if key.startswith(prefix)})
    blocks = {
        name: np.array([[row.get(key, 0.0) for key in columns[name]] for row in rows], dtype=np.float32)
        if rows
        else np.zeros((0, len(columns[name])), dtype=np.float32)
        for name in BLOCKS
    }
    prim_arr = {key: np.array(value, dtype=np.float64) for key, value in prim.items()}
    labels = build_labels(prim_arr)
    return {
        "blocks": blocks,
        "columns": columns,
        "labels": labels,
        "primitives": prim_arr,
        "rows": len(rows),
        "speeds": speeds,
        "speed_mode": speed_mode,
    }


def _lookahead_min(values: np.ndarray, horizon: int) -> np.ndarray:
    n = values.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        if i + 1 + horizon > n:
            continue
        out[i] = values[i + 1 : i + 1 + horizon].min()
    return out


def _lookahead_any(flags: np.ndarray, horizon: int) -> np.ndarray:
    n = flags.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        if i + 1 + horizon > n:
            continue
        out[i] = 1.0 if flags[i + 1 : i + 1 + horizon].max() > 0.0 else 0.0
    return out


def _lookahead_delta(values: np.ndarray, horizon: int) -> np.ndarray:
    n = values.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        j = i + horizon
        if j >= n:
            continue
        out[i] = values[j] - values[i]
    return out


def _lookahead_point(px: np.ndarray, py: np.ndarray, horizon: int) -> np.ndarray:
    """(Unused this round) 9-way bucket of the net displacement over ``horizon``.

    Kept because the T1 gate "exit direction top-3 >= 70%" needs it, but it is
    *not* wired into a cell: the only ground truth it can have offline is "where
    the recorded player actually went", which is a behaviour label, not an exit
    quality label.  Deferred to T2 where the teacher emits ``region_exit_direction``
    itself (contract §2.2).
    """
    n = px.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        j = i + horizon
        if j >= n:
            continue
        dx, dy = px[j] - px[i], py[j] - py[i]
        if math.hypot(dx, dy) < 2.0:
            out[i] = 0.0
            continue
        best, best_dot = 0, -2.0
        norm = math.hypot(dx, dy)
        for k, (ux, uy) in enumerate(_MOVE_VECTORS):
            if k == 0:
                continue
            nk = math.hypot(ux, uy)
            dot = (dx * ux + dy * uy) / (norm * nk)
            if dot > best_dot:
                best, best_dot = k, dot
        out[i] = float(best)
    return out


def _lookahead_sum(values: np.ndarray, horizon: int) -> np.ndarray:
    """Sum of ``values[i+1 : i+1+horizon]`` with NaN when the window is short."""
    n = values.shape[0]
    out = np.full(n, np.nan)
    if n == 0:
        return out
    csum = np.concatenate([[0.0], np.cumsum(np.asarray(values, dtype=np.float64))])
    for i in range(n):
        end = i + 1 + horizon
        if end > n:
            continue
        out[i] = csum[end] - csum[i + 1]
    return out


def _lookahead_at(values: np.ndarray, horizon: int) -> np.ndarray:
    """Value at ``i + horizon`` with NaN when the episode ends first."""
    n = values.shape[0]
    out = np.full(n, np.nan)
    if n > horizon:
        out[: n - horizon] = values[horizon:]
    return out


def build_labels(prim: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """All labels are read from the recorded future; see 执行计划 §4."""
    death = prim["death"]
    power = prim["power"]
    bombs = prim["bombs"]
    bombs_used = prim["bombs_used"]
    n_bullets = prim["n_bullets"]
    boss_life = prim["boss_life"]
    labels: dict[str, np.ndarray] = {}
    labels["death_60"] = _lookahead_any(death, S)
    labels["death_240"] = _lookahead_any(death, M)
    labels["min_clearance_60"] = _lookahead_min(prim["clearance"], S)
    labels["min_clearance_240"] = _lookahead_min(prim["clearance"], M)
    # Point lookups at i+S, exactly as pre-registered in 执行计划 §4.
    danger_at = _lookahead_at(prim["danger8"], S)
    labels["danger_60"] = danger_at
    region_at = _lookahead_at(prim["region20"], S)
    labels["in_region_60"] = np.where(np.isnan(region_at), np.nan, (np.nan_to_num(region_at) >= MIN_REGION_CELLS).astype(float))
    labels["bullet_delta_60"] = _lookahead_delta(n_bullets, S)
    labels["bullet_delta_240"] = _lookahead_delta(n_bullets, M)
    labels["power_delta_60"] = _lookahead_delta(power, S)
    labels["power_delta_240"] = _lookahead_delta(power, M)
    labels["p_collected_60"] = _lookahead_any(prim["item_collected"], S)
    delta_bomb = np.zeros_like(bombs_used)
    delta_bomb[1:] = bombs_used[1:] - bombs_used[:-1]
    labels["bomb_used_60"] = _lookahead_any(delta_bomb, S)
    labels["boss_life_delta_60"] = -_lookahead_delta(boss_life, S)
    labels["boss_life_delta_240"] = -_lookahead_delta(boss_life, M)
    death60 = labels["death_60"]
    labels["bomb_needed_60"] = np.where(np.isnan(death60), np.nan, (np.nan_to_num(death60) > 0.0) & (bombs > 0.0))
    # Attributed spawns exist only where the sidecar carried slot ids; elsewhere the
    # label stays NaN instead of masquerading as "zero spawns".
    known = prim.get("attributed_spawns_known")
    if known is not None and float(np.nansum(known)) > 0.0:
        spawn_sum = _lookahead_sum(prim["attributed_spawns"], S)
        labels["attributed_spawns_60"] = np.where(known > 0.0, spawn_sum, np.nan)
    else:
        labels["attributed_spawns_60"] = np.full(prim["power"].shape[0], np.nan)
    return labels


# ---------------------------------------------------------------------------
# dataset loading / caching
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    blocks: dict[str, np.ndarray]
    columns: dict[str, list[str]]
    labels: dict[str, np.ndarray]
    groups: np.ndarray  # episode index per row
    episodes: list[str]
    primitives: dict[str, np.ndarray] = field(default_factory=dict)
    speeds: list[dict | None] = field(default_factory=list)
    speed_mode: str = "config"

    @property
    def rows(self) -> int:
        return int(self.groups.shape[0])


def load_dataset(
    probe_dir: Path,
    stride: int = 1,
    cache: Path | None = None,
    speed_mode: str = "config",
    strict_frames: bool = False,
) -> Dataset:
    probe_dir = Path(probe_dir)
    if cache is not None and cache.exists():
        payload = np.load(cache, allow_pickle=True)
        cached_mode = str(payload["speed_mode"]) if "speed_mode" in payload else "config"
        cached_strict = bool(payload["strict_frames"]) if "strict_frames" in payload else False
        if cached_mode != speed_mode or cached_strict != strict_frames:
            raise ValueError(
                f"feature cache {cache} was built with speed_mode={cached_mode!r} strict_frames={cached_strict}, "
                f"requested speed_mode={speed_mode!r} strict_frames={strict_frames}; use a different --cache path"
            )
        return Dataset(
            blocks={name: payload[f"block_{name}"] for name in BLOCKS},
            columns={name: list(payload[f"cols_{name}"]) for name in BLOCKS},
            labels={key: payload[f"label_{key}"] for key in LABELS},
            groups=payload["groups"],
            episodes=[str(x) for x in payload["episodes"]],
            primitives={key: payload[f"prim_{key}"] for key in payload["prim_keys"]},
            speeds=[json.loads(str(x)) for x in payload["speeds"]] if "speeds" in payload else [],
            speed_mode=cached_mode,
        )
    episodes_meta = read_episodes(probe_dir)
    names = [ep.get("episode") for ep in episodes_meta]
    blocks_acc: dict[str, list[np.ndarray]] = {name: [] for name in BLOCKS}
    labels_acc: dict[str, list[np.ndarray]] = {key: [] for key in LABELS}
    prim_acc: dict[str, list[np.ndarray]] = {}
    groups: list[np.ndarray] = []
    columns: dict[str, list[str]] = {}
    speeds_acc: list[dict | None] = []
    for index, name in enumerate(names):
        records = read_trace(probe_dir / "trace" / f"{name}.jsonl")
        bullets = np.load(probe_dir / "bullets" / f"{name}.npz")
        payload = extract_episode(
            records, bullets, stride=stride, speed_mode=speed_mode, strict_frames=strict_frames
        )
        for block in BLOCKS:
            blocks_acc[block].append(payload["blocks"][block])
            columns.setdefault(block, payload["columns"][block])
        for key in LABELS:
            labels_acc[key].append(payload["labels"][key])
        for key, value in payload["primitives"].items():
            prim_acc.setdefault(key, []).append(value)
        speeds_acc.append(payload["speeds"])
        groups.append(np.full(payload["rows"], index, dtype=np.int64))
    dataset = Dataset(
        blocks={name: np.concatenate(values, axis=0) for name, values in blocks_acc.items()},
        columns=columns,
        labels={key: np.concatenate(values, axis=0) for key, values in labels_acc.items()},
        groups=np.concatenate(groups, axis=0),
        episodes=names,
        primitives={key: np.concatenate(values, axis=0) for key, values in prim_acc.items()},
        speeds=speeds_acc,
        speed_mode=speed_mode,
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "groups": dataset.groups,
            "episodes": np.array(names, dtype=object),
            "speed_mode": np.array(speed_mode, dtype=object),
            "strict_frames": np.array(strict_frames, dtype=bool),
            "speeds": np.array([json.dumps(speed) for speed in speeds_acc], dtype=object),
        }
        for name in BLOCKS:
            arrays[f"block_{name}"] = dataset.blocks[name]
            arrays[f"cols_{name}"] = np.array(columns[name], dtype=object)
        for key in LABELS:
            arrays[f"label_{key}"] = dataset.labels[key]
        for key, value in dataset.primitives.items():
            arrays[f"prim_{key}"] = value
        arrays["prim_keys"] = np.array(list(dataset.primitives.keys()), dtype=object)
        np.savez(cache, **arrays)
    return dataset


# ---------------------------------------------------------------------------
# models: ridge with leave-one-episode-out via Gram subtraction
# ---------------------------------------------------------------------------


def _prepare_fold(X: np.ndarray, train: np.ndarray, test: np.ndarray, all_finite: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Standardise using train-fold statistics only (median-impute if needed).

    The S0 feature blocks contain no NaN by construction (every field falls back
    to an explicit ``unknown`` default), so the nanmedian pass - ~50 ms per fold,
    which would dominate a 200-permutation test - is skipped when the matrix is
    known to be finite.  The check is done once per ``loeo_predict`` call and
    passed in; it is never cached by object identity.
    """
    if all_finite:
        Xtr = X[train]
        Xte = X[test]
    else:
        med = np.nanmedian(np.where(np.isfinite(X[train]), X[train], np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(X[train]), X[train], med)
        Xte = np.where(np.isfinite(X[test]), X[test], med)
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return (Xtr - mean) / std, (Xte - mean) / std


def _ridge_gram(X: np.ndarray, y: np.ndarray, groups: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray:
    """LOEO ridge predictions, one episode held out at a time.

    Standardisation is per fold (train statistics only).  The Gram matrix is
    rebuilt per fold because standardisation changes it; at ~150 MFLOP per fold
    this is still what makes a 200-permutation test affordable.
    """
    n, p = X.shape
    pred = np.full(n, np.nan)
    all_finite = bool(np.isfinite(X).all())
    for g in np.unique(groups):
        te = groups == g
        if te.all():
            continue
        tr = ~te
        Xtr, Xte = _prepare_fold(X.astype(np.float64), tr, te, all_finite)
        ytr = y[tr]
        finite = np.isfinite(ytr)
        if not finite.any():
            pred[te] = 0.0
            continue
        Xf = Xtr[finite]
        yf = ytr[finite]
        gram = Xf.T @ Xf + lam * np.eye(p)
        rhs = Xf.T @ yf
        try:
            weight = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            weight = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        pred[te] = Xte @ weight + float(yf.mean()) - float(Xf.mean(axis=0) @ weight)
    return pred


def loeo_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray:
    return _ridge_gram(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float64), groups, lam)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return None
    x = a[mask]
    y = b[mask]
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    rx = _rank(x)
    ry = _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    if denom <= 0.0:
        return None
    return float((rx * ry).sum() / denom)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def binary_auc(y: np.ndarray, pred: np.ndarray) -> float | None:
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() < 10:
        return None
    labels = y[mask] > 0.5
    if labels.all() or not labels.any():
        return None
    return rank_auc(pred[mask], labels)


def brier(y: np.ndarray, pred: np.ndarray) -> float | None:
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() < 10:
        return None
    clipped = np.clip(pred[mask], 0.0, 1.0)
    return float(np.mean((clipped - (y[mask] > 0.5)) ** 2))


def topk_accuracy(y: np.ndarray, scores: np.ndarray, k: int = 3) -> float | None:
    mask = np.isfinite(y)
    if mask.sum() < 10:
        return None
    yy = y[mask].astype(int)
    ss = scores[mask]
    top = np.argsort(-ss, axis=1)[:, :k]
    return float((top == yy[:, None]).any(axis=1).mean())


def metric(kind: str, y: np.ndarray, pred: np.ndarray) -> float | None:
    if kind == "auc":
        return binary_auc(y, pred)
    if kind == "spearman":
        mask = np.isfinite(y)
        return spearman(pred[mask], y[mask])
    if kind == "brier":
        return brier(y, pred)
    raise ValueError(f"unknown metric {kind}")


# ---------------------------------------------------------------------------
# ablation driver
# ---------------------------------------------------------------------------


def _stack(blocks: dict[str, np.ndarray], used: Iterable[str]) -> np.ndarray:
    names = [name for name in used if name in blocks and blocks[name].shape[1] > 0]
    if not names:
        return np.zeros((next(iter(blocks.values())).shape[0], 1), dtype=np.float32)
    return np.hstack([blocks[name] for name in names])


def per_episode_metric(kind: str, y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> tuple[float | None, int]:
    """Macro metric: score each episode separately, then average.

    Pooled out-of-fold AUC mixes per-fold base rates: a constant predictor scores
    0.317 (not 0.5) on the D1 ``death_60`` label because the death-heavy episodes
    are exactly the folds whose training set has a lower prevalence.  The macro
    version cannot do that - a constant predictor scores 0.5 inside every
    episode - so it is reported as a sensitivity column.  The pre-registered gate
    still uses the pooled value; see the report's diagnostics section.
    """
    values = []
    for g in np.unique(groups):
        mask = groups == g
        value = metric(kind, y[mask], pred[mask])
        if value is not None:
            values.append(float(value))
    return (float(np.mean(values)) if values else None), len(values)


def _cell_value(dataset: Dataset, used: list[str], label: str) -> dict:
    """Out-of-fold prediction quality for one block set and one label."""
    kind = LABELS[label]
    X = _stack(dataset.blocks, used)
    y = dataset.labels[label]
    pred = loeo_predict(X, y, dataset.groups)
    ok = bool(np.isfinite(pred).any())
    value = metric(kind, y, pred) if ok else None
    value_ep, episodes_scored = per_episode_metric(kind, y, pred, dataset.groups) if ok else (None, 0)
    positives = None
    if kind == "auc" and ok:
        finite = np.isfinite(y) & np.isfinite(pred)
        positives = int((y[finite] > 0.5).sum())
    return {
        "value": value,
        "value_ep": value_ep,
        "episodes_scored": episodes_scored,
        "brier": brier(y, pred) if ok else None,
        "n_samples": int((np.isfinite(y) & np.isfinite(pred)).sum()) if ok else 0,
        "positives": positives,
        "n_features": int(X.shape[1]),
        "kind": kind,
    }


def _permutation_p(
    dataset: Dataset,
    base_blocks: list[str],
    block: str,
    label: str,
    kind: str,
    delta: float,
    without: float,
    permutations: int,
    rng: np.random.Generator,
) -> dict:
    """Within-episode column permutation of the ablated block.

    Shuffling the block's columns *inside each episode* destroys their alignment
    with the label while keeping the marginal distribution, the episode identity
    and every other block intact.  The p-value is the fraction of null deltas at
    least as large as the observed one (执行计划 §5).

    A null with zero spread carries no information: it happens when the ablated
    block is nearly constant inside an episode, and then *any* positive delta
    scores p = 1/(N+1).  Those cells are reported with ``null_degenerate=True``
    and the effect size, not the p-value, is the gate.
    """
    X = _stack(dataset.blocks, base_blocks + [block])
    y = dataset.labels[label]
    groups = dataset.groups
    first = sum(dataset.blocks[name].shape[1] for name in base_blocks)
    columns = slice(first, first + dataset.blocks[block].shape[1])
    null: list[float] = []
    for _ in range(permutations):
        Xp = X.copy()
        for g in np.unique(groups):
            rows = np.nonzero(groups == g)[0]
            Xp[rows, columns] = X[rows[rng.permutation(rows.size)], columns]
        value = metric(kind, y, loeo_predict(Xp, y, groups))
        if value is not None:
            null.append(float(value - without))
    if not null:
        return {"p": None, "null_mean": None, "null_std": None, "null_min": None, "null_max": None, "degenerate": None}
    values = np.asarray(null, dtype=np.float64)
    std = float(values.std())
    return {
        "p": float((int((values >= delta).sum()) + 1) / (values.size + 1)),
        "null_mean": float(values.mean()),
        "null_std": std,
        "null_min": float(values.min()),
        "null_max": float(values.max()),
        "degenerate": bool(std <= NULL_STD_EPS),
    }


def run_ablation(
    dataset: Dataset,
    permutations: int = PERMUTATIONS,
    seed: int = PERM_SEED,
    cells_spec: tuple[dict[str, Any], ...] | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    cells = []
    for spec_cell in (CELLS if cells_spec is None else cells_spec):
        strategy = spec_cell["strategy"]
        block = spec_cell["block"]
        base_blocks: list[str] = [] if strategy == "LocalDodge" else ["BASE"]
        roles: dict[str, str] = {spec_cell["primary"][0]: "primary"}
        for label, _metric_name, _threshold, _combine in spec_cell["increments"]:
            roles.setdefault(label, "increment")
        for label, _metric_name, _threshold in spec_cell.get("extras", ()):
            roles.setdefault(label, "extra")
        for label, role in roles.items():
            without_cell = _cell_value(dataset, base_blocks, label)
            with_cell = _cell_value(dataset, base_blocks + [block], label)
            delta = None
            if with_cell["value"] is not None and without_cell["value"] is not None:
                delta = with_cell["value"] - without_cell["value"]
            delta_ep = None
            if with_cell["value_ep"] is not None and without_cell["value_ep"] is not None:
                delta_ep = with_cell["value_ep"] - without_cell["value_ep"]
            perm: dict = {}
            if permutations and delta is not None and with_cell["n_features"] > 0:
                perm = _permutation_p(
                    dataset,
                    base_blocks,
                    block,
                    label,
                    LABELS[label],
                    delta,
                    without_cell["value"],
                    permutations,
                    rng,
                )
            cells.append(
                {
                    "owner": strategy,
                    "strategy": strategy if role == "primary" else f"{strategy}::{label}",
                    "role": role,
                    "block": block,
                    "label": label,
                    "kind": LABELS[label],
                    "direction": "add-one-in",
                    "base_blocks": base_blocks,
                    "without": without_cell["value"],
                    "with": with_cell["value"],
                    "without_ep": without_cell["value_ep"],
                    "with_ep": with_cell["value_ep"],
                    "delta": delta,
                    "delta_ep": delta_ep,
                    "permutation_p": perm.get("p"),
                    "permutation_null_mean": perm.get("null_mean"),
                    "permutation_null_std": perm.get("null_std"),
                    "null_degenerate": perm.get("degenerate"),
                    "permutations": permutations,
                    "n_samples": with_cell["n_samples"],
                    "positives": with_cell["positives"],
                    "brier_with": with_cell["brier"],
                    "brier_without": without_cell["brier"],
                }
            )
    return {"cells": cells}


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


def judge(
    cells: list[dict],
    all_death_cell: dict | None = None,
    cells_spec: tuple[dict[str, Any], ...] | None = None,
) -> dict:
    """Apply 执行计划 §6 to the cell table.

    The gate a number is judged on is part of the pre-registration, so this
    function reads the label from the (possibly fallback-resolved) cell spec and
    never from the outcome.
    """
    spec = CELLS if cells_spec is None else cells_spec
    index = {(cell["owner"], cell["label"]): cell for cell in cells}
    verdicts: dict[str, dict] = {}
    for spec_cell in spec:
        strategy = spec_cell["strategy"]
        primary_label, absolute = spec_cell["primary"]
        cell = index.get((strategy, primary_label))
        if cell is None:
            verdicts[strategy] = {"verdict": "not_tested"}
            continue
        abs_pass = True
        gate_metric = None
        gate_threshold = None
        if absolute is not None:
            gate_metric, gate_threshold = absolute
            if gate_metric in ("auc", "spearman"):
                abs_pass = cell["with"] is not None and cell["with"] >= gate_threshold
            elif gate_metric == "brier":
                abs_pass = cell["brier_with"] is not None and cell["brier_with"] <= gate_threshold
            else:
                raise ValueError(f"unknown absolute gate metric {gate_metric}")
        branches = []
        for label, metric_name, threshold, combine in spec_cell["increments"]:
            inc = index[(strategy, label)]
            positive = inc["delta"] is not None and inc["delta"] >= threshold
            significant = inc["permutation_p"] is not None and inc["permutation_p"] < 0.05
            agrees = _sign_agrees(inc["delta"], inc["delta_ep"])
            # 执行计划 §9: a cell whose pooled and per-episode deltas point in
            # opposite directions is reported as unstable, not as a pass.  This
            # is a stop-line, not a threshold change: the pooled value is still
            # the pre-registered one and is still printed.
            macro_ok = inc["delta_ep"] is not None and inc["delta_ep"] >= threshold
            branches.append(
                {
                    "label": label,
                    "metric": metric_name,
                    "threshold": threshold,
                    "combine": combine,
                    "delta": inc["delta"],
                    "delta_ep": inc["delta_ep"],
                    "p": inc["permutation_p"],
                    "null_degenerate": inc["null_degenerate"],
                    "positive": bool(positive),
                    "macro_ok": bool(macro_ok),
                    "significant": bool(significant),
                    "sign_agrees": agrees,
                    "pass": bool(positive and macro_ok and significant and agrees is not False),
                }
            )
        if not branches:
            inc_pass = True
            combine = None
        else:
            combine = branches[0]["combine"]
            inc_pass = any(b["pass"] for b in branches) if combine == "or" else all(b["pass"] for b in branches)
        underpowered = bool(
            (cell["positives"] is not None and cell["positives"] < UNDERPOWERED_POSITIVES)
            or cell["n_samples"] < UNDERPOWERED_POSITIVES
        )
        unstable = any(b["sign_agrees"] is False for b in branches)
        if underpowered:
            verdict = "underpowered"
        elif abs_pass and inc_pass:
            verdict = "pass"
        elif unstable:
            verdict = "unstable"
        else:
            verdict = "fail"
        verdicts[strategy] = {
            "verdict": verdict,
            "stability": "unstable" if unstable else "stable",
            "primary_label": primary_label,
            "absolute_gate": absolute,
            "absolute_value": cell["with"],
            "absolute_value_ep": cell["with_ep"],
            "absolute_brier": cell["brier_with"],
            "absolute_pass": bool(abs_pass),
            "increment_combine": combine,
            "increments": branches,
            "increment_pass": bool(inc_pass),
            "n_samples": cell["n_samples"],
            "positives": cell["positives"],
            "pooled_vs_ep_agree": _sign_agrees(cell["delta"], cell["delta_ep"]),
        }
    passed = [name for name, row in verdicts.items() if row["verdict"] == "pass"]
    global_gates: dict[str, dict] = {
        "G-1": {"name": "pass 策略数 >= 3", "value": len(passed), "threshold": 3, "pass": len(passed) >= 3}
    }
    # G-2: every passing strategy's increment point estimate is positive AND p<0.05.
    g2_bad = []
    for name in passed:
        for branch in verdicts[name]["increments"]:
            if branch["delta"] is None or branch["delta"] <= 0 or not branch["significant"]:
                g2_bad.append(f"{name}:{branch['label']}")
    global_gates["G-2"] = {
        "name": "通过块的增量点估计为正且 p<0.05",
        "value": len(g2_bad),
        "threshold": 0,
        "pass": not g2_bad,
        "offenders": g2_bad,
    }
    # G-3: the union of every block must move death_60 by >= +0.05 over BASE.
    if all_death_cell is None:
        global_gates["G-3"] = {"name": "ALL vs BASE death_60 ΔAUC >= +0.05", "value": None, "threshold": 0.05, "pass": False}
    else:
        value = all_death_cell["delta"]
        global_gates["G-3"] = {
            "name": "ALL vs BASE death_60 ΔAUC >= +0.05",
            "value": value,
            "threshold": 0.05,
            "pass": value is not None and value >= 0.05,
        }
    # G-4: no cell whose permutation p contradicts the sign of its delta.
    contradictions = [
        cell["strategy"]
        for cell in cells
        if cell["delta"] is not None
        and cell["permutation_p"] is not None
        and ((cell["delta"] > 0 and cell["permutation_p"] > 0.5) or (cell["delta"] < 0 and cell["permutation_p"] < 0.5))
    ]
    global_gates["G-4"] = {
        "name": "无块的置换 p 与主结果符号相反",
        "value": len(contradictions),
        "threshold": 0,
        "pass": not contradictions,
        "offenders": contradictions,
    }
    return {
        "per_strategy": verdicts,
        "passed": passed,
        "gates": global_gates,
        "enters_T2": all(g["pass"] for g in global_gates.values()),
    }


def _sign_agrees(a: float | None, b: float | None) -> bool | None:
    if a is None or b is None:
        return None
    return bool((a >= 0) == (b >= 0))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    number = float(value)
    if number == int(number):
        return f"{int(number)}"
    fmt = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return fmt.format(number)


def render_markdown(report: dict) -> str:
    inv = report["inventory"]
    verdicts = report["verdicts"]
    lines = [
        "# v3.6.0 T1 消融主表（预注册协议）",
        "",
        f"- 生成：{report.get('generated')}",
        f"- 数据：`{report.get('source')}`（{inv['episodes']} 局 / {inv['rows']} 决策）",
        f"- 协议：LOEO {inv['episodes']} 折；岭回归 λ={RIDGE_LAMBDA:g}；局内列置换 {report.get('protocol', {}).get('permutations')} 次",
        f"- 总判定：**{'进入 T2' if verdicts['enters_T2'] else '不进入 T2'}**",
        "",
        "## 1 逐策略判定（预注册门禁）",
        "",
        "| 策略 | 主标签 | 绝对门 | 无块 | 有块 | 增量（门禁标签） | Δ | p | 判定 |",
        "|---|---|---|---:|---:|---|---:|---:|---|",
    ]
    for strategy, row in verdicts["per_strategy"].items():
        if row["verdict"] == "not_tested":
            lines.append(f"| {strategy} | - | - | - | - | - | - | - | not_tested |")
            continue
        gate = "n/a" if row["absolute_gate"] is None else f"{row['absolute_gate'][0]} {row['absolute_gate'][1]}"
        shown = _fmt(row["absolute_value"])
        if row["absolute_gate"] is not None and row["absolute_gate"][0] == "brier":
            gate = f"brier <= {row['absolute_gate'][1]} (实测 {_fmt(row['absolute_brier'])})"
        else:
            shown = _fmt(row["absolute_value"])
        branches = row["increments"]
        if branches:
            gate_label = " / ".join(f"{b['label']} {b['metric']} {b['threshold']:+.2f}" for b in branches)
            delta = " / ".join(_fmt(b["delta"], signed=True) for b in branches)
            p = " / ".join(_fmt(b["p"], 3) for b in branches)
            combine = row["increment_combine"]
        else:
            gate_label, delta, p, combine = "（对照）", "-", "-", None
        without = _fmt(next((c["without"] for c in report["cells"] if c["owner"] == strategy and c["role"] == "primary"), None))
        lines.append(
            f"| {strategy} | {row['primary_label']} | {gate} | {without} | {shown} | "
            f"{gate_label} ({combine or '-'}) | {delta} | {p} | **{row['verdict']}** |"
        )
    lines += ["", "## 2 全部单元格", "", "| 策略 | 标签 | 角色 | 指标 | 无块 | 有块 | Δ | 局均 Δ | p | 零散布 | n | 阳性 |", "|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    for cell in report["cells"]:
        lines.append(
            f"| {cell['strategy']} | {cell['label']} | {cell['role']} | {cell['kind']} | {_fmt(cell['without'])} | "
            f"{_fmt(cell['with'])} | {_fmt(cell['delta'], signed=True)} | {_fmt(cell['delta_ep'], signed=True)} | "
            f"{_fmt(cell['permutation_p'], 3)} | {_fmt(cell['null_degenerate'])} | {cell['n_samples']} | "
            f"{cell['positives'] if cell['positives'] is not None else '-'} |"
        )
    lines += ["", "## 3 全局门", ""]
    for name, gate in verdicts["gates"].items():
        value = gate.get("value")
        lines.append(f"- **{name}** {gate['name']}：{_fmt(value)}（阈值 {gate['threshold']}）→ {'通过' if gate['pass'] else '未过'}")
        if gate.get("offenders"):
            lines.append(f"  - offenders: `{gate['offenders']}`")
    global_cell = report.get("global_cells", {}).get("ALL_vs_BASE_death_60")
    if global_cell:
        lines += [
            "",
            "### ALL（八块全上）vs BASE 的 death_60",
            "",
            f"- 无块 AUC {_fmt(global_cell['without'])}（局均 {_fmt(global_cell['without_ep'])}）"
            f" → 有块 {_fmt(global_cell['with'])}（局均 {_fmt(global_cell['with_ep'])}）",
            f"- Δ = {_fmt(global_cell['delta'], signed=True)}（局均 {_fmt(global_cell['delta_ep'], signed=True)}），p = {_fmt(global_cell['permutation_p'], 3)}",
        ]
    diag = report.get("diagnostics", {})
    fallback = report.get("survival_fallback") or {}
    lines += ["", "## 4 生存标签与标签平衡", ""]
    if fallback.get("fallback"):
        lines.append(
            f"- **生存标签回退已触发**：`{fallback['label']}` 阳性 {fallback['positives']} < 100，"
            f"按执行计划-D2补遗 §3 改用 `{fallback['fallback_label']}`（绝对门 {fallback['fallback_absolute']}，"
            f"增量门 +{fallback['fallback_increment']}）。阈值方向未变，只换了判生存的标签。"
        )
    else:
        lines.append(
            f"- 生存标签 `{fallback.get('label')}` 阳性 {fallback.get('positives')} ≥ 100，按原门禁用 AUC。"
        )
    lines += ["", "| 二值标签 | 阳性 | 样本 | 基准率 | 备注 |", "|---|---:|---:|---:|---|"]
    rates = report.get("label_rates", {})
    for label, row in rates.items():
        note = "⚠ 近常量，AUC 不可与常规标签并列" if label in diag.get("degenerate_labels", []) else ""
        lines.append(f"| {label} | {row['positives']} | {row['n']} | {row['rate']:.4f} | {note} |")
    lines += [
        "",
        "## 5 诊断（读表前必读）",
        "",
        f"- **退化为常数预测器时的 pooled 指标**：生存标签空特征 = {_fmt(diag.get('degenerate_auc'))}"
        f"（局均 {_fmt(diag.get('degenerate_auc_ep'))}）。",
        "  pooled LOEO AUC 会把『每折基准率不同』混进排序：死亡集中的短局恰好是训练集基准率更低的折，"
        "常数预测器因此拿到 0.317 而不是 0.5。所有 Δ 都在同一协议下计算、两侧同受影响，所以**方向可信**；"
        "但 `death_60` 的绝对 AUC 不要读成『泛化到新局的 AUC』。局均列是该偏差的敏感性对照：常数预测器在局均口径下正好 0.5。",
        f"- **零散布置换**：{diag.get('degenerate_cells', [])}；这些块的边缘分布在局内近乎常数，"
        "打乱后 Δ 恒为 0，p 会退化成 1/(N+1)，**不可当显著证据**，只用效应量判。",
        f"- **门禁映射修正**：首版代码把增量门错挂在主标签上（REGION 用 `in_region_60` 自评），"
        "已改为按执行计划 §6.1 的门禁标签（REGION 用 `death_60`）；阈值未动。",
        f"- **pooled / 局均符号矛盾（unstable）**：{diag.get('unstable_cells', [])}。"
        "这些单元格 pooled 与局均指向相反方向，按执行计划 §9 记 `unstable`，不判 pass、不进 T2；"
        "两个数值都保留在第 2 节，不做取舍。",
        "- **置换 p 的饱和**：200 次置换落在 (1/201, 1]，所以强效应普遍显示 `0.005`。"
        "该数值只说明『置换后没出现同样大的 Δ』，不是效应量；门槛判定请以 Δ 为准，p 只作显著性下限。",
    ]
    pooled_ep = [c for c in report["cells"] if c.get("delta") is not None and c.get("delta_ep") is not None]
    if pooled_ep:
        same = sum(1 for c in pooled_ep if _sign_agrees(c["delta"], c["delta_ep"]))
        lines.append(f"- **pooled / 局均符号一致**：{same}/{len(pooled_ep)} 个单元格。不一致的单元格已在第 2 节可见，需单独读。")
    lines += ["", "## 6 数据清点", "", "```json", json.dumps(inv, ensure_ascii=False, indent=1), "```"]
    return "\n".join(lines) + "\n"


def inventory(dataset: Dataset) -> dict:
    rows = []
    for index, name in enumerate(dataset.episodes):
        mask = dataset.groups == index
        rows.append(
            {
                "episode": name,
                "split": stable_split(name),
                "decisions": int(mask.sum()),
                "deaths": int(dataset.primitives["death"][mask].sum()) if "death" in dataset.primitives else None,
                "death_60_positives": int(np.nansum(dataset.labels["death_60"][mask] > 0.5)),
                "bomb_used_60_positives": int(np.nansum(dataset.labels["bomb_used_60"][mask] > 0.5)),
                "p_collected_60_positives": int(np.nansum(dataset.labels["p_collected_60"][mask] > 0.5)),
                "boss_present_decisions": int(np.isfinite(dataset.labels["boss_life_delta_60"][mask]).sum()),
            }
        )
    return {
        "episodes": len(dataset.episodes),
        "rows": dataset.rows,
        "blocks": {name: int(dataset.blocks[name].shape[1]) for name in BLOCKS},
        "columns": {name: list(dataset.columns[name]) for name in BLOCKS},
        "labels": {key: value for key, value in LABELS.items()},
        "speed_mode": dataset.speed_mode,
        "measured_speeds": dataset.speeds,
        "per_episode": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m touhou_rl.ablate36")
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--gate-set",
        choices=sorted(GATE_SETS),
        default="v1",
        help="pre-registered gate registry: 'v1' is the D1/D2 set; 'd3' moves the "
        "ENEMY gate onto attributed_spawns_60 (执行计划-D3补遗 §4)",
    )
    parser.add_argument(
        "--strict-frames",
        action="store_true",
        help="drop decisions whose sampled frame span != frameskip (replay polling "
        "produces ~1.4%% such records); used to show the conclusions do not depend on them",
    )
    parser.add_argument(
        "--speed-mode",
        choices=["config", "measure"],
        default="config",
        help="movement model for horizon features: 'config' = the pilot's Border-Team "
        "constants (D1 numbers are bit-identical); 'measure' = estimate per episode "
        "from the recorded velocity field (required for non-Border characters)",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    dataset = load_dataset(
        args.probe,
        stride=args.stride,
        cache=args.cache,
        speed_mode=args.speed_mode,
        strict_frames=args.strict_frames,
    )
    extract_seconds = time.perf_counter() - started
    fallback = resolve_survival_fallback(dataset)
    cells_spec = resolve_cells(fallback, base=GATE_SETS[args.gate_set])
    ablation = run_ablation(dataset, permutations=args.permutations, seed=args.seed + 1, cells_spec=cells_spec)

    # Global cells use the *effective* survival label (执行计划-D2补遗 §3).  On a
    # human replay corpus death_60 can have zero positives, and a G-3 computed on
    # an untestable label would silently print "n/a" instead of the real number.
    survival_label = fallback["fallback_label"] if fallback["fallback"] else SURVIVAL_LABEL
    survival_kind = LABELS[survival_label]
    all_blocks = list(ALL_BLOCKS)
    all_with = _cell_value(dataset, all_blocks, survival_label)
    all_without = _cell_value(dataset, ["BASE"], survival_label)
    all_delta = None
    if all_with["value"] is not None and all_without["value"] is not None:
        all_delta = all_with["value"] - all_without["value"]
    all_delta_ep = None
    if all_with["value_ep"] is not None and all_without["value_ep"] is not None:
        all_delta_ep = all_with["value_ep"] - all_without["value_ep"]
    all_perm = None
    if args.permutations and all_delta is not None:
        all_perm = _permutation_p(
            dataset,
            ["BASE"],
            "REGION",
            survival_label,
            survival_kind,
            all_delta,
            all_without["value"],
            args.permutations,
            np.random.default_rng(args.seed + 2),
        ).get("p")
    global_cell = {
        "label": survival_label,
        "kind": survival_kind,
        "without": all_without["value"],
        "with": all_with["value"],
        "without_ep": all_without["value_ep"],
        "with_ep": all_with["value_ep"],
        "delta": all_delta,
        "delta_ep": all_delta_ep,
        "permutation_p": all_perm,
        "permutations": args.permutations,
    }
    degenerate = _cell_value(dataset, [], survival_label)
    # An AUC on a 95/5 split is not the same quantity as one on a 40/60 split.
    # Report the balance of every binary label so a "pass" resting on a
    # near-constant label is visible instead of impressive.  A 2% rare event is
    # *not* degenerate - 462 positives support an AUC - so the low bound is 1%.
    label_rates = {}
    for label, kind in LABELS.items():
        if kind != "auc":
            continue
        y = dataset.labels[label]
        finite = int(np.isfinite(y).sum())
        if not finite:
            continue
        positives = int((np.nan_to_num(y) > 0.5).sum())
        label_rates[label] = {"positives": positives, "n": finite, "rate": round(positives / finite, 4)}
    degenerate_labels = [
        label for label, row in label_rates.items() if 0.0 < row["rate"] < 0.01 or row["rate"] > 0.90
    ]

    verdicts = judge(ablation["cells"], global_cell, cells_spec=cells_spec)
    degenerate_cells = [c["strategy"] for c in ablation["cells"] if c["null_degenerate"]]
    report = {
        "schema": SCHEMA,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(args.probe),
        "protocol": {
            "gate_set": args.gate_set,
            "cv": "leave-one-episode-out",
            "model": "ridge",
            "lambda": RIDGE_LAMBDA,
            "stride": args.stride,
            "seed": args.seed,
            "permutations": args.permutations,
            "short_horizon_decisions": S,
            "mid_horizon_decisions": M,
            "frozen_before_results": True,
        },
        "inventory": inventory(dataset),
        "cells": ablation["cells"],
        "global_cells": {f"ALL_vs_BASE_{survival_label}": global_cell},
        "verdicts": verdicts,
        "survival_fallback": fallback,
        "label_rates": label_rates,
        "diagnostics": {
            "degenerate_labels": degenerate_labels,
            "degenerate_auc": degenerate["value"],
            "degenerate_auc_ep": degenerate["value_ep"],
            "degenerate_cells": degenerate_cells,
            "unstable_cells": [name for name, row in verdicts["per_strategy"].items() if row.get("stability") == "unstable"],
            "pooled_vs_ep_disagreements": [
                c["strategy"] for c in ablation["cells"] if _sign_agrees(c["delta"], c["delta_ep"]) is False
            ],
        },
        "timing": {"extract_seconds": round(extract_seconds, 1)},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ablation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "ablation.md").write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdicts": {k: v["verdict"] for k, v in verdicts["per_strategy"].items()},
                "gates": {k: v["pass"] for k, v in verdicts["gates"].items()},
                "enters_T2": verdicts["enters_T2"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
