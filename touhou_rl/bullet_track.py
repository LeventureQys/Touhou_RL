"""Bullet identity tracking + spawn attribution (v3.6.0 label groundwork).

Why this module exists
----------------------
v3.6.0 §2.3 wants ClearEnemy's value as *"future bullet generation removed if this
enemy dies"*.  That needs bullet->enemy attribution, and
``third_party/th08/src/BulletManager.hpp`` shows ``struct Bullet`` has **no**
enemy/source field (position, velocity, speed, angle, timers, transform flags,
color, state, graze/collision bits - no owner pointer or index).  The trace also
drops ``slot_id``, so a recorded decision cannot say "this bullet is new".

So attribution has to be reconstructed: track bullets across decisions, treat
unmatched bullets in the newer snapshot as **spawns**, and attach each spawn to
the nearest enemy at that moment.  This module does exactly that, with the
matching done on *predicted* positions (``pos + vel * frameskip``) because bullet
motion in TH08 is linear for the overwhelming majority of types.

What it does not claim
----------------------
Attribution is an inference, not a read.  ``attribution_report`` therefore
returns the unattributed fraction and the radius sensitivity, so a report can
say how much of the bullet stream is actually explained instead of presenting a
heuristic as ground truth.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MATCH_TOLERANCE = 1.5  # px between predicted and observed centre
DEFAULT_ATTRIBUTION_RADIUS = 32.0  # px from a spawn to an enemy centre
DEFAULT_CHILD_RADIUS = 24.0  # px from a spawn to a surviving bullet (child pattern)
HALF_TOLERANCE = 0.6  # bullets keep their collision half extent


def match_bullets(
    prev_pos: np.ndarray,
    prev_vel: np.ndarray,
    prev_half: np.ndarray,
    cur_pos: np.ndarray,
    cur_half: np.ndarray,
    frameskip: int = 2,
    tolerance: float = DEFAULT_MATCH_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one match of ``prev`` bullets to ``cur`` bullets.

    Returns ``(assigned_cur_to_prev, new_mask)``: ``assigned_cur_to_prev[j]`` is
    the index in ``prev`` matched by current bullet ``j`` (``-1`` if none), and
    ``new_mask[j]`` is True when current bullet ``j`` is a spawn.

    Greedy-by-nearest is enough here: two bullets only compete for the same
    prediction if they are within ``tolerance`` of each other, and at that point
    either assignment preserves the count of spawns.
    """
    cur_pos = np.asarray(cur_pos, dtype=np.float32).reshape(-1, 2)
    prev_pos = np.asarray(prev_pos, dtype=np.float32).reshape(-1, 2)
    cur_half = np.asarray(cur_half, dtype=np.float32).reshape(-1)
    prev_half = np.asarray(prev_half, dtype=np.float32).reshape(-1)
    n_cur = cur_pos.shape[0]
    assigned = np.full(n_cur, -1, dtype=np.int64)
    if prev_pos.shape[0] == 0 or n_cur == 0:
        return assigned, np.ones(n_cur, dtype=bool)

    pred = prev_pos + np.asarray(prev_vel, dtype=np.float32).reshape(-1, 2) * float(frameskip)
    # (n_cur, n_prev) distances: bullet pools are <= ~400, so this is cheap.
    dx = cur_pos[:, None, 0] - pred[None, :, 0]
    dy = cur_pos[:, None, 1] - pred[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    half_ok = np.abs(cur_half[:, None] - prev_half[None, :]) <= HALF_TOLERANCE
    dist = np.where(half_ok, dist, np.inf)

    order = np.argsort(dist, axis=None)
    used_cur = np.zeros(n_cur, dtype=bool)
    used_prev = np.zeros(prev_pos.shape[0], dtype=bool)
    flat = dist.reshape(-1)
    for flat_index in order:
        if flat[flat_index] > tolerance:
            break
        i, j = divmod(int(flat_index), prev_pos.shape[0])
        if used_cur[i] or used_prev[j]:
            continue
        used_cur[i] = True
        used_prev[j] = True
        assigned[i] = j
    return assigned, ~used_cur


def spawns_by_slot(prev_slots: np.ndarray, cur_slots: np.ndarray) -> np.ndarray:
    """Exact spawn mask from bullet pool slot ids.

    A slot that was inactive in the previous snapshot and active now is a spawn -
    no motion model, no tolerance, no ambiguity.  Slot reuse after a despawn is
    exactly a new bullet and is counted correctly (the slot is absent from
    ``prev_slots``).

    This is why ``TraceWriter`` persists ``slot_id``: the position-matching path
    (``match_bullets``) cannot separate a genuine spawn from a survivor whose
    motion is not exactly ``pos + vel * frameskip``, and the measured spawn rate
    depended on the tolerance (11.3/dec at 1.5 px down to 2.2/dec at 10 px) with
    no convergence - i.e. it was not identifiable.
    """
    prev_slots = np.asarray(prev_slots, dtype=np.int64).reshape(-1)
    cur_slots = np.asarray(cur_slots, dtype=np.int64).reshape(-1)
    if prev_slots.size == 0:
        return np.ones(cur_slots.size, dtype=bool)
    return ~np.isin(cur_slots, prev_slots)


def attribute_spawns(
    spawn_pos: np.ndarray,
    enemies: np.ndarray,
    radius: float = DEFAULT_ATTRIBUTION_RADIUS,
) -> dict:
    """Attach each spawn position to the nearest enemy centre within ``radius``.

    ``enemies`` is ``(k, 2)`` (absolute playfield coordinates, usually the enemy
    centre slightly below the sprite origin).  Returns per-enemy counts plus the
    unattributed count, so a caller can report coverage rather than assume it.
    """
    spawn_pos = np.asarray(spawn_pos, dtype=np.float32).reshape(-1, 2)
    enemies = np.asarray(enemies, dtype=np.float32).reshape(-1, 2)
    counts = np.zeros(enemies.shape[0], dtype=np.int64)
    if spawn_pos.shape[0] == 0:
        return {"counts": counts, "unattributed": 0, "mean_distance": None}
    if enemies.shape[0] == 0:
        return {"counts": counts, "unattributed": int(spawn_pos.shape[0]), "mean_distance": None}
    dx = spawn_pos[:, None, 0] - enemies[None, :, 0]
    dy = spawn_pos[:, None, 1] - enemies[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    nearest = dist.argmin(axis=1)
    nearest_dist = dist[np.arange(spawn_pos.shape[0]), nearest]
    inside = nearest_dist <= radius
    for enemy_index in nearest[inside]:
        counts[enemy_index] += 1
    attributed = int(inside.sum())
    return {
        "counts": counts,
        "unattributed": int(spawn_pos.shape[0] - attributed),
        "mean_distance": float(nearest_dist[inside].mean()) if attributed else None,
    }


def track_episode(
    bullets,
    records: list[dict],
    frameskip: int = 2,
    tolerance: float = DEFAULT_MATCH_TOLERANCE,
    radius: float = DEFAULT_ATTRIBUTION_RADIUS,
    child_radius: float = DEFAULT_CHILD_RADIUS,
) -> dict:
    """Track one episode and report spawn statistics over the whole trace.

    ``bullets`` is the ``bullets/*.npz`` sidecar; ``records`` the decoded JSONL.
    Returns counts, not per-decision arrays, because the question this answers is
    "is attribution feasible at all?" - the label builder comes next.

    Spawns are split three ways so the unattributed fraction is not blamed on the
    attribution radius:

    * ``attributed``  - within ``radius`` of an enemy centre (real generation);
    * ``child_like``  - within ``child_radius`` of a *surviving* bullet, i.e. the
      child-pattern transforms whose payload structs exist in ``Bullet``;
    * ``unexplained`` - neither, e.g. offscreen emitters or a failed match.
    """
    positions_all = bullets["positions"]
    velocities_all = bullets["velocities"]
    half_all = bullets["half"]
    offsets = bullets["offsets"]

    decisions = 0
    matched_pairs = 0
    spawns = 0
    attributed = 0
    unattributed = 0
    child_like = 0
    per_decision_spawns: list[int] = []
    distances: list[float] = []
    prev_pos = prev_vel = prev_half = None
    slots_all = bullets["slots"] if "slots" in bullets else None
    prev_slots = None
    exact = slots_all is not None
    for index, record in enumerate(records):
        start, end = int(offsets[index]), int(offsets[index + 1])
        pos = positions_all[start:end]
        vel = velocities_all[start:end]
        half = half_all[start:end]
        cur_slots = slots_all[start:end] if exact else None
        if prev_pos is not None:
            if exact:
                new_mask = spawns_by_slot(prev_slots, cur_slots)
                assigned = np.full(pos.shape[0], -1, dtype=np.int64)
                matched_pairs += int((~new_mask).sum())
            else:
                assigned, new_mask = match_bullets(prev_pos, prev_vel, prev_half, pos, half, frameskip, tolerance)
                matched_pairs += int((assigned >= 0).sum())
            decisions += 1
            spawn_pos = pos[new_mask]
            survivors = pos[~new_mask]
            spawns += int(new_mask.sum())
            per_decision_spawns.append(int(new_mask.sum()))
            enemies = np.array(
                [[float(e.get("x", 0.0)), float(e.get("y", 0.0))] for e in (record.get("enemies") or [])],
                dtype=np.float32,
            ).reshape(-1, 2)
            info = attribute_spawns(spawn_pos, enemies, radius)
            attributed += int(info["counts"].sum())
            unattributed += int(info["unattributed"])
            if info["mean_distance"] is not None:
                distances.append(info["mean_distance"])
            if spawn_pos.shape[0] and survivors.shape[0]:
                dx = spawn_pos[:, None, 0] - survivors[None, :, 0]
                dy = spawn_pos[:, None, 1] - survivors[None, :, 1]
                nearest_survivor = np.sqrt(dx * dx + dy * dy).min(axis=1)
                child_like += int((nearest_survivor <= child_radius).sum())
        prev_pos, prev_vel, prev_half = pos, vel, half
        prev_slots = cur_slots

    spawn_array = np.asarray(per_decision_spawns, dtype=np.float64)
    return {
        "decisions": decisions,
        "match_method": "slot_id" if exact else "position",
        "matched_pairs": matched_pairs,
        "spawns": spawns,
        "spawns_per_decision_mean": float(spawn_array.mean()) if spawn_array.size else 0.0,
        "spawns_per_decision_p95": float(np.percentile(spawn_array, 95)) if spawn_array.size else 0.0,
        "attributed": attributed,
        "unattributed": unattributed,
        "child_like": child_like,
        "unexplained": int(max(0, spawns - attributed - child_like)),
        "attribution_rate": float(attributed / spawns) if spawns else None,
        "child_like_rate": float(child_like / spawns) if spawns else None,
        "unexplained_rate": float(max(0, spawns - attributed - child_like) / spawns) if spawns else None,
        "mean_attribution_distance": float(np.mean(distances)) if distances else None,
        "tolerance_px": tolerance,
        "radius_px": radius,
        "child_radius_px": child_radius,
        "frameskip": frameskip,
    }
