"""Observation contract: privileged game state -> fixed-size tensors.

Why object-level features instead of pixels
-------------------------------------------
TH08 exposes exact bullet positions/velocities through read-only memory
(``state.read_bullets``).  Encoding that field directly is far more
sample-efficient than pixels and needs no HUD OCR; the v1.0.1 design keeps the
visual channel as an optional second path (``profile.state_source``), and this
package implements the privileged path only.

Layout (contract §3):

* ``self_feat``   float32 [24] -- player state, resources, progress and eight
  "danger ray" distances.
* ``bullets``     float16 [64, 6] -- the 64 nearest bullets, player-relative:
  ``dx, dy, dvx, dvy, half_size, valid``, sorted by distance.
* ``danger_map``  uint8 [16, 8] -- coarse occupancy of the whole playfield
  (24 px x 56 px cells) so global patterns stay visible.

Everything is translation-normalised against the playfield and the player, so
the same encoder works for any stage and any shot type.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import spec
from .state import BulletField, GameState

N_SELF = 24
N_BULLETS = 64
N_BULLET_FEATURES = 6
DANGER_ROWS = 16
DANGER_COLS = 8
DANGER_CELL_W = spec.PLAYFIELD_W / DANGER_COLS  # 48 px
DANGER_CELL_H = spec.PLAYFIELD_H / DANGER_ROWS  # 28 px
RAY_COUNT = 8
RAY_RANGE = 320.0
RAY_HALF_WIDTH = 6.0  # px of lateral tolerance around each ray

# Unit vectors for the eight rays, in screen coordinates (+y down).
_RAYS = np.array(
    [
        [0.0, -1.0],
        [0.7071, -0.7071],
        [1.0, 0.0],
        [0.7071, 0.7071],
        [0.0, 1.0],
        [-0.7071, 0.7071],
        [-1.0, 0.0],
        [-0.7071, -0.7071],
    ],
    dtype=np.float32,
)


@dataclass(slots=True)
class Observation:
    self_feat: np.ndarray  # float32 [24]
    bullets: np.ndarray  # float16 [64, 6]
    danger_map: np.ndarray  # uint8 [16, 8]


def empty_observation() -> Observation:
    return Observation(
        self_feat=np.zeros(N_SELF, dtype=np.float32),
        bullets=np.zeros((N_BULLETS, N_BULLET_FEATURES), dtype=np.float16),
        danger_map=np.zeros((DANGER_ROWS, DANGER_COLS), dtype=np.uint8),
    )


def _ray_distances(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Nearest-bullet distance along eight rays (normalised, 1.0 = no danger)."""
    if dx.size == 0:
        return np.ones(RAY_COUNT, dtype=np.float32)
    dist = np.sqrt(dx * dx + dy * dy)
    out = np.ones(RAY_COUNT, dtype=np.float32)
    for i in range(RAY_COUNT):
        ux, uy = _RAYS[i]
        along = dx * ux + dy * uy
        lateral = np.abs(dx * uy - dy * ux)
        mask = (along > 0.0) & (along < RAY_RANGE) & (lateral < RAY_HALF_WIDTH + 0.02 * along)
        if np.any(mask):
            out[i] = min(1.0, float(dist[mask].min()) / RAY_RANGE)
    return out


def encode(
    state: GameState,
    bullets: BulletField | None,
    *,
    nearest: int = N_BULLETS,
) -> Observation:
    """Encode one sampled frame (contract §3.1)."""
    field_w = max(1.0, state.field_w or spec.PLAYFIELD_W)
    field_h = max(1.0, state.field_h or spec.PLAYFIELD_H)
    rel_x = (state.player_x - state.field_x) / field_w
    rel_y = (state.player_y - state.field_y) / field_h

    dx = np.zeros(0, dtype=np.float32)
    dy = np.zeros(0, dtype=np.float32)
    dvx = np.zeros(0, dtype=np.float32)
    dvy = np.zeros(0, dtype=np.float32)
    half = np.zeros(0, dtype=np.float32)
    active_positions = None
    active_velocities = None
    active_half = None
    active_count = 0

    if bullets is not None:
        mask = bullets.active_mask()
        active_count = int(mask.sum())
        if active_count:
            pos = bullets.position[mask]
            vel = bullets.velocity[mask]
            size = bullets.half_size[mask]
            active_positions = pos
            active_velocities = vel
            active_half = np.maximum(size[:, 0], 1.0)
            dx = (pos[:, 0] - state.player_x).astype(np.float32)
            dy = (pos[:, 1] - state.player_y).astype(np.float32)
            dvx = vel[:, 0].astype(np.float32)
            dvy = vel[:, 1].astype(np.float32)
            half = active_half.astype(np.float32)

    # --- self features -----------------------------------------------------
    rays = _ray_distances(dx, dy)
    near_mask = (dx * dx + dy * dy) < (192.0 * 192.0)
    # Slots 2-3 are reserved (always 0) in v1.0.2: the raw player velocity is
    # *caused by* the current movement input, so feeding it to a cloned policy
    # leaks the label and collapses the policy into "keep my current velocity".
    # The field is kept so the contract shape does not change between versions.
    self_feat = np.array(
        [
            rel_x,
            rel_y,
            0.0,
            0.0,
            float(state.focus_mode),
            state.lives / 8.0,
            state.bombs / 8.0,
            state.power / 128.0,
            min(1.0, state.graze / 20000.0),
            min(1.0, state.score / 2.0e8),
            min(1.0, max(0.0, state.clock_time) / 20.0),
            min(1.0, state.time_orbs / 20.0),
            min(1.0, state.stage / 6.0),
            float(state.stage_clear_flag != 0),
            min(1.0, active_count / 700.0),
            min(1.0, float(near_mask.sum()) / 120.0),
            *rays.tolist(),
        ],
        dtype=np.float32,
    )
    assert self_feat.shape[0] == N_SELF

    # --- nearest bullets ---------------------------------------------------
    bullet_feat = np.zeros((nearest, N_BULLET_FEATURES), dtype=np.float16)
    if dx.size:
        dist2 = dx * dx + dy * dy
        order = np.argsort(dist2, kind="stable")[:nearest]
        sel_dx = dx[order]
        sel_dy = dy[order]
        bullet_feat[: order.size, 0] = sel_dx / 192.0
        bullet_feat[: order.size, 1] = sel_dy / 224.0
        bullet_feat[: order.size, 2] = dvx[order] / 6.0
        bullet_feat[: order.size, 3] = dvy[order] / 6.0
        bullet_feat[: order.size, 4] = half[order] / 8.0
        bullet_feat[: order.size, 5] = 1.0

    # --- coarse danger map -------------------------------------------------
    danger = np.zeros((DANGER_ROWS, DANGER_COLS), dtype=np.uint8)
    if active_positions is not None and active_count:
        cells_x = ((active_positions[:, 0] - state.field_x) / DANGER_CELL_W).astype(np.int32)
        cells_y = ((active_positions[:, 1] - state.field_y) / DANGER_CELL_H).astype(np.int32)
        valid = (
            (cells_x >= 0) & (cells_x < DANGER_COLS) & (cells_y >= 0) & (cells_y < DANGER_ROWS)
        )
        counts = np.zeros((DANGER_ROWS, DANGER_COLS), dtype=np.uint16)
        np.add.at(counts, (cells_y[valid], cells_x[valid]), 1)
        danger = np.minimum(counts, 255).astype(np.uint8)

    return Observation(self_feat=self_feat, bullets=bullet_feat, danger_map=danger)
