"""Entity-set observation for the privileged (step 2) policy.

Why a second observation
------------------------
``touhou_rl.obs`` truncates the bullet pool to the **64 nearest** bullets and
folds everything else into a 16x8 count grid.  In dense patterns that is a small
ring around the player that hides the column it is about to move into
(``pilot.py:48-55``), and the recorded games really do exceed the cap: active
counts of 93, 207 and a maximum of 290 appear in the v2.0.0 decision traces.

This module keeps the **whole active pool** (up to ``K_BULLETS``=320, the same cap
the scripted teacher plans with) plus a four-scalar context vector.  Bullets are
encoded player-relative, in the same units as ``obs``, so the two encoders stay
comparable; ordering is normalised by distance only to make truncation
deterministic -- the network pools over the set, so order carries no information.

Two entry points, one tensor layout
-----------------------------------
* :func:`from_snapshot` -- a ``raw_safety_snapshot`` from a recorded decision
  trace (``Document/Update/Alpha/v2.0.0/rollouts/**/episode-*-decisions.jsonl``);
* :func:`from_env` -- the live game, through ``info["bullets_raw"]``
  (``EnvConfig.expose_raw_bullets``) plus the player position recovered from
  ``obs["self_feat"][0:2]``.

``tests/test_entity_step2.py`` asserts that both paths produce the same tensors
for the same game state.
"""

from __future__ import annotations

import numpy as np

from . import spec

#: Maximum bullets kept per decision.  Equal to ``pilot.PilotConfig.max_bullets``
#: so the teacher's plan and the learned policy see the same world.
K_BULLETS = 320
#: Per-bullet features, all player-relative:
#:
#: 0 ``dx/192``        1 ``dy/224``          2 ``dvx/6``            3 ``dvy/6``
#: 4 ``half/8``        5 ``valid``           6 ``ttc/60``
#: 7 ``closing/6``     8 ``dist/224``
#:
#: ``ttc`` is "frames until this bullet would touch the hurtbox", capped at 60, with
#: 1.0 meaning "receding / no threat"; ``closing`` is the radial approach speed.
#: Those two are what make "this wave arrives in 20 frames" visible to a pooled
#: encoder, which otherwise only sees *where* bullets are right now.
BULLET_FEATURES = 9
TTC_CAP_FRAMES = 60.0
#: ``rel_x, rel_y, stage/6, active_count/700, phi``.
META_DIM = 5
#: Horizon (frames) of the escape-option potential in ``meta[4]``.
META_PHI_HORIZON = 20
#: Danger grid: occupancy of the whole playfield, now and extrapolated ahead.  The
#: entity pool says *which bullets are near*; the grid says *where the corridors are*,
#: which a max/mean pool cannot represent.
GRID_ROWS = 16
GRID_COLS = 16
GRID_FRAMES = (0, 20, 40)
GRID_CHANNELS = len(GRID_FRAMES)
GRID_CELL_W = spec.PLAYFIELD_W / GRID_COLS
GRID_CELL_H = spec.PLAYFIELD_H / GRID_ROWS
#: Occupancy is a count capped here and divided by it, keeping values in [0, 1].
GRID_SATURATION = 4.0
#: Player hurtbox half extent and bullet padding used by the option model.
PLAYER_HALF = 0.825
BULLET_PAD = 0.5
SCHEMA = "v313-entity-obs/1"


def _phi(
    player_x: float,
    player_y: float,
    positions: np.ndarray,
    velocities: np.ndarray,
    halves: np.ndarray,
    count: int,
) -> float:
    """Escape-option potential in ``[0, 1]``; ``1`` when nothing threatens."""
    if count == 0:
        return 1.0
    from .options import option_count

    total = option_count(
        (player_x, player_y), positions, velocities, halves, META_PHI_HORIZON
    )
    return float(total) / 18.0


def _empty() -> dict[str, np.ndarray]:
    return {
        "bullets": np.zeros((K_BULLETS, BULLET_FEATURES), dtype=np.float32),
        "meta": np.zeros((META_DIM,), dtype=np.float32),
        "grid": np.zeros((GRID_CHANNELS, GRID_ROWS, GRID_COLS), dtype=np.float32),
    }


def _danger_grid(positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Occupancy of the playfield now and after ``GRID_FRAMES`` frames of motion."""
    grid = np.zeros((GRID_CHANNELS, GRID_ROWS, GRID_COLS), dtype=np.float32)
    if positions.shape[0] == 0:
        return grid
    for channel, frames in enumerate(GRID_FRAMES):
        points = positions + velocities * float(frames)
        cell_x = ((points[:, 0] - spec.PLAYFIELD_X) / GRID_CELL_W).astype(np.int32)
        cell_y = ((points[:, 1] - spec.PLAYFIELD_Y) / GRID_CELL_H).astype(np.int32)
        inside = (
            (cell_x >= 0) & (cell_x < GRID_COLS) & (cell_y >= 0) & (cell_y < GRID_ROWS)
        )
        if not inside.any():
            continue
        counts = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
        np.add.at(counts, (cell_y[inside], cell_x[inside]), 1.0)
        grid[channel] = np.minimum(counts, GRID_SATURATION) / GRID_SATURATION
    return grid


def build(
    player_x: float,
    player_y: float,
    positions: np.ndarray,
    velocities: np.ndarray,
    halves: np.ndarray,
    *,
    stage: int = 0,
    field_x: float = spec.PLAYFIELD_X,
    field_y: float = spec.PLAYFIELD_Y,
    field_w: float = spec.PLAYFIELD_W,
    field_h: float = spec.PLAYFIELD_H,
) -> dict[str, np.ndarray]:
    """Encode one decision's world as a padded bullet set plus context.

    ``positions``/``velocities``/``halves`` are absolute playfield coordinates and
    velocities in px/frame.  Raises ``ValueError`` when the three arrays disagree
    in length: a silent broadcast would train on misaligned labels.
    """
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 2)
    velocities = np.asarray(velocities, dtype=np.float32).reshape(-1, 2)
    halves = np.asarray(halves, dtype=np.float32).reshape(-1)
    if not (positions.shape[0] == velocities.shape[0] == halves.shape[0]):
        raise ValueError(
            f"bullet arrays disagree: positions={positions.shape[0]} "
            f"velocities={velocities.shape[0]} halves={halves.shape[0]}"
        )

    out = _empty()
    count = positions.shape[0]
    if count:
        dx = positions[:, 0] - float(player_x)
        dy = positions[:, 1] - float(player_y)
        order = np.argsort(dx * dx + dy * dy, kind="stable")[:K_BULLETS]
        kept = order.size
        sel_dx, sel_dy = dx[order], dy[order]
        sel_vx, sel_vy = velocities[order, 0], velocities[order, 1]
        distance = np.maximum(np.hypot(sel_dx, sel_dy), 1e-6)
        # Radial closing speed: positive when the bullet is closing on the player.
        closing = -(sel_dx * sel_vx + sel_dy * sel_vy) / distance
        radius = halves[order] + PLAYER_HALF + BULLET_PAD
        gap = distance - radius
        with np.errstate(divide="ignore", invalid="ignore"):
            ttc = np.where(closing > 1e-6, gap / np.maximum(closing, 1e-6), TTC_CAP_FRAMES)
        ttc = np.clip(ttc, 0.0, TTC_CAP_FRAMES)
        out["bullets"][:kept, 0] = sel_dx / 192.0
        out["bullets"][:kept, 1] = sel_dy / 224.0
        out["bullets"][:kept, 2] = sel_vx / 6.0
        out["bullets"][:kept, 3] = sel_vy / 6.0
        out["bullets"][:kept, 4] = halves[order] / 8.0
        out["bullets"][:kept, 5] = 1.0
        out["bullets"][:kept, 6] = ttc / TTC_CAP_FRAMES
        out["bullets"][:kept, 7] = np.clip(closing / 6.0, -1.0, 1.0)
        out["bullets"][:kept, 8] = np.clip(distance / 224.0, 0.0, 1.5)

    out["grid"] = _danger_grid(positions, velocities)

    out["meta"] = np.array(
        [
            (float(player_x) - field_x) / max(1.0, field_w),
            (float(player_y) - field_y) / max(1.0, field_h),
            min(1.0, float(stage) / 6.0),
            min(1.0, float(count) / 700.0),
            _phi(player_x, player_y, positions, velocities, halves, count),
        ],
        dtype=np.float32,
    )
    return out


def from_snapshot(snapshot: dict) -> dict[str, np.ndarray]:
    """Encode a recorded ``raw_safety_snapshot`` (decision-trace JSON)."""
    player = snapshot.get("player_position")
    bullets = snapshot.get("bullets") or {}
    if player is None or "position" not in bullets:
        raise ValueError("snapshot lacks player_position or bullets.position")
    return build(
        float(player[0]),
        float(player[1]),
        bullets["position"],
        bullets["velocity"],
        bullets["half"],
        stage=int(snapshot.get("stage", 0)),
    )


def from_env(info: dict, observation: dict) -> dict[str, np.ndarray]:
    """Encode the live game from ``info["bullets_raw"]`` + ``obs["self_feat"]``.

    ``self_feat[0]``/``[1]`` are the player position normalised against the
    playfield (``obs.encode``), so the absolute position is recovered exactly.
    """
    raw = info.get("bullets_raw")
    if raw is None:
        raise ValueError("info['bullets_raw'] is missing: set EnvConfig.expose_raw_bullets=True")
    self_feat = np.asarray(observation["self_feat"], dtype=np.float32).reshape(-1)
    if self_feat.size < 2:
        raise ValueError("observation['self_feat'] must carry rel_x/rel_y")
    player_x = spec.PLAYFIELD_X + float(self_feat[0]) * spec.PLAYFIELD_W
    player_y = spec.PLAYFIELD_Y + float(self_feat[1]) * spec.PLAYFIELD_H
    return build(
        player_x,
        player_y,
        raw["position"],
        raw["velocity"],
        raw["half"],
        stage=int(info.get("stage", 0)),
    )


def batch(observations: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack observations into ``bullets[B,K,6]`` / ``meta[B,META_DIM]``."""
    if not observations:
        raise ValueError("cannot batch zero observations")
    return {
        "bullets": np.stack([item["bullets"] for item in observations]).astype(np.float32),
        "meta": np.stack([item["meta"] for item in observations]).astype(np.float32),
        "grid": np.stack([item["grid"] for item in observations]).astype(np.float32),
    }


def candidate_index(move: int, focus: int) -> int:
    """``focus*9 + move`` -- the candidate order shared with ``pilot``/``chunk_safety``."""
    if not 0 <= int(move) < 9:
        raise ValueError(f"invalid move {move}")
    if not 0 <= int(focus) < 2:
        raise ValueError(f"invalid focus {focus}")
    return int(focus) * 9 + int(move)


def action_of_candidate(candidate: int) -> tuple[int, int, int, int]:
    """18-way candidate -> the game's ``(move, shoot, focus, bomb)`` four-tuple.

    ``shoot`` is pinned to 1 and ``bomb`` to 0: the step-2 action space deliberately
    excludes bombs (see the step-2 plan), so the mapping is total and lossless.
    """
    candidate = int(candidate)
    if not 0 <= candidate < 18:
        raise ValueError(f"candidate out of range: {candidate}")
    return candidate % 9, 1, candidate // 9, 0


__all__ = [
    "BULLET_FEATURES",
    "GRID_CHANNELS",
    "GRID_COLS",
    "GRID_FRAMES",
    "GRID_ROWS",
    "K_BULLETS",
    "META_DIM",
    "SCHEMA",
    "action_of_candidate",
    "batch",
    "build",
    "candidate_index",
    "from_env",
    "from_snapshot",
]
