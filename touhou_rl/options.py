"""Escape-option geometry: how many of the 18 moves stay clear for H frames.

Why this module exists
----------------------
The step-2 policies collapsed onto "hold one direction" (move flips 4-10% versus the
scripted teacher's 32%).  A policy that never re-evaluates only gets punished at the
moment of death -- by then every candidate collides, so the gradient says nothing
useful.  :func:`option_count` turns "am I still able to escape?" into a number that
can be (a) logged and (b) used as a potential for reward shaping:

    F(s, s') = gamma * phi(s') - phi(s),   phi(s) = option_count(s, H) / 18

Potential-based shaping with a state-only potential leaves the optimal policy set
unchanged (Ng, Harada & Russell 1999), which is why this is preferred over a free-form
penalty on "standing still".

The model is the same one the scripted teacher plans with: constant-velocity bullets,
constant-direction player, collision radius = bullet half + player half + pad.  It is
deliberately simplified (no spawning, no laser bends, no invulnerability) -- it is a
*lower* bound on danger, used as a signal, not as a safety certificate.
"""

from __future__ import annotations

import numpy as np

from . import spec

#: 9 directions (idle first) x 2 focus states, in `focus*9+move` order.
DIRECTIONS = np.array(
    [(0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)],
    dtype=np.float64,
)
SPEED = 4.0
SPEED_DIAG = 2.8284271
FOCUS_SPEED = 2.0
FOCUS_SPEED_DIAG = 1.4142135
PLAYER_HALF = 0.825
BULLET_PAD = 0.5
#: Default lookahead for the potential, in game frames (20 = 0.33 s).
HORIZON_FRAMES = 20


def candidate_steps() -> np.ndarray:
    """Per-decision displacement of the 18 candidates, ``(18, 2)`` px."""
    axis = np.repeat([SPEED, FOCUS_SPEED], 9)
    diagonal = np.repeat([SPEED_DIAG, FOCUS_SPEED_DIAG], 9)
    directions = np.tile(DIRECTIONS, (2, 1))
    return directions * np.where(np.all(directions != 0, axis=1), diagonal, axis)[:, None]


STEPS = candidate_steps()


def _bounds() -> tuple[np.ndarray, np.ndarray]:
    low = np.array([spec.MOVE_AREA_X + PLAYER_HALF, spec.MOVE_AREA_Y + PLAYER_HALF])
    high = np.array(
        [
            spec.MOVE_AREA_X + spec.MOVE_AREA_W - PLAYER_HALF,
            spec.MOVE_AREA_Y + spec.MOVE_AREA_H - PLAYER_HALF,
        ]
    )
    return low, high


def option_count(
    player: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    halves: np.ndarray,
    horizon: int = HORIZON_FRAMES,
) -> int:
    """How many of the 18 moves survive ``horizon`` frames of constant velocity/heading."""
    player = np.asarray(player, dtype=np.float64).reshape(2)
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    velocities = np.asarray(velocities, dtype=np.float64).reshape(-1, 2)
    halves = np.asarray(halves, dtype=np.float64).reshape(-1)
    if positions.shape[0] != velocities.shape[0] or positions.shape[0] != halves.shape[0]:
        raise ValueError("bullet arrays disagree in length")
    if positions.shape[0] == 0:
        return len(STEPS)
    low, high = _bounds()
    radius = halves + PLAYER_HALF + BULLET_PAD
    clear = np.ones(len(STEPS), dtype=bool)
    current = np.clip(np.tile(player, (len(STEPS), 1)), low, high)
    for frame in range(horizon + 1):
        bullets = positions + velocities * frame
        distance = np.linalg.norm(current[:, None, :] - bullets[None, :, :], axis=-1) - radius[None, :]
        clear &= distance.min(axis=1) > 0.0
        if not clear.any():
            return 0
        current = np.clip(current + STEPS, low, high)
    return int(clear.sum())


def phi(
    player: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    halves: np.ndarray,
    horizon: int = HORIZON_FRAMES,
) -> float:
    """Potential in ``[0, 1]``: share of moves that keep the player clear."""
    count = option_count(player, positions, velocities, halves, horizon)
    return float(count) / float(len(STEPS))


def phi_from_snapshot(snapshot: dict, horizon: int = HORIZON_FRAMES) -> float:
    """Convenience wrapper for a recorded ``raw_safety_snapshot``."""
    bullets = snapshot["bullets"]
    return phi(
        snapshot["player_position"],
        bullets["position"],
        bullets["velocity"],
        bullets["half"],
        horizon,
    )


def shaped_reward(
    phi_before: float, phi_after: float, gamma: float, terminal: bool
) -> float:
    """``gamma*phi(s') - phi(s)`` with ``phi(terminal) = 0`` (the classic PBRS trap)."""
    target = 0.0 if terminal else float(phi_after)
    return float(gamma) * target - float(phi_before)


__all__ = [
    "DIRECTIONS",
    "HORIZON_FRAMES",
    "STEPS",
    "candidate_steps",
    "option_count",
    "phi",
    "phi_from_snapshot",
    "shaped_reward",
]
