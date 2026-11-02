"""Measured player movement model and analytic swept collision (D1 geometry).

Why not reuse ``pilot.py``
--------------------------
``pilot.py`` evaluates clearance only at the *end* of every ``frameskip`` step and
approximates the bullet as a circle of radius ``max(half_x, half_y)``.  Two
consequences are unacceptable for the v3.7.0 question:

* a crossing that happens *between* two samples is invisible (SC-08), and
* rounding a box to its larger half extent over-estimates clearance on the
  diagonal (the real test is box-vs-box, ``Player.cpp:298-307``).

So this module implements the real model: per-frame velocity, an inclusive
centre clamp, and the exact minimum of a piecewise-linear box separation over a
segment.  It is deliberately pure NumPy -- no game process, no observation dict.

Motion model evidence
---------------------
* ``Player.cpp:781-790`` sets the per-axis velocity from the shot file:
  ``normalAxisSpeed`` / ``normalDiagonalSpeed`` / ``focusedAxisSpeed`` /
  ``focusedDiagonalSpeed``.  A diagonal sets **both** axes to the diagonal value.
* ``Player.cpp:851-854`` applies ``position += velocity`` once per logic frame
  (scaled by ``framerateMultiplier``, which is 1.0 during normal play).
* ``Player.cpp:856-871`` clamps the *centre* to
  ``[topLeft, topLeft + playerMovementAreaSize]``; the measured clamp is
  ``x in [8, 376]``, ``y in [16, 432]``.
* ``tools/probes/calibrate_player.py`` measured 4.0 / 2.8284 / 2.0 / 1.4142 px
  per frame and a 0.825 px hurtbox half extent in both focus states.

Known limitation: ``framerateMultiplier`` is not always 1.  ECL instructions can
set it (``EclExIns.cpp:776-784``) and even rescale the velocity of every bullet
already in flight (``EclExIns.cpp:859-882``).  This module therefore cannot model
"bullet time" frames; the contract marks such frames unsupported instead of
pretending the linear forecast holds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import spec
from .contracts import HURTBOX_HALF_PX, MOVEMENT_MODEL

# Screen-space unit displacement per (focus, move) class, see ``spec.MOVES``.
MOVE_VECTORS = np.array(
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
    dtype=np.float64,
)

# Action index of the 18 routing candidates: focus * 9 + move (same order as pilot).
FOCUS_OF_ACTION = np.repeat(np.arange(2, dtype=np.int64), 9)
MOVE_OF_ACTION = np.tile(np.arange(9, dtype=np.int64), 2)


@dataclass(frozen=True)
class MoveModel:
    """The movement model as measured on TH08 1.00d, team 0, Lunatic."""

    axis: float = MOVEMENT_MODEL["axis"]
    diagonal: float = MOVEMENT_MODEL["diagonal"]
    focus_axis: float = MOVEMENT_MODEL["focus_axis"]
    focus_diagonal: float = MOVEMENT_MODEL["focus_diagonal"]
    hurtbox_half: float = HURTBOX_HALF_PX
    frameskip: int = 2
    x_min: float = spec.MOVE_AREA_X
    x_max: float = spec.MOVE_AREA_X + spec.MOVE_AREA_W
    y_min: float = spec.MOVE_AREA_Y
    y_max: float = spec.MOVE_AREA_Y + spec.MOVE_AREA_H

    def validate(self) -> None:
        if self.axis <= 0.0 or self.diagonal <= 0.0:
            raise ValueError("axis and diagonal speeds must be positive")
        if self.focus_axis <= 0.0 or self.focus_diagonal <= 0.0:
            raise ValueError("focus speeds must be positive")
        if self.frameskip < 1:
            raise ValueError("frameskip must be >= 1")
        if self.hurtbox_half <= 0.0:
            raise ValueError("hurtbox_half must be positive")
        # The diagonal is not axis/sqrt(2) by construction; it is a separate shot
        # file field.  A 1e-3 px tolerance is the calibration's own precision.
        if abs(self.diagonal - self.axis / np.sqrt(2.0)) > 1e-3:
            raise ValueError("diagonal speed is not the calibrated value for this build")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("movement rectangle is empty")

    def __post_init__(self) -> None:
        self.validate()

    def displacement(self, focus: int, move: int, frames: int | None = None) -> np.ndarray:
        """Exact centre displacement of holding one action for ``frames`` frames."""
        if frames is None:
            frames = self.frameskip
        if frames < 1:
            raise ValueError("frames must be >= 1")
        if focus not in (0, 1):
            raise ValueError(f"focus must be 0 or 1, got {focus}")
        if not 0 <= move <= 8:
            raise ValueError(f"move must be in 0..8, got {move}")
        ux, uy = MOVE_VECTORS[move]
        if ux != 0.0 and uy != 0.0:
            comp = self.diagonal if focus == 0 else self.focus_diagonal
        else:
            comp = self.axis if focus == 0 else self.focus_axis
        return np.array([ux * comp, uy * comp], dtype=np.float64) * float(frames)

    def action_displacements(self, frames: int | None = None) -> np.ndarray:
        """``(18, 2)`` displacement for every routing action, ordered focus*9+move."""
        out = np.empty((18, 2), dtype=np.float64)
        for action in range(18):
            out[action] = self.displacement(int(FOCUS_OF_ACTION[action]), int(MOVE_OF_ACTION[action]), frames)
        return out

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        """Clamp a centre to the movement rectangle (inclusive, as the game does)."""
        return (
            float(np.clip(x, self.x_min, self.x_max)),
            float(np.clip(y, self.y_min, self.y_max)),
        )

    def apply(self, x: float, y: float, focus: int, move: int, frames: int | None = None):
        dx, dy = self.displacement(focus, move, frames)
        return self.clamp(x + dx, y + dy)


# ---------------------------------------------------------------------------
# Box-vs-box clearance
# ---------------------------------------------------------------------------
def box_gap(pa, pb, half_a, half_b) -> float:
    """Signed separation of two axis-aligned boxes (positive = gap, negative = overlap)."""
    a = np.asarray(pa, dtype=np.float64)
    b = np.asarray(pb, dtype=np.float64)
    ha = np.asarray(half_a, dtype=np.float64)
    hb = np.asarray(half_b, dtype=np.float64)
    d = np.abs(a - b) - (ha + hb)
    return float(np.max(d))


def _kink_times(d: np.ndarray, v: np.ndarray, radius: np.ndarray, frames: float) -> list[float]:
    """Times in ``[0, frames]`` where either axis term changes slope."""
    ts = {0.0, float(frames)}
    for axis in range(2):
        if v[axis] == 0.0:
            continue
        for target in (0.0, radius[axis], -radius[axis]):
            t = (target - d[axis]) / v[axis]
            if 0.0 < t < frames:
                ts.add(float(t))
    return sorted(ts)


def _gap_at(t: float, d: np.ndarray, v: np.ndarray, radius: np.ndarray) -> float:
    return float(np.max(np.abs(d + v * t) - radius))


def swept_box_clearance(p0_a, v_a, p0_b, v_b, half_a, half_b, frames: float) -> float:
    """Exact minimum box separation while both boxes move linearly for ``frames``.

    Each axis contributes ``|d_i + v_i t| - R_i``; its kinks are at
    ``d_i + v_i t = 0, ±R_i``.  Between two consecutive kinks both axis terms are
    linear, so the maximum of the two is the maximum of two straight lines: its
    minimum on that interval is at an endpoint or where the two lines cross.
    Both cases are evaluated analytically, so a crossing that falls between two
    frame samples is still caught.
    """
    if frames < 0:
        raise ValueError("frames must be >= 0")
    d = np.asarray(p0_a, dtype=np.float64) - np.asarray(p0_b, dtype=np.float64)
    v = np.asarray(v_a, dtype=np.float64) - np.asarray(v_b, dtype=np.float64)
    radius = np.asarray(half_a, dtype=np.float64) + np.asarray(half_b, dtype=np.float64)
    if radius.shape != (2,):
        raise ValueError("half extents must describe 2-D boxes")

    times = _kink_times(d, v, radius, float(frames))
    best = min(_gap_at(t, d, v, radius) for t in times)
    for left, right in zip(times, times[1:]):
        middle = 0.5 * (left + right)
        signs = [1.0 if value >= 0.0 else -1.0 for value in d + v * middle]
        # On (left, right) the term for axis i is signs[i] * (d_i + v_i t) - R_i.
        c = [signs[axis] * d[axis] - radius[axis] for axis in range(2)]
        m = [signs[axis] * v[axis] for axis in range(2)]
        if m[0] != m[1]:
            crossing = (c[1] - c[0]) / (m[0] - m[1])
            if left <= crossing <= right:
                best = min(best, _gap_at(crossing, d, v, radius))
    return float(best)


def path_segment_clearances(path, hazard_position, hazard_velocity, hazard_half, hurtbox_half):
    """``(segments, hazards)`` exact clearance of a piecewise-linear player path.

    ``path`` is ``(T+1, 2)`` positions at consecutive frames; segment ``k`` moves
    the player from ``path[k]`` to ``path[k+1]`` while every hazard moves at its
    own constant velocity.
    """
    knots = np.asarray(path, dtype=np.float64)
    if knots.ndim != 2 or knots.shape[1] != 2 or knots.shape[0] < 2:
        raise ValueError("path must be (>=2, 2)")
    pos = np.asarray(hazard_position, dtype=np.float64).reshape(-1, 2)
    vel = np.asarray(hazard_velocity, dtype=np.float64).reshape(-1, 2)
    half = np.asarray(hazard_half, dtype=np.float64).reshape(-1, 2)
    player_half = np.asarray(hurtbox_half, dtype=np.float64)
    if half.shape[0] != pos.shape[0]:
        raise ValueError("hazard half sizes do not match positions")
    out = np.empty((knots.shape[0] - 1, pos.shape[0]), dtype=np.float64)
    for segment in range(knots.shape[0] - 1):
        start = knots[segment]
        player_velocity = knots[segment + 1] - start
        # Hazard positions are per-frame snapshots; segment k starts at pos + k*vel.
        hazard_start = pos + vel * float(segment)
        for index in range(pos.shape[0]):
            out[segment, index] = swept_box_clearance(
                start,
                player_velocity,
                hazard_start[index],
                vel[index],
                player_half,
                half[index],
                1.0,
            )
    return out


def path_min_clearance(path, hazard_position, hazard_velocity, hazard_half, hurtbox_half):
    """Return ``(min_clearance, first_segment, first_hazard)`` for a played path."""
    table = path_segment_clearances(path, hazard_position, hazard_velocity, hazard_half, hurtbox_half)
    if table.size == 0:
        return float("inf"), -1, -1
    flat = int(np.argmin(table))
    segment, hazard = divmod(flat, table.shape[1])
    return float(table[segment, hazard]), segment, hazard


def player_path(start, end, frames: int) -> np.ndarray:
    """Exact per-frame knots for a constant-velocity move (velocity is instant in TH08).

    The game sets the full speed on the first frame of a direction and applies
    ``position += velocity`` every logic frame (``Player.cpp:849-854``), so the
    position between two recorded decisions is linear in the frame index.
    """
    if frames < 1:
        raise ValueError("frames must be >= 1")
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(end, dtype=np.float64)
    steps = np.arange(frames + 1, dtype=np.float64)[:, None] / float(frames)
    return a + (b - a) * steps


def executed_path(
    start,
    velocity_previous,
    velocity_current,
    frames: int,
    *,
    model: "MoveModel | None" = None,
) -> np.ndarray:
    """Rebuild the *executed* path of a decision, including the 1-frame input settle.

    Measured on the recorded traces (D1): the first logic frame after a key
    change still runs the previous key state.  With the previous and current
    velocity the next snapshot is predicted **exactly** in 96.6% of 2,903
    decision pairs, versus 65.9% if the new velocity is assumed from frame 1.
    The remaining residual is 0.001 px at P95, and the few large ones coincide
    with deaths.

    So::

        frame 1        uses velocity_previous
        frames 2..N    use  velocity_current
        every step is clamped to the movement rectangle, like the game

    This is what ``action_latency_frames = 1`` means operationally; a planner
    that assumes the new action applies immediately is planning a path the game
    will not execute.
    """
    if frames < 1:
        raise ValueError("frames must be >= 1")
    box = model or MoveModel()
    position = np.asarray(start, dtype=np.float64).copy()
    previous = np.asarray(velocity_previous, dtype=np.float64)
    current = np.asarray(velocity_current, dtype=np.float64)
    out = np.empty((frames + 1, 2), dtype=np.float64)
    out[0] = np.array(box.clamp(position[0], position[1]))
    for step in range(1, frames + 1):
        position = position + (previous if step == 1 else current)
        out[step] = np.array(box.clamp(position[0], position[1]))
        position = out[step]
    return out


def frame_clearances(path, hazard_position, hazard_velocity, hazard_half, hurtbox_half):
    """Box separation at every **integer frame** of the path: ``(frames+1, hazards)``.

    This is the game's own predicate.  ``BulletManager::OnUpdate`` moves the
    bullet (``BulletManager.cpp:865``) and then calls
    ``Player::CheckBulletCollision`` once for that frame
    (``BulletManager.cpp:933``), so a bullet that jumps across the hurtbox
    between two frames is *not* a hit in TH08.  Path verification therefore has
    to check every frame, but not sub-frame instants.
    """
    knots = np.asarray(path, dtype=np.float64)
    if knots.ndim != 2 or knots.shape[1] != 2:
        raise ValueError("path must be (T+1, 2)")
    pos = np.asarray(hazard_position, dtype=np.float64).reshape(-1, 2)
    vel = np.asarray(hazard_velocity, dtype=np.float64).reshape(-1, 2)
    half = np.asarray(hazard_half, dtype=np.float64).reshape(-1, 2)
    if half.shape[0] != pos.shape[0]:
        raise ValueError("hazard half sizes do not match positions")
    frames = knots.shape[0] - 1
    out = np.empty((frames + 1, pos.shape[0]), dtype=np.float64)
    for step in range(frames + 1):
        centres = pos + vel * float(step)
        gap = np.abs(knots[step][None, :] - centres) - (
            np.asarray(hurtbox_half, dtype=np.float64) + half
        )
        out[step] = np.max(gap, axis=1) if gap.size else np.zeros(0)
    return out


def frame_min_clearance(path, hazard_position, hazard_velocity, hazard_half, hurtbox_half, first_frame: int = 1):
    """Minimum frame-sampled box clearance over a played path (game predicate)."""
    table = frame_clearances(path, hazard_position, hazard_velocity, hazard_half, hurtbox_half)
    if table.size == 0 or table.shape[0] <= first_frame:
        return float("inf"), -1, -1
    window = table[first_frame:]
    flat = int(np.argmin(window))
    step, hazard = divmod(flat, window.shape[1])
    return float(window[step, hazard]), step + first_frame, hazard


def max_inflation_for_gap(gap_width: float, hurtbox_half: float) -> float:
    """Largest symmetric inflation that still lets the hurtbox pass through a gap.

    A gap of width ``g`` between two obstacles leaves ``g/2 - hurtbox_half`` of
    lateral room on each side of the centre line.  Any inflation at or above that
    value seals a gap the real game allows, which is exactly the failure mode
    SC-03/E-05 were written for.
    """
    if gap_width < 0.0:
        raise ValueError("gap_width must be >= 0")
    return max(0.0, gap_width / 2.0 - float(hurtbox_half))


def segment_hits(p0_a, v_a, p0_b, v_b, half_a, half_b, frames: float) -> bool:
    """Whether the two boxes touch at any instant of the segment (gap <= 0)."""
    return swept_box_clearance(p0_a, v_a, p0_b, v_b, half_a, half_b, frames) <= 0.0
