"""Frame-region vision front-end (D11 skeleton).

What this is
------------
A *synthetic* rasteriser: given a scenario, a frame and a player position it
draws the hazard boxes and the player into a small image, exactly the shape the
live front-end will have to produce from the game's frame buffer.  It exists so
the student (D10) can be trained and evaluated without a game process, and so the
image→features→action path is tested end to end.

What this is **not**
--------------------
It is not the live front-end.  The live path has to cut the playfield out of the
game frame, deal with anti-aliasing, HUD overlap and colour thresholds; none of
that is exercised here.  Every artefact produced through this module is labelled
``synthetic_frames: true``.
"""

from __future__ import annotations

import numpy as np

from .. import spec
from . import geometry, synth

#: Image size the student consumes (rows × cols).  Small on purpose: the live
#: front-end will have to downscale anyway, and a 96×112 image keeps training
#: cheap on CPU.
IMAGE_SHAPE = (96, 112)


def render_state(
    scenario: synth.Scenario,
    frame: int,
    x: float,
    y: float,
    *,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Rasterise hazards (1.0) and the player (0.5) into a float32 image."""

    shape = IMAGE_SHAPE if shape is None else shape
    rows, cols = shape
    image = np.zeros((rows, cols), dtype=np.float32)
    positions, _velocities, halves = scenario.active_positions(frame)
    if positions.shape[0]:
        px = (positions[:, 0] - spec.PLAYFIELD_X) / spec.PLAYFIELD_W
        py = (positions[:, 1] - spec.PLAYFIELD_Y) / spec.PLAYFIELD_H
        hx = np.maximum(1.0 / cols, halves[:, 0] / spec.PLAYFIELD_W)
        hy = np.maximum(1.0 / rows, halves[:, 1] / spec.PLAYFIELD_H)
        c0 = np.clip(((px - hx) * cols).astype(int), 0, cols - 1)
        c1 = np.clip(((px + hx) * cols).astype(int) + 1, 0, cols)
        r0 = np.clip(((py - hy) * rows).astype(int), 0, rows - 1)
        r1 = np.clip(((py + hy) * rows).astype(int) + 1, 0, rows)
        for index in range(positions.shape[0]):
            image[r0[index] : r1[index], c0[index] : c1[index]] = 1.0
    model = geometry.MoveModel()
    player_col = int(np.clip((x - model.x_min) / (model.x_max - model.x_min) * (cols - 1), 0, cols - 1))
    player_row = int(np.clip((y - model.y_min) / (model.y_max - model.y_min) * (rows - 1), 0, rows - 1))
    image[max(0, player_row - 1) : player_row + 2, max(0, player_col - 1) : player_col + 2] = 0.5
    return image


def future_features(
    scenario: synth.Scenario,
    frame: int,
    x: float,
    y: float,
    *,
    horizon: int = 8,
    scale: float = 32.0,
) -> np.ndarray:
    """Eight scalar "is this direction still safe in a moment" features.

    For each of the eight move directions, project the player along that
    direction and take the minimum clearance at the target times.  It is a very
    cheap slice of the forecast — eight numbers instead of a grid — and exists to
    test one hypothesis: the frame-only student fails on the anticipation
    scenarios (E-05/SC-15/SC-16) because its input has no future, not because it
    lacks capacity.

    The features are computed from privileged state here; in the live pipeline
    they would have to come from a coarse forecast (or a second camera-style
    head).  Every artefact that uses them is labelled accordingly.
    """

    model = geometry.MoveModel()
    out = np.zeros(8, dtype=np.float32)
    for index, move in enumerate(range(1, 9)):
        velocity = model.displacement(0, move, frames=1)
        worst = 64.0
        for step in (max(1, horizon // 2), horizon):
            px = np.clip(x + velocity[0] * step, model.x_min, model.x_max)
            py = np.clip(y + velocity[1] * step, model.y_min, model.y_max)
            positions, _velocities, halves = scenario.active_positions(frame + step)
            if positions.shape[0] == 0:
                continue
            gaps = np.abs(np.array([px, py])[None, :] - positions) - (
                synth.HURTBOX_HALF_PX + halves
            )
            clearance = float(gaps.max(axis=1).min())
            worst = min(worst, clearance)
        out[index] = float(np.clip(worst, -scale, scale)) / scale
    return out


def future_map(
    scenario: synth.Scenario,
    frame: int,
    x: float,
    y: float,
    *,
    size: int = 8,
    horizon: int = 8,
) -> np.ndarray:
    """Coarse occupancy of the hazard set ``horizon`` frames ahead, in an 8×8 grid.

    This is the teacher target for the auxiliary "predict the future" head: a
    small map the trunk can be forced to represent internally, instead of eight
    hand-made scalars.  It is computed from privileged state here; the live
    pipeline would have to predict it from frames.
    """

    model = geometry.MoveModel()
    out = np.zeros((size, size), dtype=np.float32)
    positions, _velocities, halves = scenario.active_positions(frame + horizon)
    if positions.shape[0] == 0:
        return out
    width = model.x_max - model.x_min
    height = model.y_max - model.y_min
    cols = np.clip(((positions[:, 0] - model.x_min) / width * size).astype(int), 0, size - 1)
    rows = np.clip(((positions[:, 1] - model.y_min) / height * size).astype(int), 0, size - 1)
    out[rows, cols] = 1.0
    return out


def render_pair(
    scenario: synth.Scenario,
    frame: int,
    previous_xy: tuple[float, float] | None,
    xy: tuple[float, float],
    *,
    shape: tuple[int, int] | None = None,
    gap: int = 2,
) -> np.ndarray:
    """Two-channel input: the previous decision's frame and the current one.

    A single frame cannot tell a right-moving bullet from a left-moving one; the
    two-frame stack is the cheapest way to give the student that information.
    """

    prev_frame = max(0, int(frame) - gap)
    previous_xy = previous_xy or xy
    previous = render_state(scenario, prev_frame, previous_xy[0], previous_xy[1], shape=shape)
    current = render_state(scenario, frame, xy[0], xy[1], shape=shape)
    return np.stack([previous, current], axis=0)
