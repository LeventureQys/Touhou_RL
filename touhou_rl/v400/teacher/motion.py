"""Exact self-ship motion model (11-P1 §2.2).

The four speeds have one definition, :class:`touhou_rl.v400.teacher.geometry.SpeedTable`
(this module re-exports it), because ``hazard_tape`` imports the table from
``geometry`` and a second class with the same fields is exactly the kind of
duplication that broke that import.  What stays here is the movement model:
:func:`split_cand` and :func:`step_delta`.

Nothing here is a constant of the game: the four speeds are read out of the
``PLAYER`` reply, which the engine fills from the loaded ``.sht`` files
(Player.cpp:783-806).
"""

from __future__ import annotations

from touhou_rl.v400.teacher.geometry import SpeedTable

#: Direction table, index == ``cand % 9``; ``+y`` points down (03 §3.2).
#: Index 0 is idle, then N, NE, E, SE, S, SW, W, NW.
MOVES = (
    (0, 0),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
)


def split_cand(cand: int) -> tuple[int, int]:
    """``cand = focus * 9 + move`` -> ``(move, focus)``."""
    if not 0 <= cand < 18:
        raise ValueError(f"candidate out of range: {cand}")
    focus, move = divmod(cand, 9)
    return move, focus


def step_delta(cand: int, speeds: SpeedTable) -> tuple[float, float]:
    """One frame of displacement for a candidate action.

    A single-axis direction moves by the axial speed; a diagonal moves by the
    diagonal speed on **both** components, so its magnitude is the axial speed
    (2.8284 = 4/√2 for team 0).  Idle does not move.  Focus changes the speed
    only — it never changes the hurtbox size (11-P1 §2.1).
    """
    move, focus = split_cand(cand)
    sx, sy = MOVES[move]
    if sx == 0 and sy == 0:
        return (0.0, 0.0)
    axial, diagonal = (
        (speeds.focused_axial, speeds.focused_diagonal)
        if focus
        else (speeds.normal_axial, speeds.normal_diagonal)
    )
    if sx and sy:
        return (sx * diagonal, sy * diagonal)
    return (sx * axial, sy * axial)
