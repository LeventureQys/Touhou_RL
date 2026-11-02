"""Independent evacuation-event labelling (B-02) from a privileged snapshot.

The acceptance asks for "predefined, independently verified evacuable events and a
timely-evacuation rate".  This module defines an event **without** using the
controller's planner:

* *danger*: holding still for ``horizon`` frames ends with the hurtbox overlapping
  a hazard (linear extrapolation of the snapshot);
* *exit exists*: at least one of the eight directions keeps the hurtbox clear for
  the whole ``horizon``;
* *event* = danger and exit exists;
* *timely*: the action the pilot actually took is one of those safe directions.

The 1-frame input settle is ignored, which makes the check slightly optimistic for
"hold still" and slightly strict for the action (it is evaluated as if it applied
immediately); both biases apply equally to both arms, and the number is reported as
a *label*, not as a physical proof.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import spec
from . import geometry

#: Directions used for the exit probe: (focus, move) for move 0..8.
DIRECTIONS = tuple((focus, move) for focus in (0, 1) for move in range(9))


@dataclass(frozen=True)
class EvacuationEvent:
    danger: bool
    exit_exists: bool
    safe_directions: tuple[tuple[int, int], ...]
    chosen_safe: bool | None
    hold_clearance: float
    best_clearance: float

    @property
    def is_event(self) -> bool:
        return self.danger and self.exit_exists


def _arrays(hazards) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not hazards:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
        )
    positions = np.asarray([h.position for h in hazards], dtype=np.float64)
    velocities = np.asarray([h.velocity for h in hazards], dtype=np.float64)
    halves = np.asarray([h.half for h in hazards], dtype=np.float64)
    return positions, velocities, halves


def clearance_at(
    player: tuple[float, float],
    positions: np.ndarray,
    velocities: np.ndarray,
    halves: np.ndarray,
    frames: int,
    hurtbox_half: float,
) -> np.ndarray:
    """Signed box gap for every frame in ``0..frames`` (min over hazards)."""

    if positions.shape[0] == 0:
        return np.full((frames + 1,), np.inf)
    point = np.asarray(player, dtype=np.float64)
    shifts = velocities[None, :, :] * np.arange(frames + 1)[:, None, None]
    moved = positions[None, :, :] + shifts
    gap = np.abs(point[None, None, :] - moved) - (halves + hurtbox_half)[None, :, :]
    return gap.max(axis=2).min(axis=1)


def path_clearance(
    start: tuple[float, float],
    velocity: tuple[float, float],
    positions: np.ndarray,
    velocities: np.ndarray,
    halves: np.ndarray,
    frames: int,
    hurtbox_half: float,
    model: geometry.MoveModel,
) -> float:
    """Minimum clearance along the straight path implied by ``velocity``."""

    path = []
    x, y = start
    for _ in range(frames + 1):
        path.append((x, y))
        x, y = model.clamp(x + velocity[0], y + velocity[1])
    points = np.asarray(path, dtype=np.float64)
    if positions.shape[0] == 0:
        return float("inf")
    # Hazards move too: a static comparison would call a closing bullet harmless.
    shifts = velocities[None, :, :] * np.arange(frames + 1)[:, None, None]
    moved = positions[None, :, :] + shifts
    gap = np.abs(points[:, None, :] - moved) - (halves + hurtbox_half)[None, :, :]
    return float(gap.max(axis=2).min())


def classify(
    player: tuple[float, float],
    hazards,
    chosen_action: tuple[int, int, int, int] | None,
    *,
    frames: int = 30,
    hurtbox_half: float = 0.825,
    model: geometry.MoveModel | None = None,
    tolerance: float = 0.5,
) -> EvacuationEvent:
    """Label one decision: is there an evacuable danger, and did the pilot take it?"""

    model = model or geometry.MoveModel()
    positions, velocities, halves = _arrays(hazards)
    hold = clearance_at(player, positions, velocities, halves, frames, hurtbox_half)
    danger = bool(hold.min() <= tolerance)
    safe: list[tuple[int, int]] = []
    best = -float("inf")
    for focus, move in DIRECTIONS:
        velocity = model.displacement(focus, move, frames=1)
        value = path_clearance(
            player, velocity, positions, velocities, halves, frames, hurtbox_half, model
        )
        best = max(best, value)
        if value > tolerance:
            safe.append((focus, move))
    chosen_safe = None
    if chosen_action is not None:
        focus, move = int(chosen_action[2]), int(chosen_action[0])
        chosen_safe = (
            path_clearance(
                player,
                model.displacement(focus, move, frames=1),
                positions,
                velocities,
                halves,
                frames,
                hurtbox_half,
                model,
            )
            > tolerance
        )
    return EvacuationEvent(
        danger=danger,
        exit_exists=bool(safe),
        safe_directions=tuple(safe),
        chosen_safe=chosen_safe,
        hold_clearance=float(hold.min()),
        best_clearance=float(best),
    )


def summarise(events: list[EvacuationEvent]) -> dict:
    """B-02's rate over *events* (danger with an exit), not over all decisions."""

    considered = [event for event in events if event.is_event and event.chosen_safe is not None]
    timely = sum(1 for event in considered if event.chosen_safe)
    return {
        "decisions": len(events),
        "events": len(considered),
        "timely": timely,
        "rate": round(timely / len(considered), 4) if considered else None,
        "mean_safe_directions": (
            round(float(np.mean([len(event.safe_directions) for event in considered])), 2)
            if considered
            else None
        ),
    }
