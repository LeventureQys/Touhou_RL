"""Deterministic synthetic testbed for v3.7.0 (D2).

Design constraints that shaped this file
----------------------------------------
1. **The game's collision predicate is per-frame AABB**, applied once per logic
   frame after the bullet moves (``BulletManager.cpp:865`` then ``:933``).  The
   testbed therefore advances players *and* hazards frame by frame and tests
   box overlap at those discrete instants.  A swept test would be stricter than
   the game and would make "no route" claims wrong.
2. **The first frame of a decision runs the previous action** (measured in D1:
   96.62% exact reconstruction of 2,903 recorded pairs).  The testbed models it,
   because a planner that assumes otherwise is planning a path the game will not
   execute.
3. **Hazards are linear with optional spawn/despawn windows**, which is enough
   for walls, fans, rings, second waves and "position occupied then vacated".
   Accelerating/turning bullets are *not* modelled; scenarios that need them are
   marked unsupported instead of being silently approximated.
4. Everything is a pure function of ``(family, seed)``: no wall clock, no global
   RNG, no game process.

The testbed is the ground truth for *physics*; it is deliberately not the
subject planner.  The reference solver (``reference.py``) shares this physics
because that is the definition of the world, but implements its own search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import spec
from . import geometry
from .contracts import HURTBOX_HALF_PX

#: How many frames after a key change still use the previous action (D1).
SETTLE_FRAMES = 1

MAX_DESPAWN = 1_000_000


@dataclass(frozen=True)
class HazardSpec:
    """One linear hazard with an optional visibility window."""

    position: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    half: tuple[float, float] = (2.0, 2.0)
    spawn_frame: int = 0
    despawn_frame: int = MAX_DESPAWN

    def active(self, frame: int) -> bool:
        return self.spawn_frame <= frame < self.despawn_frame

    def at(self, frame: int) -> tuple[float, float]:
        return (
            self.position[0] + self.velocity[0] * frame,
            self.position[1] + self.velocity[1] * frame,
        )


@dataclass(frozen=True)
class Scenario:
    """A single deterministic instance."""

    scenario_id: str
    family: str
    seed: int
    player_start: tuple[float, float]
    hazards: tuple[HazardSpec, ...]
    horizon_frames: int = 180
    hold_frames: int = 60
    frameskip: int = 2
    notes: str = ""
    #: Reference label for the edge instances; ``None`` means "not hand-labelled".
    expected: str | None = None
    #: Non-empty when the scenario needs a hazard class this testbed cannot model.
    unsupported: str | None = None

    def __post_init__(self) -> None:
        if self.frameskip < 1:
            raise ValueError("frameskip must be >= 1")
        if self.horizon_frames < self.frameskip:
            raise ValueError("horizon must cover at least one decision")
        for hazard in self.hazards:
            if not all(math.isfinite(value) for value in (*hazard.position, *hazard.velocity, *hazard.half)):
                raise ValueError(f"{self.scenario_id}: non-finite hazard")
            if hazard.half[0] <= 0.0 or hazard.half[1] <= 0.0:
                raise ValueError(f"{self.scenario_id}: hazard half extents must be positive")
        # Precompute flat arrays: the reference solver calls this millions of
        # times, and a per-hazard Python loop dominated the runtime (measured:
        # >2 minutes for one 60-frame instance instead of milliseconds).
        count = len(self.hazards)
        positions = np.asarray([h.position for h in self.hazards], dtype=np.float64).reshape(count, 2)
        velocities = np.asarray([h.velocity for h in self.hazards], dtype=np.float64).reshape(count, 2)
        halves = np.asarray([h.half for h in self.hazards], dtype=np.float64).reshape(count, 2)
        spawn = np.asarray([h.spawn_frame for h in self.hazards], dtype=np.int64)
        despawn = np.asarray([h.despawn_frame for h in self.hazards], dtype=np.int64)
        for array in (positions, velocities, halves, spawn, despawn):
            array.setflags(write=False)
        object.__setattr__(self, "_positions", positions)
        object.__setattr__(self, "_velocities", velocities)
        object.__setattr__(self, "_halves", halves)
        object.__setattr__(self, "_spawn", spawn)
        object.__setattr__(self, "_despawn", despawn)
        object.__setattr__(self, "_cache", {})

    # -- derived views ---------------------------------------------------
    @property
    def spawn_frames(self) -> np.ndarray:
        return self._spawn

    @property
    def despawn_frames(self) -> np.ndarray:
        return self._despawn

    def active(self, frame: int) -> np.ndarray:
        return (self._spawn <= frame) & (frame < self._despawn)

    def positions(self, frame: int) -> np.ndarray:
        return self._positions + self._velocities * float(frame)

    def velocities(self) -> np.ndarray:
        return self._velocities

    def halves(self) -> np.ndarray:
        return self._halves

    def active_positions(self, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._cache.get(frame)
        if cached is None:
            mask = self.active(frame)
            positions = (self._positions + self._velocities * float(frame))[mask]
            cached = (positions, self._velocities[mask], self._halves[mask])
            self._cache[frame] = cached
        return cached

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "family": self.family,
            "seed": self.seed,
            "player_start": list(self.player_start),
            "horizon_frames": self.horizon_frames,
            "hold_frames": self.hold_frames,
            "frameskip": self.frameskip,
            "hazards": len(self.hazards),
            "notes": self.notes,
            "expected": self.expected,
            "unsupported": self.unsupported,
        }


@dataclass(frozen=True)
class StepResult:
    frame: int
    position: tuple[float, float]
    focus: int
    clearance: float
    hit: bool
    done: bool


@dataclass(frozen=True)
class AdvanceResult:
    """Outcome of executing one decision from an explicit state (no env object)."""

    frame: int
    x: float
    y: float
    hit: bool
    hit_frame: int | None
    min_clearance: float
    path: tuple[tuple[float, float], ...]


def action_velocity(model: geometry.MoveModel, action: tuple[int, int]) -> np.ndarray:
    """``(focus, move)`` -> per-frame velocity vector (px/frame)."""
    focus, move = action
    displacement = model.displacement(focus, move, frames=1)
    return displacement


def frame_clearance(scenario: Scenario, frame: int, x: float, y: float) -> float:
    positions, _velocities, halves = scenario.active_positions(frame)
    if positions.shape[0] == 0:
        return float("inf")
    gap = np.abs(np.array([x, y])[None, :] - positions) - (HURTBOX_HALF_PX + halves)
    return float(np.max(gap, axis=1).min())


def advance(
    scenario: Scenario,
    frame: int,
    x: float,
    y: float,
    action: tuple[int, int],
    previous_action: tuple[int, int],
    *,
    model: geometry.MoveModel | None = None,
    settle_frames: int = SETTLE_FRAMES,
) -> AdvanceResult:
    """Execute one decision exactly as the game would (settle + clamp + per-frame AABB)."""
    model = model or geometry.MoveModel()
    velocity = action_velocity(model, (int(action[0]), int(action[1])))
    previous_velocity = action_velocity(model, (int(previous_action[0]), int(previous_action[1])))
    knots: list[tuple[float, float]] = [(x, y)]
    best = frame_clearance(scenario, frame, x, y)
    if best <= 0.0:
        # Already overlapping at the decision instant: the previous frame's check
        # would have killed the player, so the decision cannot help.
        return AdvanceResult(frame, x, y, True, frame, best, tuple(knots))
    hit = False
    hit_frame: int | None = None
    for step in range(1, scenario.frameskip + 1):
        delta = previous_velocity if step <= settle_frames else velocity
        x, y = model.clamp(x + delta[0], y + delta[1])
        frame += 1
        knots.append((x, y))
        clearance = frame_clearance(scenario, frame, x, y)
        best = min(best, clearance)
        if clearance <= 0.0:
            hit = True
            hit_frame = frame
            break
    return AdvanceResult(frame, x, y, hit, hit_frame, best, tuple(knots))


class SyntheticEnv:
    """Frame-accurate subset of the game: linear hazards, AABB hits, 1-frame settle."""

    def __init__(
        self,
        scenario: Scenario,
        model: geometry.MoveModel | None = None,
        settle_frames: int = SETTLE_FRAMES,
    ) -> None:
        self.scenario = scenario
        self.model = model or geometry.MoveModel()
        self.settle_frames = int(settle_frames)
        self.reset()

    # -- lifecycle -------------------------------------------------------
    def reset(self) -> dict:
        self.frame = 0
        self.x, self.y = self.scenario.player_start
        self.x, self.y = self.model.clamp(self.x, self.y)
        self.focus = 0
        self.previous_action: tuple[int, int] = (0, 0)
        self.hit = False
        self.min_clearance = float("inf")
        self.path: list[tuple[float, float]] = [(self.x, self.y)]
        self.actions: list[tuple[int, int]] = []
        self._check(0)
        return self.observation()

    def observation(self) -> dict:
        positions, velocities, halves = self.scenario.active_positions(self.frame)
        return {
            "frame": self.frame,
            "player": (self.x, self.y, self.focus),
            "hazards": positions.shape[0],
            "positions": positions,
            "velocities": velocities,
            "halves": halves,
            "hit": self.hit,
        }

    # -- physics ---------------------------------------------------------
    def _clearance(self, frame: int, x: float, y: float) -> float:
        return frame_clearance(self.scenario, frame, x, y)

    def _check(self, frame: int) -> None:
        clearance = self._clearance(frame, self.x, self.y)
        self.min_clearance = min(self.min_clearance, clearance)
        if clearance <= 0.0:
            self.hit = True

    def step(self, action: tuple[int, int]) -> dict:
        if self.hit:
            return self.observation()
        result = advance(
            self.scenario,
            self.frame,
            self.x,
            self.y,
            action,
            self.previous_action,
            model=self.model,
            settle_frames=self.settle_frames,
        )
        self.frame, self.x, self.y = result.frame, result.x, result.y
        self.path.extend(result.path[1:])
        self.min_clearance = min(self.min_clearance, result.min_clearance)
        self.hit = result.hit
        self.focus = int(action[0])
        self.previous_action = (int(action[0]), int(action[1]))
        self.actions.append((int(action[0]), int(action[1])))
        return self.observation()

    def done(self) -> bool:
        return self.hit or self.frame >= self.scenario.horizon_frames


def simulate(
    scenario: Scenario,
    actions: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    model: geometry.MoveModel | None = None,
) -> dict:
    """Replay an action sequence exactly; return the outcome dict."""
    env = SyntheticEnv(scenario, model=model)
    for action in actions:
        if env.done():
            break
        env.step(action)
    return {
        "scenario_id": scenario.scenario_id,
        "frames": env.frame,
        "hit": env.hit,
        "hit_frame": env.frame if env.hit else None,
        "min_clearance": None if env.min_clearance == float("inf") else round(env.min_clearance, 4),
        "survived": (not env.hit) and env.frame >= scenario.horizon_frames,
        "path": env.path,
        "actions": env.actions,
    }


def initial_clearance(scenario: Scenario) -> float:
    return frame_clearance(scenario, 0, *scenario.player_start)


def scenario_signature(scenario: Scenario) -> str:
    """Stable short hash used to prove two generators agree."""
    import hashlib
    import json

    payload = {
        "id": scenario.scenario_id,
        "family": scenario.family,
        "seed": scenario.seed,
        "start": list(scenario.player_start),
        "hazards": [
            [list(h.position), list(h.velocity), list(h.half), h.spawn_frame, h.despawn_frame]
            for h in scenario.hazards
        ],
        "horizon": scenario.horizon_frames,
        "hold": scenario.hold_frames,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def move_area_rect() -> tuple[float, float, float, float]:
    return (spec.MOVE_AREA_X, spec.MOVE_AREA_Y, spec.MOVE_AREA_W, spec.MOVE_AREA_H)
