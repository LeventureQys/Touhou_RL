"""Replayable v3.7.3 development worlds, NOT pixel/game calibration.

Internal contracts deliberately isolated from tactical_contracts. Horizontal-only
fixed-y diagnostics; 2 logical frames per decision, no input-settle assumption.
F2/F3 births aim at the actual player at emission, never a fixed future replay.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from math import hypot
import hashlib
import json

ACTIONS = (-2, -1, 0, 1, 2)  # horizontal speed = action * 2 px/frame
FAMILIES = tuple(f"F{i}" for i in range(1, 7))

@dataclass(frozen=True)
class Scenario:
    event_id: str
    family: str
    template_group: str
    seed: int
    start_x: float
    mirror: bool = False
    horizon: int = 160
    fire_interval: int = 16
    bullet_speed: float = 8.0
    wall_frame: int = 64
    internal_gap: bool = False
    model_known: bool = True
    split: str = "development"
    depletion_pressure: bool = False

    def canonical_x(self, x):
        return 384.0 - x if self.mirror else x

    def firing(self, frame):
        if self.family not in ("F2", "F3", "F6"):
            return False
        if self.family == "F3" and 33 <= frame < 104:
            return False
        return frame % self.fire_interval == 0 and frame < self.horizon - 30

@dataclass(frozen=True)
class Bullet:
    born: int
    x: float
    y: float
    vx: float
    vy: float
    aim_x: float

@dataclass(frozen=True)
class WorldState:
    frame: int
    x: float
    bullets: tuple[Bullet, ...] = ()
    hit: bool = False
    emissions: tuple[dict, ...] = ()


def development_scenarios(per_family=12):
    """Only development instances; no fake independent validation/test splits."""
    if not 1 <= per_family <= 12:
        raise ValueError("development count must be in 1..12")
    result = []
    for family in FAMILIES:
        for i in range(per_family):
            mirror = bool(i % 2)
            x = 336.0 if family == "F3" else 192.0 + (i // 2 % 3 - 1) * 8
            result.append(Scenario(f"dev-{family}-{i:02d}", family,
                f"dev-{family}-template-{i//2}", 7300+i, 384-x if mirror else x,
                mirror, fire_interval=16 + 2 * (i//4), bullet_speed=8.0+i//6,
                wall_frame=64+4*(i//4), internal_gap=family == "F1" and i >= 8,
                model_known=family != "F6"))
    return tuple(result)


def initial_state(scenario):
    return WorldState(0, scenario.start_x)


def static_collision(s, frame, x):
    cx = s.canonical_x(x)
    if s.family == "F3" and s.depletion_pressure:
        if frame < 34 and cx <= 180:
            return True
        if frame >= 104 and cx >= 180:
            return True
    if s.family == "F1" and s.wall_frame <= frame < s.wall_frame+12:
        return 100 <= cx <= 284 and not (s.internal_gap and 180 <= cx <= 204)
    if s.family == "F4":
        return 40 <= frame < 64 and 220 <= cx <= 240
    if s.family == "F5":
        return 20 <= frame < 40 and 212 <= cx <= 222
    return False


def advance(s: Scenario, state: WorldState, action: int, frames=2):
    if action not in ACTIONS or frames < 1:
        raise ValueError("invalid action/hold")
    x, bullets, events, hit = state.x, list(state.bullets), list(state.emissions), state.hit
    for frame in range(state.frame, min(state.frame+frames, s.horizon)):
        # Direction is fixed BEFORE moving this frame; history is actual, not proposed.
        if s.firing(frame):
            origin = (192.0, 100.0)
            dx, dy = x-origin[0], 300.0
            length = hypot(dx, dy)
            b = Bullet(frame, *origin, s.bullet_speed*dx/length,
                       s.bullet_speed*dy/length, x)
            bullets.append(b)
            events.append({"frame": frame, "aim_x": x, "vx": b.vx, "vy": b.vy})
        x = min(376., max(8., x + action*2.))
        for b in bullets:
            age = frame+1-b.born
            if abs(b.x+b.vx*age-x) <= 3.0 and abs(b.y+b.vy*age-400.) <= 3.0:
                hit = True
        hit |= static_collision(s, frame+1, x)
    return WorldState(min(state.frame+frames, s.horizon), x, tuple(bullets), bool(hit), tuple(events))


def replay(s, actions):
    state = initial_state(s)
    trace = [{"frame": 0, "x": state.x, "hit": False}]
    for action in actions:
        if state.frame >= s.horizon:
            break
        state = advance(s, state, action)
        trace.append({"frame": state.frame, "x": state.x, "hit": state.hit, "action": action, "actual_hold": 2})
    return state, trace


def fingerprint(s):
    return hashlib.sha256(json.dumps(asdict(s), sort_keys=True).encode()).hexdigest()
