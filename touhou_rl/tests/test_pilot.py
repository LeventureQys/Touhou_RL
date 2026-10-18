"""Unit tests for the scripted avoidance teacher (``touhou_rl.pilot``).

These run without the game: the pilot only needs an observation dict, so the
planner can be regression-tested on synthetic bullet layouts.  The properties
checked here are the ones that would silently destroy the teacher's usefulness:

* the action is always a valid ``[move, shoot, focus, bomb]`` tuple;
* a bullet falling onto the player makes the pilot move *away* (never into it);
* an unavoidable situation spends a bomb, and only when one is available;
* identical inputs produce identical outputs (so DAgger labels are reproducible).
"""

from __future__ import annotations

import numpy as np

from touhou_rl import spec
from touhou_rl.pilot import AvoidPilot, PilotConfig


def _observation(px: float = 192.0, py: float = 384.0, bullets=None) -> dict:
    feat = np.zeros(24, dtype=np.float32)
    feat[0] = (px - spec.PLAYFIELD_X) / spec.PLAYFIELD_W
    feat[1] = (py - spec.PLAYFIELD_Y) / spec.PLAYFIELD_H
    encoded = np.zeros((64, 6), dtype=np.float16)
    if bullets is not None:
        for index, row in enumerate(bullets):
            encoded[index] = row
    return {
        "self_feat": feat,
        "bullets": encoded,
        "danger": np.zeros((16, 8), dtype=np.uint8),
    }


def _raw(px: float = 192.0, py: float = 384.0, bullets=()) -> dict:
    rows = np.array(bullets, dtype=np.float32).reshape(-1, 4) if len(bullets) else np.zeros((0, 4))
    return {
        "position": (rows[:, :2] + np.array([px, py], dtype=np.float32)).astype(np.float32),
        "velocity": rows[:, 2:].astype(np.float32),
        "half": np.full(rows.shape[0], 2.0, dtype=np.float32),
    }


def test_action_shape_and_shot():
    pilot = AvoidPilot(PilotConfig())
    action = pilot.act(_observation())
    assert action.shape == (4,)
    assert 0 <= action[0] <= 8
    assert action[1] == 1  # always_shoot


def test_moves_away_from_incoming_bullet():
    """A bullet 48 px above falling straight down must not make the pilot move up."""
    pilot = AvoidPilot(PilotConfig())
    raw = _raw(bullets=[(0.0, -48.0, 0.0, 3.0), (10.0, -60.0, 0.0, 3.0)])
    action = pilot.act(_observation(), raw)
    assert int(action[0]) not in (1, 2, 8), "pilot moved into the incoming bullet"


def test_clear_field_prefers_no_bomb():
    pilot = AvoidPilot(PilotConfig())
    action = pilot.act(_observation(), _raw(), bombs=3)
    assert int(action[3]) == 0


def test_bombs_when_unavoidable():
    pilot = AvoidPilot(PilotConfig())
    # A dense grid of *static* bullets covering the whole planning window: every
    # reachable position inside the horizon is already inside a bullet, so the
    # only way out is a bomb.
    grid = [
        (float(x), float(y), 0.0, 0.0)
        for x in range(-20, 21, 2)
        for y in range(-20, 21, 2)
    ]
    action = pilot.act(_observation(), _raw(bullets=grid), bombs=2)
    assert int(action[3]) == 1


def test_no_bomb_without_resource():
    pilot = AvoidPilot(PilotConfig())
    grid = [
        (float(x), float(y), 0.0, 0.0)
        for x in range(-20, 21, 2)
        for y in range(-20, 21, 2)
    ]
    action = pilot.act(_observation(), _raw(bullets=grid), bombs=0)
    assert int(action[3]) == 0


def test_aims_at_boss_when_field_is_clear():
    """With no bullets, the only thing that matters is the boss column."""
    observation = _observation()  # player at (192, 384)
    # aim_weight defaults to 0 (it measured worse on the live game); the aiming
    # mechanism itself is tested by enabling it explicitly.
    pilot = AvoidPilot(PilotConfig(aim_weight=0.6))
    action = pilot.act(observation, _raw(), bosses=np.array([[300.0, 120.0]]))
    assert int(action[0]) in (2, 3, 4), "pilot did not move towards the boss on its right"

    pilot_left = AvoidPilot(PilotConfig(aim_weight=0.6))
    action_left = pilot_left.act(observation, _raw(), bosses=np.array([[80.0, 120.0]]))
    assert int(action_left[0]) in (6, 7, 8), "pilot did not move towards the boss on its left"


def test_aim_does_not_override_danger():
    """A boss on the right must not drag the pilot into a bullet wall on the right."""
    wall = [(60.0 + 0.0, dy, 0.0, 0.0) for dy in np.linspace(-60.0, 60.0, 25)]
    pilot = AvoidPilot(PilotConfig(aim_weight=0.6))
    action = pilot.act(_observation(), _raw(bullets=wall), bosses=np.array([[300.0, 120.0]]))
    assert int(action[0]) not in (2, 3, 4), "aim term dragged the pilot into the bullet wall"


def test_deterministic():
    raw = _raw(bullets=[(-20.0, -80.0, 1.0, 2.5), (30.0, -120.0, -1.5, 2.0)])
    first = AvoidPilot(PilotConfig()).act(_observation(), raw)
    second = AvoidPilot(PilotConfig()).act(_observation(), raw)
    assert np.array_equal(first, second)


def test_observation_path_matches_raw_path_semantics():
    """The decoded-observation path must not invert the vertical axis."""
    encoded_bullet = (0.0, -48.0 / 224.0, 0.0, 3.0 / 6.0, 2.0 / 8.0, 1.0)
    pilot = AvoidPilot(PilotConfig())
    action_obs = pilot.act(_observation(bullets=[encoded_bullet]))
    pilot_raw = AvoidPilot(PilotConfig())
    action_raw = pilot_raw.act(_observation(), _raw(bullets=[(0.0, -48.0, 0.0, 3.0)]))
    assert int(action_obs[0]) == int(action_raw[0])
