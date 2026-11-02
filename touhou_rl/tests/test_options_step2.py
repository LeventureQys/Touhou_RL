"""Contracts for the escape-option potential and the danger-grid observation."""

from __future__ import annotations

import numpy as np
import pytest

from touhou_rl import entity_obs, entity_policy, options


def _player() -> np.ndarray:
    return np.array([192.0, 384.0])


def test_option_count_is_full_when_nothing_is_near():
    positions = np.array([[32.0, 16.0]], dtype=np.float64)  # far top-left corner
    assert options.option_count(_player(), positions, np.zeros((1, 2)), np.array([2.0])) == 18


def test_option_count_is_zero_when_surrounded():
    # a dense ring right on top of the player: no direction survives 20 frames
    angles = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 6.0 + _player()
    count = options.option_count(
        _player(), ring, np.zeros_like(ring), np.full(len(ring), 20.0)
    )
    assert count == 0


def test_incoming_wall_reduces_the_count():
    # a horizontal wall above the player closes in at 2 px/frame
    xs = np.linspace(100.0, 290.0, 40)
    wall = np.stack([xs, np.full_like(xs, 300.0)], axis=1)
    velocity = np.tile(np.array([0.0, 2.0]), (40, 1))
    halves = np.full(len(wall), 8.0)
    now = options.option_count(_player(), wall, velocity, halves)
    later = options.option_count(_player(), wall + velocity * 30.0, velocity, halves)
    assert later <= now
    assert now > 0


def test_phi_is_share_of_moves_and_in_range():
    positions = np.array([[32.0, 16.0]], dtype=np.float64)
    value = options.phi(_player(), positions, np.zeros((1, 2)), np.array([2.0]))
    assert value == pytest.approx(1.0)
    assert 0.0 <= value <= 1.0


def test_shaped_reward_is_the_telescoping_difference():
    assert options.shaped_reward(0.2, 0.5, 0.995, terminal=False) == pytest.approx(0.995 * 0.5 - 0.2)
    # terminal states drop the potential (the classic PBRS trap)
    assert options.shaped_reward(0.2, 0.9, 0.995, terminal=True) == pytest.approx(-0.2)


def test_shaped_reward_does_not_depend_on_the_action_taken():
    """PBRS requires a state-only potential: the same two states give the same term."""
    first = options.shaped_reward(0.1, 0.4, 0.99, terminal=False)
    second = options.shaped_reward(0.1, 0.4, 0.99, terminal=False)
    assert first == second


def test_mismatched_bullet_arrays_are_rejected():
    with pytest.raises(ValueError, match="disagree"):
        options.option_count(_player(), np.zeros((2, 2)), np.zeros((3, 2)), np.zeros((2,)))


# --------------------------------------------------------------------------- #
# danger grid
# --------------------------------------------------------------------------- #


def _snapshot(positions, velocities) -> dict:
    return {
        "player_position": [192.0, 384.0],
        "player_half": [0.825, 0.825],
        "stage": 0,
        "bullets": {"position": positions, "velocity": velocities, "half": [2.0] * len(positions)},
    }


def test_grid_has_three_extrapolated_channels():
    positions = [[200.0, 100.0]]
    velocities = [[0.0, 3.0]]
    observation = entity_obs.from_snapshot(_snapshot(positions, velocities))
    grid = observation["grid"]
    assert grid.shape == (entity_obs.GRID_CHANNELS, entity_obs.GRID_ROWS, entity_obs.GRID_COLS)
    assert grid.dtype == np.float32
    # one occupied cell per channel, and it moves down with the extrapolation
    for channel in range(entity_obs.GRID_CHANNELS):
        assert grid[channel].sum() > 0.0
    rows = [int(np.argwhere(grid[channel] > 0)[0][0]) for channel in range(entity_obs.GRID_CHANNELS)]
    assert rows[0] < rows[1] < rows[2]


def test_grid_is_empty_when_there_are_no_bullets():
    observation = entity_obs.from_snapshot(_snapshot([], []))
    assert float(observation["grid"].sum()) == 0.0


def test_grid_saturates_at_the_cap():
    positions = [[200.0, 100.0]] * 20  # 20 bullets in the same cell
    observation = entity_obs.from_snapshot(_snapshot(positions, [[0.0, 0.0]] * 20))
    assert float(observation["grid"][0].max()) == pytest.approx(1.0)


def test_policy_consumes_the_grid_and_rejects_a_wrong_shape():
    model = entity_policy.EntityPolicy(
        entity_policy.EntityPolicyConfig(trunk_hidden=32, object_hidden=16, grid_hidden=16)
    )
    bullets = np.zeros((2, entity_obs.K_BULLETS, entity_obs.BULLET_FEATURES), dtype=np.float32)
    bullets[:, :3, 5] = 1.0
    batch = entity_obs.batch(
        [
            entity_obs.from_snapshot(_snapshot([[200.0, 100.0]], [[0.0, 1.0]])),
            entity_obs.from_snapshot(_snapshot([[200.0, 100.0]], [[0.0, 1.0]])),
        ]
    )
    tensors = entity_policy.to_tensors(batch, "cpu")
    logits, _value = model(tensors)
    assert logits.shape == (2, 18)

    with pytest.raises(ValueError, match="grid must be"):
        model({"bullets": tensors["bullets"], "meta": tensors["meta"], "grid": tensors["grid"][:, :2]})


def test_checkpoint_round_trip_keeps_the_grid_branch(tmp_path):
    model = entity_policy.EntityPolicy(
        entity_policy.EntityPolicyConfig(trunk_hidden=32, object_hidden=16, grid_hidden=16)
    )
    path = entity_policy.save_checkpoint(tmp_path / "p.pt", model)
    restored, _extra = entity_policy.load_checkpoint(path)
    assert restored.config.grid_channels == entity_obs.GRID_CHANNELS
    batch = entity_obs.batch([entity_obs.from_snapshot(_snapshot([[200.0, 100.0]], [[0.0, 1.0]]))])
    first, _ = model(entity_policy.to_tensors(batch, "cpu"))
    second, _ = restored(entity_policy.to_tensors(batch, "cpu"))
    assert np.allclose(first.detach().numpy(), second.detach().numpy())
