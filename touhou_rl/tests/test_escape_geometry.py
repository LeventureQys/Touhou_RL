"""Geometry tests: measured speeds, inclusive clamp, exact swept box collision.

The last two tests are the reason this module exists: the old pilot samples
clearance every ``frameskip`` frames, and a bullet can cross the hurtbox between
two such samples.  The swept test must catch it; the sampled grid must not.
"""

from __future__ import annotations

import numpy as np
import pytest

from touhou_rl.escape import contracts, geometry


def test_default_speeds_match_the_measured_calibration():
    model = geometry.MoveModel()
    assert model.axis == contracts.MOVEMENT_MODEL["axis"] == 4.0
    assert model.focus_axis == contracts.MOVEMENT_MODEL["focus_axis"] == 2.0
    assert model.diagonal == contracts.MOVEMENT_MODEL["diagonal"] == 2.8284
    assert model.focus_diagonal == contracts.MOVEMENT_MODEL["focus_diagonal"] == 1.4142
    assert model.hurtbox_half == contracts.HURTBOX_HALF_PX
    assert abs(model.diagonal - model.axis / np.sqrt(2.0)) < 1e-3
    assert abs(model.focus_diagonal - model.focus_axis / np.sqrt(2.0)) < 1e-3


def test_diagonal_sets_both_axes():
    model = geometry.MoveModel()
    assert model.displacement(0, 2, frames=1).tolist() == pytest.approx([2.8284, -2.8284])
    assert model.displacement(1, 4, frames=1).tolist() == pytest.approx([1.4142, 1.4142])


def test_displacement_scales_with_frames_and_respects_focus():
    model = geometry.MoveModel()
    assert model.displacement(0, 3, frames=2).tolist() == pytest.approx([8.0, 0.0])
    assert model.displacement(1, 3, frames=2).tolist() == pytest.approx([4.0, 0.0])
    assert model.displacement(0, 0, frames=10).tolist() == [0.0, 0.0]
    with pytest.raises(ValueError):
        model.displacement(0, 3, frames=0)
    with pytest.raises(ValueError):
        model.displacement(2, 3, frames=1)


def test_action_order_matches_pilot_encoding():
    model = geometry.MoveModel()
    table = model.action_displacements(frames=2)
    assert table.shape == (18, 2)
    assert table[0].tolist() == [0.0, 0.0]  # focus 0, idle
    assert table[9].tolist() == [0.0, 0.0]  # focus 1, idle
    assert table[3].tolist() == pytest.approx([8.0, 0.0])  # focus 0, east
    assert table[12].tolist() == pytest.approx([4.0, 0.0])  # focus 1, east
    assert table[2].tolist() == pytest.approx([5.6568, -5.6568])


def test_centre_clamp_is_inclusive_like_the_game():
    model = geometry.MoveModel()
    assert model.clamp(376.0, 432.0) == (376.0, 432.0)
    assert model.clamp(8.0, 16.0) == (8.0, 16.0)
    assert model.clamp(400.0, 500.0) == (376.0, 432.0)
    assert model.clamp(-10.0, 0.0) == (8.0, 16.0)
    # Walking into the wall does not overshoot and does not get an extra inset.
    assert model.apply(374.0, 400.0, 0, 3, frames=1) == (376.0, 400.0)


def test_box_gap_sign_and_axis_choice():
    assert geometry.box_gap((0, 0), (0, 0), (1, 1), (1, 1)) == -2.0
    assert geometry.box_gap((0, 0), (5, 0), (1, 1), (1, 1)) == 3.0
    # Diagonal separation is decided by the better axis, not by the Euclidean gap.
    assert geometry.box_gap((0, 0), (4, 4), (1, 1), (1, 1)) == 2.0


def test_circle_approximation_over_estimates_diagonal_clearance():
    player_half = geometry.MoveModel().hurtbox_half
    bullet_half = 2.0
    radius_sum = player_half + bullet_half
    # Place the bullet exactly on the 45-degree contact line: box gap is 0.
    offset = np.array([radius_sum, radius_sum])
    box = geometry.box_gap((0, 0), offset, (player_half, player_half), (bullet_half, bullet_half))
    assert box == pytest.approx(0.0)
    geometric = float(np.linalg.norm(offset)) - radius_sum
    assert geometric == pytest.approx((np.sqrt(2.0) - 1.0) * radius_sum, abs=1e-12)
    assert geometric > 1.1
    # What the old pilot would report at the same instant, pad included.
    pilot_clearance = float(np.linalg.norm(offset)) - (radius_sum + 0.5)
    assert pilot_clearance > 0.6  # still "safe" while the boxes are touching
    assert pilot_clearance < geometric  # the 0.5 px pad only absorbs part of it


def test_swept_collision_catches_a_crossing_between_samples():
    player = np.array([192.0, 384.0])
    player_half = geometry.MoveModel().hurtbox_half
    start = np.array([163.0, 384.0])
    velocity = np.array([10.0, 0.0])
    half = np.array([2.0, 2.0])

    swept = geometry.swept_box_clearance(player, np.zeros(2), start, velocity, player_half, half, 20.0)
    assert swept < 0.0

    # What a per-step (frameskip=2) evaluation can see: positive clearance everywhere.
    sampled = min(
        float(np.abs(player[0] - (start[0] + velocity[0] * step)) - (player_half + half[0]))
        for step in range(0, 21, 2)
    )
    assert sampled > 5.0
    assert geometry.segment_hits(player, np.zeros(2), start, velocity, player_half, half, 20.0)


def test_swept_collision_has_no_false_positive_when_passing_nearby():
    player = np.array([192.0, 384.0])
    player_half = geometry.MoveModel().hurtbox_half
    start = np.array([150.0, 384.0 - 3.5])
    velocity = np.array([10.0, 0.0])
    clearance = geometry.swept_box_clearance(
        player, np.zeros(2), start, velocity, player_half, np.array([2.0, 2.0]), 20.0
    )
    assert clearance == pytest.approx(3.5 - (player_half + 2.0))
    assert not geometry.segment_hits(
        player, np.zeros(2), start, velocity, player_half, np.array([2.0, 2.0]), 20.0
    )


def test_swept_minimum_matches_dense_sampling():
    rng = np.random.default_rng(7)
    player_half = geometry.MoveModel().hurtbox_half
    for _ in range(50):
        start = rng.uniform(-40.0, 40.0, size=2)
        velocity = rng.uniform(-6.0, 6.0, size=2)
        player_velocity = rng.uniform(-4.0, 4.0, size=2)
        half = rng.uniform(0.5, 4.0, size=2)
        frames = 10.0
        exact = geometry.swept_box_clearance(
            np.zeros(2), player_velocity, start, velocity, player_half, half, frames
        )
        times = np.linspace(0.0, frames, 4001)
        d = -start + (player_velocity - velocity) * times[:, None]
        sampled = float(np.min(np.max(np.abs(d) - (player_half + half), axis=1)))
        assert exact <= sampled + 1e-9
        assert sampled - exact < 1e-2


def test_path_min_clearance_reports_the_segment():
    path = np.repeat(np.array([[192.0, 384.0]]), 21, axis=0)
    clearance, segment, hazard = geometry.path_min_clearance(
        path,
        np.array([[163.0, 384.0]]),
        np.array([[10.0, 0.0]]),
        np.array([[2.0, 2.0]]),
        geometry.MoveModel().hurtbox_half,
    )
    assert clearance < 0.0
    assert segment in (2, 3)  # t ~ 2.9 crosses the hurtbox
    assert hazard == 0


def test_empty_field_is_infinite_clearance():
    path = np.repeat(np.array([[192.0, 384.0]]), 21, axis=0)
    clearance, segment, hazard = geometry.path_min_clearance(
        path,
        np.zeros((0, 2)),
        np.zeros((0, 2)),
        np.zeros((0, 2)),
        geometry.MoveModel().hurtbox_half,
    )
    assert clearance == float("inf")
    assert (segment, hazard) == (-1, -1)


def test_bad_shapes_are_rejected():
    with pytest.raises(ValueError):
        geometry.path_segment_clearances(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), 0.825)
    with pytest.raises(ValueError):
        geometry.MoveModel(frameskip=0)
