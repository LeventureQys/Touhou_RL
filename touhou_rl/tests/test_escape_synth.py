"""Synthetic testbed tests: settle model, clamp, windows, per-frame collision."""

from __future__ import annotations

import numpy as np
import pytest

from touhou_rl.escape import geometry, synth


def _scenario(hazards=(), *, start=(192.0, 384.0), horizon=8, frameskip=2, unsupported=None):
    return synth.Scenario(
        "test/0", "TEST", 0, start, tuple(hazards),
        horizon_frames=horizon, hold_frames=0, frameskip=frameskip, unsupported=unsupported,
    )


def test_first_frame_uses_the_previous_action():
    env = synth.SyntheticEnv(_scenario(horizon=2))
    env.step((0, 3))  # east; frame 1 should still be idle
    assert env.path[1] == pytest.approx((192.0, 384.0))
    assert env.path[2] == pytest.approx((196.0, 384.0))


def test_second_decision_uses_the_new_action_from_frame_one():
    env = synth.SyntheticEnv(_scenario(horizon=4))
    env.step((0, 3))  # east
    env.step((0, 3))  # east again: no change, both frames move
    assert env.path[3] == pytest.approx((200.0, 384.0))
    assert env.path[4] == pytest.approx((204.0, 384.0))


def test_focus_halves_the_step():
    env = synth.SyntheticEnv(_scenario(horizon=2))
    env.step((1, 3))
    assert env.path[2] == pytest.approx((194.0, 384.0))


def test_movement_is_clamped_each_frame():
    env = synth.SyntheticEnv(_scenario(start=(374.0, 430.0), horizon=4))
    env.step((0, 3))
    env.step((0, 3))
    assert env.x <= 376.0
    assert env.y <= 432.0


def test_collision_is_detected_on_the_frame_it_happens():
    # A static box that only exists at frame 2.
    hazard = synth.HazardSpec((196.0, 384.0), (0.0, 0.0), (2.0, 2.0), spawn_frame=2)
    env = synth.SyntheticEnv(_scenario([hazard], horizon=4))
    env.step((0, 3))  # frame 1: idle, frame 2: x=196 -> hit
    assert env.hit
    assert env.frame == 2


def test_spawn_window_lets_a_cell_clear_later():
    hazard = synth.HazardSpec((196.0, 384.0), (0.0, 0.0), (2.0, 2.0), despawn_frame=1)
    env = synth.SyntheticEnv(_scenario([hazard], horizon=4))
    env.step((0, 3))
    assert not env.hit


def test_simulate_reports_survival_and_clearance():
    outcome = synth.simulate(_scenario(horizon=6), [(0, 0)] * 3)
    assert outcome["survived"] is True
    assert outcome["min_clearance"] is None  # no hazards


def test_initial_overlap_is_reported():
    hazard = synth.HazardSpec((192.0, 384.0), (0.0, 0.0), (2.0, 2.0))
    assert synth.initial_clearance(_scenario([hazard])) <= 0.0


def test_scenario_signature_is_stable_and_seed_sensitive():
    from touhou_rl.escape import families

    first = synth.scenario_signature(families.build("SC-01", 3))
    second = synth.scenario_signature(families.build("SC-01", 3))
    other = synth.scenario_signature(families.build("SC-01", 4))
    assert first == second
    assert first != other


def test_advance_matches_env():
    scenario = _scenario([synth.HazardSpec((300.0, 200.0), (0.0, 1.0), (2.0, 2.0))], horizon=4)
    env = synth.SyntheticEnv(scenario)
    expected = synth.advance(scenario, 0, *scenario.player_start, (0, 3), (0, 0))
    env.step((0, 3))
    assert env.x == pytest.approx(expected.x)
    assert env.y == pytest.approx(expected.y)
    assert env.frame == expected.frame


def test_bad_scenarios_are_rejected():
    with pytest.raises(ValueError):
        synth.Scenario("bad", "TEST", 0, (192.0, 384.0), (), horizon_frames=1, frameskip=2)
    with pytest.raises(ValueError):
        synth.Scenario("bad", "TEST", 0, (192.0, 384.0),
                       (synth.HazardSpec((float("nan"), 0.0)),))
    with pytest.raises(ValueError):
        synth.Scenario("bad", "TEST", 0, (192.0, 384.0),
                       (synth.HazardSpec((0.0, 0.0), half=(0.0, 2.0)),))
