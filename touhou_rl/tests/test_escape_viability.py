"""Offline tests for v3.7.2's positive-only macro witnesses (no game IO)."""
import unittest

import numpy as np

from touhou_rl.escape.forecast import DangerField, ForecastConfig
from touhou_rl.escape.geometry import MoveModel
from touhou_rl.escape.viability import (
    ACTIONS, ViabilityConfig, _Node, _diverse, apply_policy_constraint,
    search_viability, select_policy_action,
)


def field(positions=(), velocities=(), halves=(), **kwargs):
    count = len(positions)
    return DangerField(0, np.asarray(positions).reshape(-1, 2),
                       np.asarray(velocities).reshape(-1, 2),
                       np.asarray(halves).reshape(-1, 2),
                       np.zeros(count, dtype=int), np.full(count, 1000),
                       config=ForecastConfig(uncertainty_per_frame=0), **kwargs)


class ViabilityTests(unittest.TestCase):
    def test_empty_default_90_frames_policy_not_teacher_ranking(self):
        logits = np.arange(18, dtype=float)
        original = logits.copy()
        result = search_viability(field(), (192., 300.), (0, 0), logits)
        self.assertEqual(result.status, 'feasible')
        self.assertTrue(all(result.witness_mask))
        self.assertFalse(any(result.known_negative_mask))
        self.assertEqual(result.chosen_candidate, 17)
        self.assertEqual(len(result.verified_route.positions), 91)
        np.testing.assert_array_equal(logits, original)
        self.assertLessEqual(result.frame_checks, 500000)

    def test_safe_underfoot_but_wide_wall_demands_early_escape(self):
        # A 304px-wide wall descends toward a player near the bottom clamp.
        # Staying/going left is safe NOW, but cannot reach either end in time.
        danger = field([(120, 300)], [(0, 4)], [(152, 8)])
        self.assertGreater(danger.clearance(192, 400, 0), 80)
        logits = np.zeros(18)
        logits[7] = 100  # locally safe left; wrong macro direction
        logits[3] = 90   # right is a witnessed way around the wall's end
        config = ViabilityConfig(horizon_frames=40, beam_width=12, max_frame_checks=1000000)
        result = search_viability(danger, (192., 400.), (0, 0), logits, config=config)
        self.assertEqual(result.status, 'feasible')
        self.assertTrue(result.witness_mask[3])
        self.assertFalse(result.witness_mask[7])  # NOT a certified negative
        self.assertEqual(result.chosen_candidate, 3)
        self.assertFalse(any(result.known_negative_mask))
        self.assertTrue(all(value > 0 for value in result.route.clearances))

    def test_unknown_coverage_model_clipping_calibration_do_not_mask(self):
        logits = np.arange(18)
        for kwargs in ({'coverage_complete': False}, {'model_known': False},
                       {'t_offset': 1},
                       {'config': ViabilityConfig(horizon_frames=181)}):
            with self.subTest(kwargs=kwargs):
                result = search_viability(field(), (192., 300.), (0, 0), logits, **kwargs)
                self.assertEqual(result.status, 'unknown')
                self.assertTrue(all(result.constraint_mask))
                self.assertEqual(result.chosen_candidate, 17)
        result = search_viability(field(dropped_over_cap=1), (192., 300.), (0, 0), logits)
        self.assertEqual(result.status, 'unknown')
        self.assertTrue(all(result.constraint_mask))

    def test_budget_never_hard_masks(self):
        for budget in (0, 1, 25, 500):
            result = search_viability(field(), (192., 300.), (0, 0), np.arange(18),
                                      config=ViabilityConfig(max_frame_checks=budget))
            self.assertEqual(result.status, 'budget')
            self.assertTrue(all(result.constraint_mask))
            self.assertEqual(result.chosen_candidate, 17)
            self.assertLessEqual(result.frame_checks, budget)

    def test_collision_in_transition_even_if_endpoint_clear(self):
        # Previous right motion hits a tiny obstacle on settle frame 1; requested
        # up ends the 6-frame transition far from it, which cannot undo collision.
        danger = field([(196, 300)], [(0, 0)], [(0.5, 0.5)])
        self.assertGreater(danger.clearance(192, 280, 6), 0)
        result = search_viability(danger, (192., 300.), (0, 3), np.zeros(18),
                                  config=ViabilityConfig(horizon_frames=6))
        self.assertEqual(result.status, 'unknown')
        self.assertFalse(any(result.witness_mask))
        self.assertTrue(all(result.constraint_mask))
        self.assertIn('not_infeasibility', result.reasons[0])

    def test_every_frame_route_replay_settle_bounds_and_offset(self):
        model = MoveModel(axis=2, diagonal=2 / np.sqrt(2), focus_axis=1,
                          focus_diagonal=1 / np.sqrt(2))
        danger = field()
        config = ViabilityConfig(horizon_frames=13, frameskip=5, beam_width=3)
        result = search_viability(danger, (375., 431.), (0, 3), np.arange(18),
                                  config=config, model=model, t_offset=10)
        for first, route in enumerate(result.routes):
            self.assertIsNotNone(route)
            self.assertEqual(route.first_action, first)
            self.assertEqual(route.actions[0], first)
            x, y = route.positions[0]
            previous = (0, 3)
            frame = 0
            for index in route.actions:
                action = ACTIONS[index]
                for step in range(min(config.frameskip, config.horizon_frames - frame)):
                    effective = previous if step == 0 else action
                    x, y = model.apply(x, y, *effective, frames=1)
                    frame += 1
                    self.assertEqual(route.positions[frame], (x, y))
                    self.assertEqual(route.clearances[frame-1], danger.clearance(x, y, 10+frame))
                previous = action
            self.assertEqual(frame, 13)
            self.assertEqual(route.t_offset, 10)
        self.assertEqual(result.route.positions[1], (376., 431.))

    def test_nonzero_offset_is_used_for_collision(self):
        danger = field([(196, 300)], [(0, 0)], [(0.5, 0.5)])
        danger.spawn[:] = 11
        config = ViabilityConfig(horizon_frames=1)
        early = search_viability(danger, (192., 300.), (0, 3), np.zeros(18), config=config)
        later = search_viability(danger, (192., 300.), (0, 3), np.zeros(18),
                                 config=config, t_offset=10)
        self.assertEqual(early.status, 'feasible')
        self.assertEqual(later.status, 'unknown')

    def test_action_order_and_auxiliary_helper_are_unchanged(self):
        self.assertEqual(ACTIONS, tuple((f, m) for f in (0, 1) for m in range(9)))
        logits = np.zeros(18)
        logits[12] = 3
        result = search_viability(field(), (192., 300.), (0, 0), logits,
                                  config=ViabilityConfig(horizon_frames=2))
        shoot, bomb, helper = object(), object(), object()
        action = (0, 0, shoot, bomb, helper)
        changed = apply_policy_constraint(action, result)
        self.assertEqual(changed[:2], (1, 3))
        self.assertEqual(action[:2], (0, 0))
        for actual, original in zip(changed[2:], action[2:]):
            self.assertIs(actual, original)
        self.assertEqual(select_policy_action(logits, result.constraint_mask), 12)

    def test_spatial_diversity_retains_tighter_remote_exit(self):
        nodes = [_Node(((x, 300.),), (clearance,), (0,), 0, clearance)
                 for x, clearance in ((100., 100.), (101., 99.), (102., 98.), (200., 1.))]
        kept = _diverse(nodes, ViabilityConfig(beam_width=2), None)
        self.assertEqual({n.positions[-1][0] for n in kept}, {100., 200.})

    def test_unknown_clearance_propagates(self):
        danger = field()
        danger.clearance_many = lambda *args, **kwargs: None
        result = search_viability(danger, (192., 300.), (0, 0), np.arange(18))
        self.assertEqual(result.status, 'unknown')
        self.assertTrue(all(result.constraint_mask))

    def test_declared_exit_deadline_requires_early_lateral_action(self):
        # Locally empty space, but caller requires the declared right-hand exit
        # by frame 6. Idle first cannot cover 20px after its 6-frame commitment.
        logits = np.zeros(18)
        logits[0], logits[3] = 100, 90
        config = ViabilityConfig(horizon_frames=18, frameskip=6,
                                  target_radius_px=0, target_deadline_frames=6)
        result = search_viability(field(), (192., 400.), (0, 0), logits,
                                  config=config, target=(212., 400.))
        self.assertEqual(result.status, 'feasible')
        self.assertFalse(result.witness_mask[0])
        self.assertTrue(result.witness_mask[3])
        self.assertEqual(result.chosen_candidate, 3)
        self.assertEqual(result.route.arrival_frame, 6)
        self.assertEqual(result.route.positions[6], (212., 400.))
        self.assertEqual(len(result.route.positions), 19)
        self.assertFalse(any(result.known_negative_mask))
        # Same target without deadline is only ranking, so idle remains positive.
        no_goal = search_viability(field(), (192., 400.), (0, 0), logits,
                                   config=ViabilityConfig(horizon_frames=18),
                                   target=(212., 400.))
        self.assertTrue(all(no_goal.witness_mask))
        self.assertEqual(no_goal.chosen_candidate, 0)
        self.assertIsNone(no_goal.route.arrival_frame)

    def test_goal_arrival_must_have_safe_successor_to_horizon(self):
        danger = field([(192, 300)], [(0, 0)], [(1000, 1000)])
        danger.spawn[:] = 7  # all positions killed AFTER reaching exit at frame 6
        result = search_viability(danger, (192., 400.), (0, 0), np.zeros(18),
                                  config=ViabilityConfig(horizon_frames=12,
                                      target_radius_px=0, target_deadline_frames=6),
                                  target=(212., 400.))
        self.assertEqual(result.status, 'unknown')
        self.assertFalse(any(result.witness_mask))
        self.assertTrue(all(result.constraint_mask))

    def test_goal_origin_and_deadline_validation(self):
        config = ViabilityConfig(horizon_frames=6, target_deadline_frames=0)
        result = search_viability(field(), (192., 300.), (0, 0), np.zeros(18),
                                  config=config, target=(192., 300.))
        self.assertTrue(all(result.witness_mask))
        self.assertEqual(result.route.arrival_frame, 0)
        missed = search_viability(field(), (192., 300.), (0, 0), np.zeros(18),
                                  config=config, target=(212., 300.))
        self.assertEqual(missed.status, 'unknown')
        with self.assertRaises(ValueError):
            search_viability(field(), (192., 300.), (0, 0), np.zeros(18), config=config)
        for kwargs in ({'target_deadline_frames': -1}, {'target_deadline_frames': 91},
                       {'target_radius_px': -1}, {'target_deadline_frames': 1.5}):
            with self.assertRaises(ValueError):
                ViabilityConfig(**kwargs)

    def test_arrived_branch_survives_beam_one_even_after_leaving_goal(self):
        arrived = _Node(((240., 300.),), (1.,), (3,), 3, 1., arrival_frame=6)
        near = _Node(((212., 300.),), (100.,), (0,), 0, 100.)
        config = ViabilityConfig(beam_width=1, target_deadline_frames=6)
        self.assertEqual(_diverse([near, arrived], config, (212., 300.)), [arrived])

    def test_configuration_and_policy_validation(self):
        for kwargs in ({'frameskip': 0}, {'horizon_frames': 0}, {'beam_width': 0},
                       {'spatial_cell_px': 0}, {'max_frame_checks': -1}):
            with self.assertRaises(ValueError):
                ViabilityConfig(**kwargs)
        with self.assertRaises(ValueError):
            select_policy_action(np.zeros(18), np.zeros(18, dtype=bool))


if __name__ == '__main__':
    unittest.main()
