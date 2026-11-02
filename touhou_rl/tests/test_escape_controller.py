"""Controller tests: intent transitions, unknown handling, bomb arbitration, reset."""

from __future__ import annotations

import numpy as np
import pytest

from touhou_rl.escape import contracts, controller, families, forecast, planner, synth


def _closing_area_scenario() -> synth.Scenario:
    hazards = tuple(
        synth.HazardSpec((150.0 + x, 300.0 + y), (0.0, 0.0), (6.0, 6.0), spawn_frame=30)
        for x in range(0, 110, 10)
        for y in range(0, 150, 10)
    )
    return synth.Scenario("closing/0", "TEST", 0, (192.0, 384.0), hazards,
                          horizon_frames=40, hold_frames=0)


def _field(scenario, config=None):
    return forecast.DangerField.from_scenario(scenario, 0, config or forecast.ForecastConfig())


def test_empty_field_stays_local_and_validates():
    scenario = synth.Scenario("empty/0", "TEST", 0, (192.0, 384.0), (),
                              horizon_frames=20, hold_frames=0)
    field = _field(scenario)
    config = controller.ControllerConfig(
        planner=planner.PlannerConfig(horizon_decisions=4, continuation_decisions=1, beam_width=1)
    )
    ctrl = controller.EscapeController(config)
    decision, _ = ctrl.decide(x=192.0, y=384.0, frame=0, previous_action=(0, 0), field=field)
    assert decision.intent is contracts.Intent.LOCAL
    assert contracts.validate_document(contracts.to_json_dict(decision)) == []


def test_intent_evacuates_then_returns_to_local():
    scenario = _closing_area_scenario()
    outcome = controller.run_episode(
        scenario,
        config=controller.ControllerConfig(
            planner=planner.PlannerConfig(horizon_decisions=8, continuation_decisions=2, beam_width=1)
        ),
    )
    assert outcome["survived"] is True
    assert outcome["intents"].get("evacuate", 0) > 0
    assert outcome["intents"].get("local", 0) > 0
    # The evacuation must come first: it starts while the area is still open.
    first = outcome["decisions"][0]
    assert first["intent"] == "evacuate"


def test_unusable_observation_holds_and_reports_unknown():
    scenario = synth.Scenario("empty/0", "TEST", 0, (192.0, 384.0), (),
                              horizon_frames=20, hold_frames=0)
    field = _field(scenario)
    ctrl = controller.EscapeController()
    decision, _ = ctrl.decide(
        x=192.0, y=384.0, frame=0, previous_action=(3, 1), field=field,
        observation_ok=False, observation_reason="frame: state and pool never agreed",
    )
    assert decision.intent is contracts.Intent.UNKNOWN
    assert decision.fallback == "hold_previous_action"
    assert decision.accepted_action == (1, 1, 3, 0)  # move, shoot, focus, bomb
    assert "observation unusable" in decision.reason


def test_bomb_arbiter_fires_once_and_then_cools_down():
    """The synthetic env has no deathbomb window, so the arbiter is tested directly."""

    ctrl = controller.EscapeController(controller.ControllerConfig(bomb_cooldown=3))
    fired, reason = ctrl._maybe_bomb(-0.5, bombs=2)
    assert fired is True and "clearance" in (reason or "")
    again, _ = ctrl._maybe_bomb(-0.5, bombs=2)
    assert again is False  # cooldown
    # No resource -> no bomb, and no cooldown is consumed.
    fresh = controller.EscapeController(controller.ControllerConfig(bomb_cooldown=3))
    assert fresh._maybe_bomb(-0.5, bombs=0) == (False, None)
    assert fresh._maybe_bomb(1.0, bombs=2) == (False, None)


def test_decision_inside_a_hazard_emerges_with_bomb():
    hazard = synth.HazardSpec((192.0, 384.0), (0.0, 0.0), (4.0, 4.0))
    scenario = synth.Scenario("inside/0", "TEST", 0, (192.0, 384.0), (hazard,),
                              horizon_frames=10, hold_frames=0)
    field = _field(scenario)
    ctrl = controller.EscapeController()
    decision, _ = ctrl.decide(x=192.0, y=384.0, frame=0, previous_action=(0, 0), field=field, bombs=2)
    assert decision.intent is contracts.Intent.EMERGENCY
    assert decision.emergency_override is True
    assert decision.accepted_action[3] == 1
    assert "bomb" in decision.fallback
    assert contracts.validate_document(contracts.to_json_dict(decision)) == []


def test_reset_clears_intent_route_and_cooldown():
    scenario = _closing_area_scenario()
    field = _field(scenario)
    ctrl = controller.EscapeController(
        controller.ControllerConfig(
            planner=planner.PlannerConfig(horizon_decisions=4, continuation_decisions=1, beam_width=1)
        )
    )
    ctrl.decide(x=192.0, y=384.0, frame=0, previous_action=(0, 0), field=field)
    assert ctrl.state.intent is contracts.Intent.EVACUATE
    assert ctrl.state.plan is not None
    ctrl.reset()
    assert ctrl.state.intent is contracts.Intent.LOCAL
    assert ctrl.state.plan is None and ctrl.state.route is None
    assert ctrl.state.bomb_cooldown == 0 and ctrl.state.decision_id == 0


def test_verify_first_action_rejects_an_immediate_collision():
    hazard = synth.HazardSpec((196.0, 384.0), (0.0, 0.0), (4.0, 4.0))
    scenario = synth.Scenario("hit/0", "TEST", 0, (192.0, 384.0), (hazard,),
                              horizon_frames=20, hold_frames=0)
    field = _field(scenario)
    ctrl = controller.EscapeController()
    assert ctrl._verify_first_action(field, 192.0, 384.0, 0, (0, 3), (0, 0)) is False


def test_route_continuity_follows_the_committed_actions_and_reports_it():
    """A committed route is followed while it re-verifies; reasons say "continuing"."""

    scenario = _closing_area_scenario()
    field = _field(scenario)
    ctrl = controller.EscapeController(
        controller.ControllerConfig(
            continuation_decisions=3,
            planner=planner.PlannerConfig(horizon_decisions=8, continuation_decisions=2, beam_width=1),
        )
    )
    decisions = []
    x, y, previous = 192.0, 384.0, (0, 0)
    for frame in range(0, 8, 2):
        decision, _ = ctrl.decide(x=x, y=y, frame=frame, previous_action=previous, field=field)
        decisions.append(decision.reason)
        action = decision.accepted_action
        previous = (action[2], action[0])
    assert any("continuing" in reason for reason in decisions)
    # With continuation off, every decision is a fresh route.
    fresh = controller.EscapeController(
        controller.ControllerConfig(
            continuation_decisions=1,
            planner=planner.PlannerConfig(horizon_decisions=8, continuation_decisions=2, beam_width=1),
        )
    )
    fresh_reasons = []
    x, y, previous = 192.0, 384.0, (0, 0)
    for frame in range(0, 8, 2):
        decision, _ = fresh.decide(x=x, y=y, frame=frame, previous_action=previous, field=field)
        fresh_reasons.append(decision.reason)
        action = decision.accepted_action
        previous = (action[2], action[0])
    assert all("new route" in reason or "route survives" in reason for reason in fresh_reasons)


def test_stale_observation_is_not_used_for_a_decision():
    """A-01 also asks for a missing/late-frame test: age > bound must be UNKNOWN."""

    scenario = _closing_area_scenario()
    field = _field(scenario)
    ctrl = controller.EscapeController(controller.ControllerConfig(max_observation_age=2))
    fresh, _ = ctrl.decide(x=192.0, y=384.0, frame=0, previous_action=(0, 0), field=field, observation_age=0)
    assert fresh.intent is not contracts.Intent.UNKNOWN

    stale, diagnostics = ctrl.decide(
        x=192.0, y=384.0, frame=2, previous_action=(0, 0), field=field, observation_age=5
    )
    assert stale.intent is contracts.Intent.UNKNOWN
    assert "stale observation" in stale.reason
    assert stale.fallback == "hold_previous_action"
    assert diagnostics["observation_age"] == 5
    assert contracts.validate_document(contracts.to_json_dict(stale)) == []


def test_deadline_miss_is_measured_not_hardcoded():
    """A-01 asks for a timeout test; the flag must follow the measured latency."""

    scenario = _closing_area_scenario()
    field = _field(scenario)
    slow = controller.EscapeController(
        controller.ControllerConfig(deadline_ms=0.001)
    )
    decision, _ = slow.decide(x=192.0, y=384.0, frame=0, previous_action=(0, 0), field=field)
    assert decision.deadline_miss is True
    assert decision.timing_breakdown["controller_ms"] > 0.001
    assert decision.timing_breakdown["budget_ms"] == 0.001

    fast = controller.EscapeController(controller.ControllerConfig(deadline_ms=10_000.0))
    decision, _ = fast.decide(x=192.0, y=384.0, frame=0, previous_action=(0, 0), field=field)
    assert decision.deadline_miss is False
    assert contracts.validate_document(contracts.to_json_dict(decision)) == []


def test_episode_is_deterministic():
    scenario = _closing_area_scenario()
    config = controller.ControllerConfig(
        planner=planner.PlannerConfig(horizon_decisions=4, continuation_decisions=1, beam_width=1)
    )
    first = controller.run_episode(scenario, config=config)
    second = controller.run_episode(scenario, config=config)
    assert first["decisions"] == second["decisions"]
    assert first["intents"] == second["intents"]


def test_controller_matches_planner_on_sc15_survival():
    scenario = families.build("SC-15", 0)
    outcome = controller.run_episode(
        scenario,
        config=controller.ControllerConfig(
            planner=planner.PlannerConfig(horizon_decisions=12, continuation_decisions=2, beam_width=1)
        ),
        max_decisions=30,
    )
    assert outcome["hit"] is False
