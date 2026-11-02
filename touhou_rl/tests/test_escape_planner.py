"""Planner tests: per-first-action status, proofs vs beams, replay validity."""

from __future__ import annotations

import time

import pytest

from touhou_rl.escape import contracts, families, forecast, planner, synth


def _config(**overrides):
    base = dict(horizon_decisions=4, continuation_decisions=1, beam_width=1)
    base.update(overrides)
    return planner.PlannerConfig(**base)


def _field(scenario, config=None):
    return forecast.DangerField.from_scenario(scenario, 0, config or forecast.ForecastConfig())


def _edge(name):
    return families.edge_instances()[name]


def _to_routing(action):
    move, _shoot, focus, _bomb = action
    return (int(focus), int(move))


def test_empty_field_is_feasible_and_contract_valid():
    scenario = synth.Scenario("empty/0", "TEST", 0, (192.0, 384.0), (),
                              horizon_frames=40, hold_frames=0)
    result = planner.plan(_field(scenario), scenario.player_start, config=_config())
    assert result.status is contracts.SolutionStatus.FEASIBLE
    assert result.route is not None
    document = contracts.to_json_dict(result.record)
    assert contracts.validate_document(document) == []
    assert len(result.first_action_status) == 18
    assert len(result.record.action_sequence) == 4


def test_immediate_overlap_is_a_proof_not_a_beam_failure():
    scenario = synth.Scenario("overlap/0", "TEST", 0, (192.0, 384.0),
                              (synth.HazardSpec((192.0, 384.0), (0.0, 0.0), (4.0, 4.0)),),
                              horizon_frames=20, hold_frames=0)
    result = planner.plan(_field(scenario), scenario.player_start, config=_config())
    assert result.status is contracts.SolutionStatus.NO_SOLUTION_CERTIFIED
    assert result.route is None
    assert set(result.first_action_status) == {contracts.FirstActionStatus.IMMEDIATE_COLLISION.value}
    assert contracts.validate_document(contracts.to_json_dict(result.record)) == []


def test_over_cap_field_is_unknown_not_feasible():
    hazards = [synth.HazardSpec((60.0 + 12.0 * index, 300.0), (0.0, 0.0), (2.0, 2.0))
               for index in range(12)]
    scenario = synth.Scenario("cap/0", "TEST", 0, (192.0, 384.0), tuple(hazards),
                              horizon_frames=40, hold_frames=0)
    field = _field(scenario, forecast.ForecastConfig(max_entities=3))
    assert field.dropped_over_cap > 0
    result = planner.plan(field, scenario.player_start, config=_config())
    assert result.status is contracts.SolutionStatus.MODEL_UNKNOWN
    assert any("coverage_incomplete" in reason for reason in result.reasons)


def test_routes_replay_without_collision():
    for name in ("E-09", "E-17"):
        scenario = _edge(name)
        result = planner.plan(_field(scenario), scenario.player_start, config=_config())
        assert result.status is contracts.SolutionStatus.FEASIBLE, name
        actions = [_to_routing(action) for action in result.record.action_sequence]
        outcome = synth.simulate(scenario, actions)
        assert outcome["hit"] is False, name


def test_budget_is_reported_as_unknown_not_blocked():
    scenario = _edge("E-01")
    result = planner.plan(_field(scenario), scenario.player_start,
                          config=_config(max_nodes=1))
    assert result.status in (contracts.SolutionStatus.BUDGET_EXHAUSTED,
                             contracts.SolutionStatus.NO_SOLUTION_CERTIFIED)
    if result.status is contracts.SolutionStatus.BUDGET_EXHAUSTED:
        assert any("budget" in reason for reason in result.reasons)


def test_closed_loop_survives_an_open_field():
    scenario = families.build("SC-17", 0)
    outcome = planner.run_closed_loop(scenario, plan_config=_config())
    assert outcome["hit"] is False
    assert outcome["survived"] is True
    assert outcome["fallbacks"] == 0


def test_closed_loop_is_deterministic():
    scenario = families.build("SC-17", 0)
    first = planner.run_closed_loop(scenario, plan_config=_config())
    second = planner.run_closed_loop(scenario, plan_config=_config())
    for payload in (first, second):
        payload.pop("plan_ms_p50")  # wall-clock timing is not part of determinism
    assert first == second


def test_planning_is_fast_enough_for_a_short_horizon():
    scenario = _edge("E-17")
    field = _field(scenario)
    started = time.perf_counter()
    planner.plan(field, scenario.player_start, config=_config())
    assert (time.perf_counter() - started) < 2.0


def test_plan_emits_a_corridor_inside_the_move_area():
    scenario = _edge("E-09")
    field = _field(scenario)
    result = planner.plan(field, scenario.player_start, config=_config())
    corridor = result.record.corridor_by_time
    assert corridor
    model = planner.geometry.MoveModel()
    for box in corridor:
        assert box.x0 >= model.x_min - 1e-6 and box.x1 <= model.x_max + 1e-6
        assert box.y0 >= model.y_min - 1e-6 and box.y1 <= model.y_max + 1e-6
        assert box.x0 <= box.x1 and box.y0 <= box.y1


def test_corridor_preference_keeps_the_chosen_route_inside():
    scenario = _edge("E-09")
    field = _field(scenario)
    first = planner.plan(field, scenario.player_start, config=_config())
    corridor = first.record.corridor_by_time
    again = planner.plan(field, scenario.player_start, config=_config(), corridor=corridor)
    assert again.record.route_id == first.record.route_id
    """Re-deciding freedom/clearance every decision made SC-16 oscillate and die."""

    scenario = families.build("SC-16", 0)
    config = planner.PlannerConfig(
        horizon_decisions=12, continuation_decisions=2, beam_width=1, score_mode="auto"
    )
    outcome = planner.run_closed_loop(scenario, plan_config=config)
    assert outcome["survived"] is True
    assert outcome["fallbacks"] == 0
