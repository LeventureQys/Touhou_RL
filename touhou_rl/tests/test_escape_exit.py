"""Exit analysis and dead-end veto tests (D4b)."""

from __future__ import annotations

import numpy as np

from touhou_rl.escape import exit as exit_module
from touhou_rl.escape import families, forecast, planner, synth

START = (192.0, 384.0)


def _closing_field():
    """A wall with a gap; the region beyond the gap fills from frame 16."""

    wall = tuple(
        synth.HazardSpec((210.0, y), (0.0, 0.0), (2.0, 2.0))
        for y in list(range(16, 378, 4)) + list(range(392, 436, 4))
    )
    fill = tuple(
        synth.HazardSpec((x, y), (0.0, 0.0), (8.0, 8.0), spawn_frame=16)
        for x in range(224, 380, 16)
        for y in range(16, 436, 16)
    )
    scenario = synth.Scenario("exit/0", "TEST", 0, START, wall + fill,
                              horizon_frames=80, hold_frames=0)
    return scenario, forecast.DangerField.from_scenario(scenario, 0)


def test_open_field_has_no_closure():
    scenario = synth.Scenario("open/0", "TEST", 0, START, (), horizon_frames=80, hold_frames=0)
    field = forecast.DangerField.from_scenario(scenario, 0)
    report = exit_module.analyze(field, *START, 10)
    assert report.closure_eta is None
    assert report.component_size > 0


def test_closing_region_is_detected_and_open_region_is_not():
    _scenario, field = _closing_field()
    closed = exit_module.analyze(field, 230.0, 384.0, 8)
    open_side = exit_module.analyze(field, 150.0, 384.0, 8)
    assert closed.closure_eta is not None
    assert closed.closure_eta >= 16
    assert closed.closes_inside_horizon
    assert open_side.closure_eta is None


def test_rank_prefers_later_closure():
    early = exit_module.ExitReport((0, 0), 10, 2560.0, 20, 180)
    late = exit_module.ExitReport((0, 0), 5, 1280.0, None, 180)
    assert exit_module.rank_key(late) > exit_module.rank_key(early)


def test_planner_vetoes_a_dead_end_when_an_alternative_exists(monkeypatch):
    scenario, field = _closing_field()
    config = planner.PlannerConfig(horizon_decisions=6, continuation_decisions=1, beam_width=1)
    model = planner.geometry.MoveModel()
    east = (0, 3)
    west = (0, 7)

    def fake_search(field_arg, start, previous, action, config_arg, model_arg, total, t_offset, **kwargs):
        if action not in (east, west):
            return None, "collision", 1
        path_actions = tuple([action] * 6)
        # Exact clearances are not needed: this test targets the ranking/veto path.
        clearances = tuple([10.0] * (6 * config_arg.frameskip))
        knots = planner._knots(field_arg, start, path_actions, config_arg, previous, model_arg)
        route = planner.Route(
            first=action,
            actions=path_actions,
            clearances=clearances,
            min_clearance=10.0,
            end_clearance=10.0,
            switches=0,
        )
        assert len(knots) == 13
        return route, "", 7

    monkeypatch.setattr(planner, "_search_first_action", fake_search)
    result = planner.plan(field, START, config=config)
    assert result.status is planner.SolutionStatus.FEASIBLE
    assert result.route is not None and result.route.first == west
    east_index = planner.ACTIONS.index(east)
    assert "dead_end" in result.first_action_reason[east_index]
    assert result.record.closure_eta is None  # the chosen (west) region stays open


def test_planner_records_closure_eta_for_a_closing_route(monkeypatch):
    scenario, field = _closing_field()
    config = planner.PlannerConfig(horizon_decisions=6, continuation_decisions=1, beam_width=1)
    east = (0, 3)

    def fake_search(field_arg, start, previous, action, config_arg, model_arg, total, t_offset, **kwargs):
        if action != east:
            return None, "collision", 1
        route = planner.Route(
            first=action,
            actions=tuple([action] * 6),
            clearances=tuple([10.0] * (6 * config_arg.frameskip)),
            min_clearance=10.0,
            end_clearance=10.0,
            switches=0,
        )
        return route, "", 7

    monkeypatch.setattr(planner, "_search_first_action", fake_search)
    result = planner.plan(field, START, config=config)
    assert result.status is planner.SolutionStatus.FEASIBLE
    assert result.record.closure_eta is not None
    assert result.record.closure_eta >= 16


def test_families_provide_a_long_horizon_scenario():
    scenario = families.build("SC-16", 0)
    field = forecast.DangerField.from_scenario(scenario, 0)
    # The gap closes inside the 180-frame window, so the exit analysis must see it.
    assert field.config.horizon_frames == 180


def test_playable_mask_stops_routing_around_a_wall():
    """E-05's wall spans the movement area; the coarse layer must not slip past it."""

    scenario = families.edge_instances()["E-05"]
    field = forecast.DangerField.from_scenario(scenario, 0)
    table = exit_module.closure_map(field)
    reachable = exit_module.coarse_reachable(field, scenario.player_start)
    reachable_max = int(table[reachable].max()) if reachable.any() else -1
    assert reachable_max < int(table.max())


def test_adaptive_mode_classification():
    cases = {
        "E-05": ("clearance", families.edge_instances()["E-05"]),
        "E-06": ("clearance", families.edge_instances()["E-06"]),
        "SC-15": ("freedom", families.build("SC-15", 0)),
        "SC-16": ("freedom", families.build("SC-16", 0)),
    }
    for name, (expected, scenario) in cases.items():
        field = forecast.DangerField.from_scenario(scenario, 0)
        table = exit_module.closure_map(field)
        mode, diagnostics = exit_module.adaptive_mode(field, scenario.player_start, table)
        assert mode == expected, f"{name}: {mode} != {expected} ({diagnostics})"
        assert "criterion" in diagnostics


def test_sc15_plan_moves_toward_the_safe_mouth_instead_of_idling():
    """Regression: identical route ends used to leave the sort on ACTIONS[0] = idle."""

    from dataclasses import replace

    scenario = families.build("SC-15", 0)
    field = forecast.DangerField.from_scenario(scenario, 0)
    config = planner.PlannerConfig(horizon_decisions=12, continuation_decisions=2, beam_width=1)
    for mode in ("freedom", "clearance"):
        result = planner.plan(field, scenario.player_start, config=replace(config, score_mode=mode))
        first = result.record.action_sequence[0]
        assert first[0] != 0, f"{mode} idled on its first decision"
        assert first[0] in (2, 3), f"{mode} did not head east: {first}"


def test_rank_key_prefers_future_freedom():
    from touhou_rl.escape.exit import ExitReport, rank_key

    pocket = ExitReport(cell=(0, 0), component_size=10, free_area_px2=2560.0,
                        closure_eta=154, horizon_frames=180, future_freedom=154)
    safe = ExitReport(cell=(1, 1), component_size=10, free_area_px2=2560.0,
                      closure_eta=170, horizon_frames=180, future_freedom=181)
    assert rank_key(safe) > rank_key(pocket)

