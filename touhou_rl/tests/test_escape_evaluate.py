"""Evaluation and gate-arithmetic tests (D7 dev slice)."""

from __future__ import annotations

import numpy as np

from touhou_rl.escape import evaluate, families, synth


def _row(scenario_id, *, survived, hit, baseline_hit, expected=None, reference=None, pocket=0):
    return {
        "scenario_id": scenario_id,
        "family": scenario_id.split("/")[0],
        "expected": expected,
        "reference": reference,
        "subject": {
            "scenario_id": scenario_id, "arm": "subject", "survived": survived, "hit": hit,
            "hit_frame": None, "frames": 20, "min_clearance": 1.0,
            "entered_pocket": False, "evacuate_frame": None, "pocket_frames": pocket,
        },
        "baseline": (
            None
            if baseline_hit is None
            else {
                "scenario_id": scenario_id, "arm": "baseline", "survived": not baseline_hit,
                "hit": baseline_hit, "hit_frame": None, "frames": 20, "min_clearance": 0.5,
                "entered_pocket": False, "evacuate_frame": None, "pocket_frames": 0,
            }
        ),
    }


def test_a01_requires_all_boundary_instances():
    rows = [
        _row("E-01", survived=True, hit=False, baseline_hit=False),
        _row("E-02", survived=True, hit=False, baseline_hit=False),
        _row("E-05", survived=True, hit=False, baseline_hit=False),
        _row("E-06", survived=False, hit=True, baseline_hit=True),
    ]
    report = evaluate.gates({"rows": rows})
    assert report["A-01"]["status"] == "pass"
    rows[2] = _row("E-05", survived=False, hit=True, baseline_hit=False)
    assert evaluate.gates({"rows": rows})["A-01"]["status"] == "fail"


def test_a04_is_underpowered_without_baseline_deaths():
    rows = [_row("E-09", survived=True, hit=False, baseline_hit=False)]
    assert evaluate.gates({"rows": rows})["A-04"]["status"] == "underpowered"


def test_a04_uses_relative_reduction():
    rows = [
        _row("A", survived=True, hit=False, baseline_hit=True),
        _row("B", survived=True, hit=False, baseline_hit=True),
        _row("C", survived=True, hit=False, baseline_hit=True),
        _row("D", survived=False, hit=True, baseline_hit=True),
    ]
    report = evaluate.gates({"rows": rows})["A-04"]
    assert report["baseline_deaths"] == 4 and report["subject_deaths"] == 1
    assert report["relative_reduction"] == 0.75
    assert report["status"] == "pass"


def test_a02_denominator_is_reference_feasible_only():
    rows = [
        _row("F1", survived=True, hit=False, baseline_hit=False, reference="feasible"),
        _row("F2", survived=False, hit=True, baseline_hit=True, reference="feasible"),
        _row("F3", survived=False, hit=True, baseline_hit=True, reference="budget_exhausted"),
        _row("F4", survived=False, hit=True, baseline_hit=True, reference="model_unknown"),
    ]
    report = evaluate.gates({"rows": rows})["A-02"]
    assert report["denominator"] == 2
    assert report["rate"] == 0.5
    assert report["status"] == "fail"


def test_a07_reads_the_pocket_counter():
    rows = [_row("SC-15/0", survived=True, hit=False, baseline_hit=False, pocket=5)]
    assert evaluate.gates({"rows": rows})["A-07"]["status"] == "fail"


def test_a05_rejects_survival_on_must_not_pass_instances():
    rows = [
        _row("E-06", survived=False, hit=True, baseline_hit=True, expected="infeasible_unproven"),
        _row("E-11", survived=False, hit=True, baseline_hit=True, expected="blocked"),
    ]
    assert evaluate.gates({"rows": rows})["A-05"]["status"] == "pass"
    rows[1] = _row("E-11", survived=True, hit=False, baseline_hit=False, expected="blocked")
    report = evaluate.gates({"rows": rows})["A-05"]
    assert report["status"] == "fail"
    assert report["survived_anyway"] == ["E-11"]


def test_a06_requires_certification_for_blocked_and_silence_for_unproven():
    def row(scenario_id, expected, status, reason=""):
        base = _row(scenario_id, survived=False, hit=True, baseline_hit=True, expected=expected)
        base["subject"]["final_status"] = status
        base["subject"]["final_reason"] = reason
        return base

    rows = [
        row("E-11", "blocked", "no_solution_certified", "coverage_proof: every lattice point is covered"),
        row("E-06", "infeasible_unproven", "budget_exhausted", "no_route_found_within_beam"),
    ]
    assert evaluate.gates({"rows": rows})["A-06"]["status"] == "pass"
    rows[0] = row("E-11", "blocked", "budget_exhausted", "no_route_found_within_beam")
    assert evaluate.gates({"rows": rows})["A-06"]["status"] == "fail"
    # A late local proof on an unproven instance is legitimate.
    rows = [
        row("E-11", "blocked", "no_solution_certified", "coverage_proof"),
        row("E-06", "infeasible_unproven", "no_solution_certified", "all_first_actions_collide: 18/18 die"),
    ]
    assert evaluate.gates({"rows": rows})["A-06"]["status"] == "pass"
    # A bare search failure reported as certified is not.
    rows[1] = row("E-06", "infeasible_unproven", "no_solution_certified", "no_route_found_within_beam")
    assert evaluate.gates({"rows": rows})["A-06"]["status"] == "fail"


def test_a03_is_underpowered_without_last_safe_start_labels():
    rows = [_row("SC-16/0", survived=True, hit=False, baseline_hit=True)]
    report = evaluate.gates({"rows": rows})["A-03"]
    assert report["status"] == "underpowered"
    assert report["denominator"] == 0


def test_baseline_arm_runs_on_the_synthetic_world():
    scenario = synth.Scenario("empty/0", "TEST", 0, (192.0, 384.0), (),
                              horizon_frames=20, hold_frames=0)
    result = evaluate.run_baseline(scenario)
    assert result.arm == "baseline"
    assert result.survived is True
    assert result.frames == 20


def test_evaluate_marks_reference_status_per_row():
    scenario = families.edge_instances()["E-17"]
    document = evaluate.evaluate(
        [scenario], include_baseline=True, reference_status={scenario.scenario_id: "feasible"}
    )
    row = document["rows"][0]
    assert row["reference"] == "feasible"
    assert row["subject"]["survived"] is True


def test_over_cap_field_is_certified_by_a_coverage_proof():
    """E-11's fill exceeds the field cap; the proof must use every entity."""

    from touhou_rl.escape import forecast, planner

    scenario = families.edge_instances()["E-11"]
    field = forecast.DangerField.from_scenario(scenario, 0)
    assert field.dropped_over_cap > 0
    result = planner.plan(
        field, scenario.player_start,
        config=planner.PlannerConfig(horizon_decisions=4, continuation_decisions=1, beam_width=1),
    )
    assert result.status.value == "no_solution_certified"
    assert "coverage_proof" in result.reasons[0]
    assert "frame 6" in result.reasons[0]


def test_coverage_proof_respects_spawn_frames():
    from touhou_rl.escape import forecast, planner

    hazards = tuple(
        synth.HazardSpec((16.0 + x, 20.0 + y), (0.0, 0.0), (8.0, 8.0), spawn_frame=10)
        for x in range(0, 368, 12)
        for y in range(0, 432, 12)
    )
    scenario = synth.Scenario("fill10/0", "TEST", 0, (192.0, 384.0), hazards,
                              horizon_frames=40, hold_frames=0)
    field = forecast.DangerField.from_scenario(scenario, 0)
    assert planner.coverage_proof(field, planner.PlannerConfig()) is not None
    assert planner.coverage_proof(field, planner.PlannerConfig()) >= 10
